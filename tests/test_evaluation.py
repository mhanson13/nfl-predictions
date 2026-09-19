"""Tests for src/evaluation modules.

Covers:
  - calibrate_winprob._load_history
  - evaluate_predictions.find_prediction_files
  - evaluate_predictions.load_prediction_frames
  - evaluate_predictions.unify_predictions
  - evaluate_predictions.compute_metrics
  - evaluate_predictions.compute_overall_metrics
  - evaluate_predictions.PREDICTION_PATTERN
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation import calibrate_winprob as calibrate_module
from src.evaluation import evaluate_predictions as eval_module
from src.evaluation.calibrate_winprob import _load_history
from src.evaluation.evaluate_predictions import (
    PREDICTION_PATTERN,
    PredictionFrame,
    compute_metrics,
    compute_overall_metrics,
    find_prediction_files,
    load_prediction_frames,
    plot_metrics,
    summarize_underperformance,
    unify_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pred_csv(tmp_path: Path, filename: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / filename
    df.to_csv(p, index=False)
    return p


def _base_row(game_id="2023_01_KC_BUF", home="KC", away="BUF",
              hwp=0.65, margin=4.5, season=2023, week=1) -> dict:
    return {
        "game_id": game_id,
        "home_team": home,
        "away_team": away,
        "home_win_prob": hwp,
        "pred_home_margin": margin,
        "season": season,
        "week": week,
    }


def _make_pred_frame(game_id="2023_01_KC_BUF", hwp=0.65, margin=4.5,
                     home_margin=7.0, week=1, season=2023) -> pd.DataFrame:
    return pd.DataFrame([{
        "game_id": game_id,
        "home_team": "KC",
        "away_team": "BUF",
        "home_win_prob": hwp,
        "pred_home_margin": margin,
        "season": season,
        "week": week,
        "home_margin": home_margin,
        "actual_home_win": 1.0 if home_margin > 0 else (0.0 if home_margin < 0 else np.nan),
        "predicted_home_win": 1 if hwp >= 0.5 else 0,
        "prob_error_sq": (hwp - (1.0 if home_margin > 0 else 0.0)) ** 2,
        "margin_error": margin - home_margin,
        "abs_margin_error": abs(margin - home_margin),
        "sq_margin_error": (margin - home_margin) ** 2,
        "log_loss_component": (
            math.log(max(hwp, 1e-9)) if home_margin > 0 else math.log(max(1 - hwp, 1e-9))
        ),
        "misclassified": (1 if hwp >= 0.5 else 0) != (1 if home_margin > 0 else 0),
    }])


# ---------------------------------------------------------------------------
# PREDICTION_PATTERN
# ---------------------------------------------------------------------------

class TestPredictionPattern:
    @pytest.mark.parametrize("name", [
        "w1_predictions.csv",
        "w10_predictions.csv",
        "w12_predictions_full.csv",
        "W3_predictions.csv",
        "w7_predictions_history2023.csv",
    ])
    def test_valid_names_match(self, name):
        assert PREDICTION_PATTERN.match(name)

    @pytest.mark.parametrize("name", [
        "predictions.csv",
        "weekly_predictions.csv",
        "w_predictions.csv",
    ])
    def test_invalid_names_do_not_match(self, name):
        assert PREDICTION_PATTERN.match(name) is None


# ---------------------------------------------------------------------------
# _load_history (calibrate_winprob)
# ---------------------------------------------------------------------------

class TestLoadHistory:
    def test_returns_empty_for_no_paths(self, tmp_path):
        result = _load_history([])
        assert result.empty

    def test_loads_single_csv(self, tmp_path):
        p = tmp_path / "hist.csv"
        pd.DataFrame({"home_win_prob": [0.6], "home_margin": [3.0]}).to_csv(p, index=False)
        result = _load_history([p])
        assert len(result) == 1
        assert "_source_path" in result.columns

    def test_concatenates_multiple_csvs(self, tmp_path):
        for i in range(3):
            p = tmp_path / f"h{i}.csv"
            pd.DataFrame({"home_win_prob": [0.5 + i * 0.1]}).to_csv(p, index=False)
        result = _load_history(list(tmp_path.glob("*.csv")))
        assert len(result) == 3

    def test_skips_bad_path(self, tmp_path):
        result = _load_history([tmp_path / "nonexistent.csv"])
        assert result.empty


# ---------------------------------------------------------------------------
# find_prediction_files
# ---------------------------------------------------------------------------

class TestFindPredictionFiles:
    def test_raises_if_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_prediction_files(tmp_path / "does_not_exist")

    def test_finds_matching_files(self, tmp_path):
        (tmp_path / "w1_predictions.csv").write_text("a,b\n1,2\n")
        (tmp_path / "w5_predictions_full.csv").write_text("a,b\n1,2\n")
        (tmp_path / "other.csv").write_text("a,b\n1,2\n")
        result = find_prediction_files(tmp_path)
        names = {p.name for p in result}
        assert "w1_predictions.csv" in names
        assert "w5_predictions_full.csv" in names
        assert "other.csv" not in names

    def test_returns_empty_list_for_no_matches(self, tmp_path):
        (tmp_path / "noisy.csv").write_text("a\n1\n")
        result = find_prediction_files(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# load_prediction_frames
# ---------------------------------------------------------------------------

class TestLoadPredictionFrames:
    def test_loads_valid_file(self, tmp_path):
        p = _write_pred_csv(tmp_path, "w1_predictions.csv", [_base_row()])
        frames = load_prediction_frames([p])
        assert len(frames) == 1
        assert frames[0].week == 1

    def test_injects_week_if_missing(self, tmp_path):
        row = _base_row()
        row.pop("week")
        p = _write_pred_csv(tmp_path, "w3_predictions.csv", [row])
        frames = load_prediction_frames([p])
        assert frames[0].df["week"].iloc[0] == 3

    def test_skips_non_matching_names(self, tmp_path):
        p = tmp_path / "misc.csv"
        pd.DataFrame([_base_row()]).to_csv(p, index=False)
        frames = load_prediction_frames([p])
        assert frames == []

    def test_coerces_numeric_columns(self, tmp_path):
        row = _base_row()
        row["home_win_prob"] = "0.65"
        p = _write_pred_csv(tmp_path, "w2_predictions.csv", [row])
        frames = load_prediction_frames([p])
        assert pd.api.types.is_float_dtype(frames[0].df["home_win_prob"])


# ---------------------------------------------------------------------------
# unify_predictions
# ---------------------------------------------------------------------------

class TestUnifyPredictions:
    def _make_frame(self, tmp_path, filename, rows) -> PredictionFrame:
        p = _write_pred_csv(tmp_path, filename, rows)
        return PredictionFrame(
            path=p,
            week=int(PREDICTION_PATTERN.match(filename).group(1)),
            df=pd.read_csv(p),
        )

    def test_empty_input_returns_empty(self):
        result = unify_predictions([])
        assert result.empty

    def test_skips_frame_missing_required_cols(self, tmp_path):
        # Missing home_win_prob → should be skipped
        p = tmp_path / "w1_predictions.csv"
        pd.DataFrame([{"game_id": "x", "home_team": "KC", "away_team": "BUF",
                        "season": 2023, "week": 1, "pred_home_margin": 3.0}]).to_csv(p, index=False)
        frame = PredictionFrame(path=p, week=1, df=pd.read_csv(p))
        result = unify_predictions([frame])
        assert result.empty

    def test_deduplicates_same_game(self, tmp_path):
        row = _base_row()
        p1 = _write_pred_csv(tmp_path, "w1_predictions.csv", [row])
        p2 = _write_pred_csv(tmp_path, "w1_predictions_full.csv", [row])
        f1 = PredictionFrame(path=p1, week=1, df=pd.read_csv(p1))
        f2 = PredictionFrame(path=p2, week=1, df=pd.read_csv(p2))
        result = unify_predictions([f1, f2])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_preds_actuals(self, n: int = 8):
        rng = np.random.default_rng(0)
        games = [f"2023_01_{i:02d}" for i in range(n)]
        preds = pd.DataFrame({
            "game_id": games,
            "season": [2023] * n,
            "week": [1] * n,
            "home_win_prob": rng.uniform(0.3, 0.8, n),
            "pred_home_margin": rng.normal(0, 7, n),
        })
        actuals = pd.DataFrame({
            "game_id": games,
            "season": [2023] * n,
            "week": [1] * n,
            "home_margin": rng.normal(0, 10, n),
        })
        return preds, actuals

    def test_returns_two_dataframes(self):
        preds, actuals = self._make_preds_actuals()
        merged, weekly = compute_metrics(preds, actuals, min_games=1)
        assert isinstance(merged, pd.DataFrame)
        assert isinstance(weekly, pd.DataFrame)

    def test_weekly_has_expected_columns(self):
        preds, actuals = self._make_preds_actuals()
        _, weekly = compute_metrics(preds, actuals, min_games=1)
        for col in ("accuracy", "brier", "log_loss", "mae_margin", "rmse_margin"):
            assert col in weekly.columns

    def test_empty_actuals_returns_empty(self):
        preds, _ = self._make_preds_actuals()
        actuals = pd.DataFrame(columns=["game_id", "season", "week", "home_margin"])
        merged, weekly = compute_metrics(preds, actuals, min_games=1)
        assert merged.empty or weekly.empty

    def test_min_games_filter(self):
        preds, actuals = self._make_preds_actuals(n=3)
        _, weekly = compute_metrics(preds, actuals, min_games=5)
        assert weekly.empty


# ---------------------------------------------------------------------------
# compute_overall_metrics
# ---------------------------------------------------------------------------

class TestComputeOverallMetrics:
    def test_empty_returns_empty_dict(self):
        result = compute_overall_metrics(pd.DataFrame())
        assert result == {}

    def test_returns_expected_keys(self):
        merged = _make_pred_frame()
        result = compute_overall_metrics(merged)
        for key in ("n_games", "accuracy", "brier", "log_loss", "mae_margin", "rmse_margin"):
            assert key in result

    def test_n_games_correct(self):
        merged = pd.concat([_make_pred_frame(f"id{i}") for i in range(5)], ignore_index=True)
        result = compute_overall_metrics(merged)
        assert result["n_games"] == 5

    def test_brier_bounded(self):
        merged = _make_pred_frame()
        result = compute_overall_metrics(merged)
        assert 0.0 <= result["brier"] <= 1.0

    def test_accuracy_bounded(self):
        merged = _make_pred_frame()
        result = compute_overall_metrics(merged)
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# load_actuals
# ---------------------------------------------------------------------------
from src.evaluation.evaluate_predictions import load_actuals


class TestLoadActuals:
    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_actuals(tmp_path / "nonexistent.parquet")

    def _make_actuals(self, home_margin: float, game_id: str = "2023_01_KC_BUF") -> pd.DataFrame:
        return pd.DataFrame({
            "season": [2023],
            "week": [1],
            "game_id": [game_id],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "home_score": [24.0 if home_margin > 0 else 17.0],
            "away_score": [17.0 if home_margin > 0 else 24.0],
            "home_margin": [home_margin],
        })

    def test_reads_parquet(self, tmp_path):
        p = tmp_path / "features.parquet"
        self._make_actuals(7.0).to_parquet(p, index=False)
        result = load_actuals(p)
        assert "actual_home_win" in result.columns
        assert result["actual_home_win"].iloc[0] == 1.0

    def test_home_win_is_0_for_loss(self, tmp_path):
        p = tmp_path / "features.parquet"
        self._make_actuals(-7.0).to_parquet(p, index=False)
        result = load_actuals(p)
        assert result["actual_home_win"].iloc[0] == 0.0

    def test_tie_produces_nan(self, tmp_path):
        p = tmp_path / "features.parquet"
        df = self._make_actuals(0.0, game_id="tie_game")
        df.to_parquet(p, index=False)
        result = load_actuals(p)
        assert pd.isna(result["actual_home_win"].iloc[0])


# ---------------------------------------------------------------------------
# main workflows and report writers
# ---------------------------------------------------------------------------

def _make_evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    output_dir = tmp_path / "evaluation"
    games = [
        ("2024_01_KC_BUF", "KC", "BUF", 0.70, 6.0, 7.0),
        ("2024_01_DAL_PHI", "DAL", "PHI", 0.45, -2.0, -3.0),
        ("2024_01_SF_SEA", "SF", "SEA", 0.80, 8.0, -4.0),
        ("2024_01_BAL_CIN", "BAL", "CIN", 0.35, -4.0, 10.0),
    ]
    pred_rows = [
        {
            "game_id": game_id,
            "season": 2024,
            "week": 1,
            "home_team": home,
            "away_team": away,
            "home_win_prob": prob,
            "pred_home_margin": pred_margin,
        }
        for game_id, home, away, prob, pred_margin, _actual_margin in games
    ]
    actual_rows = [
        {
            "game_id": game_id,
            "season": 2024,
            "week": 1,
            "home_team": home,
            "away_team": away,
            "home_score": 24 + max(int(actual_margin), 0),
            "away_score": 24 + max(int(-actual_margin), 0),
            "home_margin": actual_margin,
        }
        for game_id, home, away, _prob, _pred_margin, actual_margin in games
    ]
    pd.DataFrame(pred_rows).to_csv(pred_dir / "w1_predictions.csv", index=False)
    features_path = tmp_path / "matchup_features.parquet"
    pd.DataFrame(actual_rows).to_parquet(features_path, index=False)
    return pred_dir, features_path, output_dir


class TestEvaluationWorkflow:
    def test_main_writes_expected_artifacts(self, tmp_path, monkeypatch):
        pred_dir, features_path, output_dir = _make_evaluation_fixture(tmp_path)
        output_dir.mkdir()
        pd.DataFrame([{"n_games": 1, "accuracy": 1.0}]).to_csv(
            output_dir / "overall_metrics.csv",
            index=False,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_predictions",
                "--pred-dir",
                str(pred_dir),
                "--features-path",
                str(features_path),
                "--output-dir",
                str(output_dir),
                "--min-games",
                "4",
                "--skip-plots",
            ],
        )

        eval_module.main()

        assert (output_dir / "merged_predictions_actuals.csv").exists()
        assert (output_dir / "weekly_metrics.csv").exists()
        overall = pd.read_csv(output_dir / "overall_metrics.csv")
        assert "run_timestamp" in overall.columns
        assert len(overall) == 2

    def test_main_raises_when_no_predictions(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "empty_predictions"
        pred_dir.mkdir()
        _, features_path, output_dir = _make_evaluation_fixture(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_predictions",
                "--pred-dir",
                str(pred_dir),
                "--features-path",
                str(features_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(FileNotFoundError):
            eval_module.main()

    def test_plot_metrics_writes_pngs(self, tmp_path):
        weekly = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 2],
                "accuracy": [0.75, 0.50],
                "brier": [0.18, 0.24],
                "log_loss": [0.55, 0.70],
                "mae_margin": [4.0, 6.0],
                "rmse_margin": [5.0, 8.0],
            }
        )

        assert plot_metrics(weekly, tmp_path)
        assert (tmp_path / "classification_metrics.png").exists()
        assert (tmp_path / "margin_metrics.png").exists()

    def test_plot_metrics_skips_empty_and_large_inputs(self, tmp_path):
        assert not plot_metrics(pd.DataFrame(), tmp_path)
        weekly = pd.DataFrame(
            {
                "season": [2024] * 181,
                "week": list(range(1, 182)),
                "accuracy": [0.5] * 181,
                "brier": [0.25] * 181,
                "log_loss": [0.69] * 181,
                "mae_margin": [7.0] * 181,
                "rmse_margin": [9.0] * 181,
            }
        )
        assert not plot_metrics(weekly, tmp_path)

    def test_summarize_underperformance_writes_reports(self, tmp_path):
        merged = pd.concat(
            [
                _make_pred_frame(
                    game_id="g1",
                    hwp=0.80,
                    margin=10.0,
                    home_margin=-7.0,
                ),
                _make_pred_frame(
                    game_id="g2",
                    hwp=0.35,
                    margin=-3.0,
                    home_margin=4.0,
                ),
            ],
            ignore_index=True,
        )

        summarize_underperformance(merged, tmp_path)

        assert (tmp_path / "largest_margin_errors.csv").exists()
        assert (tmp_path / "high_confidence_misclassifications.csv").exists()
        assert (tmp_path / "calibration_by_prob_bin.csv").exists()
        assert (tmp_path / "team_error_summary.csv").exists()


class TestCalibrationWorkflow:
    def test_main_writes_calibrator_metrics_and_plot(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(calibrate_module, "MODELS_DIR", tmp_path / "models")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        history = pd.DataFrame(
            {
                "game_id": [f"game_{i}" for i in range(8)],
                "season": [2024] * 8,
                "home_win_prob": [0.20, 0.30, 0.40, 0.45, 0.55, 0.65, 0.75, 0.85],
                "actual_home_win": [0, 0, 0, 1, 0, 1, 1, 1],
            }
        )
        history.to_csv(history_dir / "history_2024.csv", index=False)
        volatility_path = tmp_path / "volatility.csv"
        pd.DataFrame(
            {
                "game_id": ["game_4", "game_5", "game_6", "game_7"],
                "volatility_prob": [0.60, 0.70, 0.80, 0.90],
            }
        ).to_csv(volatility_path, index=False)
        output_path = tmp_path / "models" / "winprob_calibrator.pkl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "calibrate_winprob",
                "--history-dir",
                str(history_dir),
                "--seasons",
                "2024",
                "--min-games",
                "8",
                "--output",
                str(output_path),
                "--volatility-dataset",
                str(volatility_path),
                "--volatility-threshold",
                "0.55",
                "--volatility-strength",
                "0.25",
            ],
        )

        calibrate_module.main()

        artifact = joblib.load(output_path)
        assert artifact["sample_count"] == 8
        assert artifact["volatility_used"] is True
        assert Path("predictions/evaluation/overall_metrics.csv").exists()
        assert Path("analysis/reliability_curve.png").exists()

    def test_main_uses_threshold_from_volatility_dataset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(calibrate_module, "MODELS_DIR", tmp_path / "models")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        history = pd.DataFrame(
            {
                "game_id": [f"game_{i}" for i in range(8)],
                "season": [2024] * 8,
                "home_win_prob": [0.20, 0.30, 0.40, 0.45, 0.55, 0.65, 0.75, 0.85],
                "actual_home_win": [0, 0, 0, 1, 0, 1, 1, 1],
            }
        )
        history.to_csv(history_dir / "history_2024.csv", index=False)
        volatility_path = tmp_path / "volatility.csv"
        pd.DataFrame(
            {
                "game_id": ["game_4", "game_5", "game_6", "game_7"],
                "volatility_prob": [0.31, 0.33, 0.35, 0.37],
                "volatility_threshold": [0.34, 0.34, 0.34, 0.34],
            }
        ).to_csv(volatility_path, index=False)
        output_path = tmp_path / "models" / "winprob_calibrator.pkl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "calibrate_winprob",
                "--history-dir",
                str(history_dir),
                "--seasons",
                "2024",
                "--min-games",
                "8",
                "--output",
                str(output_path),
                "--volatility-dataset",
                str(volatility_path),
                "--volatility-strength",
                "0.25",
            ],
        )

        calibrate_module.main()

        artifact = joblib.load(output_path)
        assert artifact["volatility_used"] is True
        assert artifact["volatility_metadata"]["volatility_threshold"] == pytest.approx(0.34)

    def test_main_rejects_missing_history_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "calibrate_winprob",
                "--history-dir",
                str(tmp_path / "missing"),
            ],
        )

        with pytest.raises(FileNotFoundError):
            calibrate_module.main()
