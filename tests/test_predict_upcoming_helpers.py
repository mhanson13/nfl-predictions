import pandas as pd

import src.predict.predict_upcoming as predict_upcoming
from src.predict.predict_upcoming import (
    QB_PLAYER_PROJECTION_COLUMNS,
    _apply_probability_calibrator,
    _best_probability_signal,
    _cleanup_prediction_columns,
    _fallback_if_probability_collapsed,
    _has_probability_signal,
    _is_collapsed_probability,
    _is_saturated_probability_signal,
    _player_qb_predictions,
    _should_skip_volatility_shrinkage,
    _volatility_artifact_skip_reason,
)


def test_probability_signal_helpers_detect_collapsed_and_variable_series():
    assert _is_collapsed_probability(pd.Series([0.15, 0.15, 0.15]))
    assert not _has_probability_signal(pd.Series([0.15, 0.15, 0.15]))

    assert not _is_collapsed_probability(pd.Series([0.11, 0.15, 0.23]))
    assert _has_probability_signal(pd.Series([0.11, 0.15, 0.23]))
    assert _is_saturated_probability_signal(pd.Series([0.98, 0.98, 0.99, 1.0]))
    assert not _is_saturated_probability_signal(pd.Series([0.58, 0.62, 0.66, 0.69]))


def test_fallback_restores_signal_when_stage_collapses():
    candidate = pd.Series([0.15, 0.15, 0.15])
    fallback = pd.Series([0.03, 0.12, 0.24])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="caps",
    )

    assert reason == "caps_collapsed"
    assert restored.tolist() == [0.03, 0.12, 0.24]


def test_fallback_clips_extreme_restored_probabilities():
    candidate = pd.Series([0.15, 0.15, 0.15])
    fallback = pd.Series([0.0, 0.5, 1.0])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="calibration",
    )

    assert reason == "calibration_collapsed"
    assert restored.tolist() == [0.02, 0.5, 0.98]


def test_fallback_does_not_clip_away_near_one_signal():
    candidate = pd.Series([0.98, 0.98, 0.98])
    fallback = pd.Series([0.996, 0.998, 1.0])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="volatility",
    )

    assert reason == "volatility_collapsed"
    assert restored.tolist() == [0.996, 0.998, 0.999]


def test_fallback_restores_signal_when_stage_severely_compresses():
    candidate = pd.Series([0.15, 0.15, 0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
    fallback = pd.Series([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.17, 0.19])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="caps",
    )

    assert reason == "caps_collapsed"
    assert restored.tolist() == fallback.tolist()


def test_fallback_restores_signal_when_stage_range_compresses():
    candidate = pd.Series([0.8225, 0.8230, 0.8234, 0.8240])
    fallback = pd.Series([0.62, 0.64, 0.66, 0.68])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="volatility",
    )

    assert reason == "volatility_collapsed"
    assert restored.tolist() == fallback.tolist()


def test_fallback_restores_raw_signal_when_calibration_saturates():
    candidate = pd.Series([0.78, 0.96, 0.97, 0.98, 0.979, 0.976, 0.974, 0.971])
    fallback = pd.Series([0.54, 0.65, 0.66, 0.68, 0.67, 0.66, 0.65, 0.64])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="calibration",
    )

    assert reason == "calibration_saturated"
    assert restored.tolist() == fallback.tolist()


def test_fallback_allows_unsaturated_calibration_shift():
    candidate = pd.Series([0.58, 0.62, 0.65, 0.69])
    fallback = pd.Series([0.54, 0.58, 0.61, 0.64])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="calibration",
    )

    assert reason is None
    assert restored.tolist() == candidate.tolist()


def test_best_probability_signal_skips_collapsed_columns():
    df = pd.DataFrame(
        {
            "home_win_prob_capped": [0.98, 0.98, 0.98],
            "home_win_prob_calibrated": [0.41, 0.52, 0.63],
            "home_win_prob_model_raw": [0.31, 0.33, 0.35],
        }
    )

    signal, source = _best_probability_signal(df, df.index)

    assert source == "home_win_prob_calibrated"
    assert signal.tolist() == [0.41, 0.52, 0.63]


def test_best_probability_signal_prefers_widest_available_range():
    df = pd.DataFrame(
        {
            "home_win_prob_capped": [0.996, 0.997, 0.998],
            "home_win_prob_calibrated": [0.996, 0.998, 0.999],
            "home_win_prob_model_raw": [0.62, 0.65, 0.68],
        }
    )

    signal, source = _best_probability_signal(df, df.index)

    assert source == "home_win_prob_model_raw"
    assert signal.tolist() == [0.62, 0.65, 0.68]


def test_should_skip_volatility_shrinkage_when_coverage_is_extreme():
    skip, coverage = _should_skip_volatility_shrinkage([1, 1, 1, 0], min_coverage=0.02, max_coverage=0.70)

    assert skip
    assert coverage == 0.75

    skip, coverage = _should_skip_volatility_shrinkage([0, 0, 0, 0], min_coverage=0.02, max_coverage=0.70)

    assert skip
    assert coverage == 0.0

    skip, coverage = _should_skip_volatility_shrinkage([1, 0, 0, 0], min_coverage=0.02, max_coverage=0.70)

    assert not skip
    assert coverage == 0.25


def test_cleanup_preserves_zero_volatility_diagnostics():
    df = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "volatility_prob": [0.0, 0.0],
            "volatility_label": [0, 0],
            "unused_all_zero": [0, 0],
        }
    )

    cleaned = _cleanup_prediction_columns(df)

    assert "volatility_prob" in cleaned.columns
    assert "volatility_label" in cleaned.columns
    assert "unused_all_zero" not in cleaned.columns


def test_volatility_artifact_skip_reason_rejects_weak_artifacts():
    artifact = type("Artifact", (), {"metrics": {"auc": 0.49, "pred_positive_rate": 0.40}})()

    assert _volatility_artifact_skip_reason(artifact).startswith("auc=")

    artifact = type("Artifact", (), {"metrics": {"auc": 0.61, "pred_positive_rate": 0.01}})()

    assert _volatility_artifact_skip_reason(artifact).startswith("pred_positive_rate=")

    artifact = type("Artifact", (), {"metrics": {"auc": 0.61, "pred_positive_rate": 0.20}})()

    assert _volatility_artifact_skip_reason(artifact) is None


def test_apply_probability_calibrator_uses_predict_proba_when_available():
    class FakeClassifier:
        def predict_proba(self, arr):
            return [[1 - float(value[0]), float(value[0])] for value in arr]

        def predict(self, arr):
            return [1 for _ in arr]

    result = _apply_probability_calibrator(FakeClassifier(), pd.Series([0.2, 0.8]))

    assert result.tolist() == [0.2, 0.8]


def test_qb_projection_writes_empty_current_file_when_no_rows_survive(tmp_path):
    pred_games = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
            }
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "season_type": 2,
                "recent_team": "CCC",
                "player_id": "00-test",
                "player_name": "Test QB",
                "position_group": "QB",
                "passing_yards": 100,
                "passing_tds": 1,
                "rushing_tds": 0,
                "attempts": 20,
                "interceptions": 0,
                "completions": 10,
                "rushing_yards": 5,
            }
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "season": 2026,
                "team": "BBB",
                "gsis_id": "00-other",
                "jersey_number": 1,
                "full_name": "Other QB",
                "status": "ACT",
                "position": "QB",
                "depth_chart_position": "QB",
            }
        ]
    )
    out_path = tmp_path / "players_qb.csv"

    result = _player_qb_predictions(pred_games, str(out_path), stats, roster, debug=True)

    assert result is not None
    assert result.empty
    assert out_path.exists()
    exported = pd.read_csv(out_path)
    assert exported.empty
    assert list(exported.columns) == QB_PLAYER_PROJECTION_COLUMNS


def test_load_player_stats_frame_accepts_release_team_and_reg_season_type(tmp_path, monkeypatch):
    raw_path = tmp_path / "nfl_player_stats.parquet"
    raw_path.touch()
    raw = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "season_type": "REG",
                "team": "den",
                "player_id": "00-test",
                "player_name": "Test Player",
            },
            {
                "season": 2026,
                "week": 2,
                "season_type": "POST",
                "team": "KC",
                "player_id": "00-post",
                "player_name": "Post Player",
            },
        ]
    )

    monkeypatch.setattr(predict_upcoming, "RAW_DIR", tmp_path)
    monkeypatch.setattr(predict_upcoming, "read_df", lambda path: raw)

    result = predict_upcoming._load_player_stats_frame()

    assert result["player_id"].tolist() == ["00-test"]
    assert result["recent_team"].tolist() == ["DEN"]
