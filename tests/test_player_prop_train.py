from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.player_props.train import main, select_feature_columns, train_and_evaluate


def _feature_row(
    *,
    season: int,
    week: int,
    market: str,
    family: str,
    player_id: str,
    actual_value: float,
    feature_value: float,
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_ATL_CAR",
        "team": "ATL" if family == "offense" else "CAR",
        "opponent": "CAR" if family == "offense" else "ATL",
        "player_id": player_id,
        "player_name": player_id,
        "position": "WR" if market == "wrte_receiving_yards" else "DE",
        "position_group": "WR" if market == "wrte_receiving_yards" else "DL",
        "market": market,
        "market_family": family,
        "actual_value": actual_value,
        "actual_over_zero": actual_value > 0,
        "played_flag": True,
        "active_flag": True,
        "value_type": "yards" if family == "offense" else "count",
        "source": "test",
        "player_games_prior": week - 1,
        "player_avg_prior": feature_value,
        "player_last1_value": feature_value,
        "team_market_weekly_total_avg_prior": feature_value * 2,
        "opponent_allowed_weekly_total_avg_prior": feature_value * 1.5,
        "is_home": family == "offense",
    }


def _offense_features() -> pd.DataFrame:
    rows = []
    for season, base in [(2024, 10), (2025, 20), (2026, 30)]:
        for week in [1, 2, 3]:
            rows.append(
                _feature_row(
                    season=season,
                    week=week,
                    market="wrte_receiving_yards",
                    family="offense",
                    player_id=f"wr-{week}",
                    actual_value=base + week,
                    feature_value=base + week - 1,
                )
            )
    return pd.DataFrame(rows)


def _defense_features() -> pd.DataFrame:
    rows = []
    for season, base in [(2025, 0.0), (2026, 0.5)]:
        for week in [1, 2, 3, 4]:
            rows.append(
                _feature_row(
                    season=season,
                    week=week,
                    market="def_sacks",
                    family="defense",
                    player_id=f"dl-{week}",
                    actual_value=base + (1.0 if week % 2 == 0 else 0.0),
                    feature_value=float(week % 2),
                )
            )
    return pd.DataFrame(rows)


def test_select_feature_columns_excludes_targets_and_ids():
    features = _offense_features()

    cols = select_feature_columns(features)

    assert "actual_value" not in cols
    assert "player_id" not in cols
    assert "player_avg_prior" in cols
    assert "is_home" in cols


def test_train_and_evaluate_returns_walk_forward_predictions_and_metrics():
    predictions, metrics, bundles = train_and_evaluate(
        offense_features=_offense_features(),
        defense_features=_defense_features(),
        markets=["wrte_receiving_yards", "def_sacks"],
        model_type="ridge",
        min_train_rows=3,
        generated_at="2026-09-21T00:00:00+00:00",
    )

    assert set(predictions["market"]) == {"wrte_receiving_yards", "def_sacks"}
    assert set(metrics["scope"]) == {"season", "overall"}
    assert set(bundles) == {"wrte_receiving_yards", "def_sacks"}
    assert predictions["season"].min() > 2024
    sacks = metrics[(metrics["scope"] == "overall") & (metrics["market"] == "def_sacks")].iloc[0]
    assert sacks["brier_over_zero"] >= 0


def test_train_and_evaluate_honors_live_cutoff():
    predictions, _, _ = train_and_evaluate(
        offense_features=_offense_features(),
        defense_features=pd.DataFrame(),
        markets=["wrte_receiving_yards"],
        model_type="ridge",
        min_train_rows=3,
        exclude_from_season=2026,
        exclude_from_week=3,
    )

    assert predictions[predictions["season"] == 2026]["week"].max() == 2


def test_player_prop_train_cli_writes_outputs_and_models(tmp_path):
    offense_path = tmp_path / "offense.parquet"
    defense_path = tmp_path / "defense.parquet"
    output_dir = tmp_path / "eval"
    models_dir = tmp_path / "models"
    baseline_metrics = output_dir / "baseline_metrics.csv"
    baseline_predictions = output_dir / "baseline_predictions.csv"
    offense = _offense_features()
    defense = _defense_features()
    output_dir.mkdir(parents=True, exist_ok=True)
    offense.to_parquet(offense_path, index=False)
    defense.to_parquet(defense_path, index=False)
    pd.DataFrame(
        [
            {
                "scope": "overall",
                "market": "wrte_receiving_yards",
                "n_scored": 2,
                "mae": 5.0,
                "rmse": 6.0,
            }
        ]
    ).to_csv(baseline_metrics, index=False)
    pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_ATL_CAR",
                "team": "ATL",
                "player_id": "wr-1",
                "market": "wrte_receiving_yards",
                "actual_value": 21.0,
                "baseline_projection": 19.0,
                "label_available": True,
            }
        ]
    ).to_csv(baseline_predictions, index=False)

    rc = main(
        [
            "--offense-features",
            str(offense_path),
            "--defense-features",
            str(defense_path),
            "--output-dir",
            str(output_dir),
            "--models-dir",
            str(models_dir),
            "--baseline-metrics",
            str(baseline_metrics),
            "--baseline-predictions",
            str(baseline_predictions),
            "--markets",
            "wrte_receiving_yards",
            "def_sacks",
            "--model-type",
            "ridge",
            "--min-train-rows",
            "3",
        ]
    )

    assert rc == 0
    assert (output_dir / "model_predictions.csv").exists()
    assert (output_dir / "model_metrics.csv").exists()
    assert (output_dir / "model_vs_baseline.csv").exists()
    assert (models_dir / "offense" / "wrte_receiving_yards_model.pkl").exists()
    assert (models_dir / "defense" / "def_sacks_model.pkl").exists()
