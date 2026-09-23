from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.player_props.features import build_player_prop_features, build_player_prop_prediction_features, main


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "opponent": "CAR",
                "player_id": "qb-1",
                "player_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "market": "qb_passing_yards",
                "actual_value": 200.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_ATL_CAR",
                "team": "ATL",
                "opponent": "CAR",
                "player_id": "qb-1",
                "player_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "market": "qb_passing_yards",
                "actual_value": 260.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "opponent": "CAR",
                "player_id": "qb-1",
                "player_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "market": "qb_passing_yards",
                "actual_value": 320.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "opponent": "CAR",
                "player_id": "rb-1",
                "player_name": "RB One",
                "position": "RB",
                "position_group": "RB",
                "market": "rb_rushing_yards",
                "actual_value": 40.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_ATL_CAR",
                "team": "ATL",
                "opponent": "CAR",
                "player_id": "rb-1",
                "player_name": "RB One",
                "position": "RB",
                "position_group": "RB",
                "market": "rb_rushing_yards",
                "actual_value": 80.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "CAR",
                "opponent": "ATL",
                "player_id": "dl-1",
                "player_name": "DL One",
                "position": "DE",
                "position_group": "DL",
                "market": "def_sacks",
                "actual_value": 0.0,
                "actual_over_zero": False,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_ATL_CAR",
                "team": "CAR",
                "opponent": "ATL",
                "player_id": "dl-1",
                "player_name": "DL One",
                "position": "DE",
                "position_group": "DL",
                "market": "def_sacks",
                "actual_value": 1.0,
                "actual_over_zero": True,
            },
        ]
    )


def test_build_player_prop_features_splits_offense_and_defense():
    offense, defense = build_player_prop_features(_labels(), seasons=[2026])

    assert set(offense["market_family"]) == {"offense"}
    assert set(defense["market_family"]) == {"defense"}
    assert set(offense["market"]) == {"qb_passing_yards", "rb_rushing_yards"}
    assert set(defense["market"]) == {"def_sacks"}


def test_build_player_prop_features_uses_only_prior_player_values():
    offense, _ = build_player_prop_features(_labels(), seasons=[2026])

    qb_week1 = offense[(offense["player_id"] == "qb-1") & (offense["week"] == 1)].iloc[0]
    qb_week2 = offense[(offense["player_id"] == "qb-1") & (offense["week"] == 2)].iloc[0]
    qb_week3 = offense[(offense["player_id"] == "qb-1") & (offense["week"] == 3)].iloc[0]

    assert qb_week1["player_games_prior"] == 0
    assert pd.isna(qb_week1["player_avg_prior"])
    assert qb_week2["player_games_prior"] == 1
    assert qb_week2["player_last1_value"] == 200.0
    assert qb_week2["player_avg_prior"] == 200.0
    assert qb_week3["player_last3_avg"] == 230.0
    assert qb_week3["player_season_avg_prior"] == 230.0


def test_build_player_prop_features_context_uses_prior_weeks():
    offense, defense = build_player_prop_features(_labels(), seasons=[2026])

    rb_week2 = offense[(offense["player_id"] == "rb-1") & (offense["week"] == 2)].iloc[0]
    sack_week2 = defense[(defense["player_id"] == "dl-1") & (defense["week"] == 2)].iloc[0]

    assert rb_week2["team_market_weekly_total_avg_prior"] == 40.0
    assert rb_week2["opponent_allowed_weekly_total_avg_prior"] == 40.0
    assert sack_week2["team_market_weekly_total_avg_prior"] == 0.0
    assert sack_week2["player_over_zero_rate_prior"] == 0.0


def test_build_player_prop_features_live_cutoff_excludes_active_week():
    offense, defense = build_player_prop_features(
        _labels(),
        seasons=[2026],
        exclude_from_season=2026,
        exclude_from_week=3,
    )

    assert offense["week"].max() == 2
    assert defense["week"].max() == 2


def test_build_player_prop_prediction_features_use_history_before_target_week():
    labels = _labels()
    candidates = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "opponent": "CAR",
                "player_id": "qb-1",
                "player_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "market": "qb_passing_yards",
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "team": "CAR",
                "opponent": "ATL",
                "player_id": "dl-1",
                "player_name": "DL One",
                "position": "DE",
                "position_group": "DL",
                "market": "def_sacks",
            },
        ]
    )

    offense, defense = build_player_prop_prediction_features(
        labels,
        candidates,
        exclude_from_season=2026,
        exclude_from_week=3,
    )

    qb = offense.iloc[0]
    sack = defense.iloc[0]
    assert qb["actual_value"] != qb["actual_value"]
    assert qb["player_games_prior"] == 2
    assert qb["player_avg_prior"] == 230.0
    assert qb["player_last3_avg"] == 230.0
    assert sack["player_games_prior"] == 2
    assert sack["player_over_zero_rate_prior"] == 0.5


def test_player_prop_features_cli_writes_split_outputs(tmp_path):
    labels_path = tmp_path / "labels.parquet"
    offense_path = tmp_path / "features_offense.parquet"
    defense_path = tmp_path / "features_defense.parquet"
    _labels().to_parquet(labels_path, index=False)

    rc = main(
        [
            "--labels",
            str(labels_path),
            "--offense-output",
            str(offense_path),
            "--defense-output",
            str(defense_path),
            "--exclude-from-season",
            "2026",
            "--exclude-from-week",
            "3",
        ]
    )

    assert rc == 0
    offense = pd.read_parquet(offense_path)
    defense = pd.read_parquet(defense_path)
    assert not offense.empty
    assert not defense.empty
    assert set(offense["market_family"]) == {"offense"}
    assert set(defense["market_family"]) == {"defense"}
