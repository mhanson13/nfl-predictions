from __future__ import annotations

import pandas as pd

from src.player_props.labels import build_player_prop_labels, build_touchdown_novelty_labels


def _sample_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "player_id": "qb-1",
                "player_name": None,
                "player_display_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "recent_team": "ATL",
                "opponent_team": "CAR",
                "season_type": 2,
                "passing_yards": 275,
                "passing_tds": 2,
                "rushing_yards": 12,
                "rushing_tds": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "def_sacks": None,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "player_id": "rb-1",
                "player_name": "RB One",
                "player_display_name": "RB One",
                "position": "RB",
                "position_group": "RB",
                "recent_team": "ATL",
                "opponent_team": "CAR",
                "season_type": "REG",
                "passing_yards": 0,
                "passing_tds": 0,
                "rushing_yards": 88,
                "rushing_tds": 1,
                "receiving_yards": 14,
                "receiving_tds": 0,
                "def_sacks": None,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "player_id": "wr-1",
                "player_name": "WR One",
                "player_display_name": "WR One",
                "position": "WR",
                "position_group": "WR",
                "recent_team": "CAR",
                "opponent_team": "ATL",
                "season_type": 2,
                "passing_yards": 0,
                "passing_tds": 0,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "receiving_yards": 102,
                "receiving_tds": 1,
                "def_sacks": None,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "player_id": "dl-1",
                "player_name": "DL One",
                "player_display_name": "DL One",
                "position": "DE",
                "position_group": "DL",
                "recent_team": "CAR",
                "opponent_team": "ATL",
                "season_type": 2,
                "passing_yards": 0,
                "passing_tds": 0,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "def_sacks": 1.5,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_ATL_CAR",
                "player_id": "qb-1",
                "player_name": "QB One",
                "player_display_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "recent_team": "ATL",
                "opponent_team": "CAR",
                "season_type": 2,
                "passing_yards": 310,
                "passing_tds": 3,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "def_sacks": None,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR_POST",
                "player_id": "qb-post",
                "player_name": "QB Post",
                "player_display_name": "QB Post",
                "position": "QB",
                "position_group": "QB",
                "recent_team": "ATL",
                "opponent_team": "CAR",
                "season_type": 3,
                "passing_yards": 400,
                "passing_tds": 4,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "def_sacks": None,
            },
        ]
    )


def test_build_player_prop_labels_creates_supported_markets():
    labels = build_player_prop_labels(_sample_stats(), seasons=[2026])

    assert set(labels["market"]) == {
        "qb_passing_yards",
        "qb_passing_tds",
        "rb_rushing_yards",
        "rb_rushing_tds",
        "wrte_receiving_yards",
        "wrte_receiving_tds",
        "anytime_td",
        "def_sacks",
    }
    qb = labels[(labels["market"] == "qb_passing_yards") & (labels["player_id"] == "qb-1")].iloc[0]
    assert qb["actual_value"] == 275
    assert qb["player_name"] == "QB One"
    assert bool(qb["actual_over_zero"]) is True
    assert qb["team"] == "ATL"
    assert qb["opponent"] == "CAR"

    sack = labels[labels["market"] == "def_sacks"].iloc[0]
    assert sack["actual_value"] == 1.5
    assert sack["value_type"] == "count"

    td = labels[(labels["market"] == "wrte_receiving_tds") & (labels["player_id"] == "wr-1")].iloc[0]
    assert td["actual_value"] == 1
    assert td["value_type"] == "count"

    anytime = labels[(labels["market"] == "anytime_td") & (labels["player_id"] == "rb-1")].iloc[0]
    assert anytime["actual_value"] == 1
    assert bool(anytime["actual_over_zero"]) is True


def test_build_player_prop_labels_filters_non_regular_and_live_cutoff():
    labels = build_player_prop_labels(
        _sample_stats(),
        seasons=[2026],
        exclude_from_season=2026,
        exclude_from_week=2,
    )

    assert "qb-post" not in set(labels["player_id"])
    assert labels["week"].max() == 1
    assert labels[labels["market"] == "qb_passing_yards"]["actual_value"].tolist() == [275]


def test_build_player_prop_labels_can_select_market_subset():
    labels = build_player_prop_labels(_sample_stats(), seasons=[2026], markets=["wrte_receiving_yards"])

    assert labels["market"].unique().tolist() == ["wrte_receiving_yards"]
    assert labels["player_id"].tolist() == ["wr-1"]


def test_build_player_prop_labels_drops_self_opponent_rows():
    stats = _sample_stats()
    stats.loc[len(stats)] = {
        "season": 2026,
        "week": 1,
        "game_id": "bad_self_opponent",
        "player_id": "wr-returner",
        "player_name": "WR Returner",
        "player_display_name": "WR Returner",
        "position": "WR",
        "position_group": "WR",
        "recent_team": "DET",
        "opponent_team": "DET",
        "season_type": 2,
        "passing_yards": 0,
        "passing_tds": 0,
        "rushing_yards": 0,
        "rushing_tds": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "def_sacks": 0.0,
    }

    labels = build_player_prop_labels(stats, seasons=[2026])

    assert "wr-returner" not in set(labels["player_id"])


def test_build_player_prop_labels_fills_missing_game_id_from_schedule():
    stats = _sample_stats()
    stats["game_id"] = None
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "home_team": "ATL",
                "away_team": "CAR",
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_ATL_CAR",
                "home_team": "ATL",
                "away_team": "CAR",
            },
        ]
    )

    labels = build_player_prop_labels(stats, seasons=[2026], schedule=schedule)

    assert labels["game_id"].notna().all()
    assert set(labels["game_id"]) == {"2026_01_ATL_CAR", "2026_02_ATL_CAR"}


def test_build_touchdown_novelty_labels_from_pbp_and_anytime_candidates():
    labels = build_player_prop_labels(_sample_stats(), seasons=[2026])
    pbp = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "touchdown": 1,
                "td_player_id": "rb-1",
                "td_player_name": "RB One",
                "game_seconds_remaining": 3100,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "touchdown": 1,
                "td_player_id": "wr-1",
                "td_player_name": "WR One",
                "game_seconds_remaining": 1900,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "touchdown": 1,
                "td_player_id": "rb-1",
                "td_player_name": "RB One",
                "game_seconds_remaining": 1200,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "touchdown": 1,
                "td_player_id": "rb-1",
                "td_player_name": "RB One",
                "game_seconds_remaining": 200,
            },
        ]
    )

    novelty = build_touchdown_novelty_labels(pbp, candidate_labels=labels, seasons=[2026])

    by_game_market_player = novelty.set_index(["game_id", "market", "player_id"])["actual_value"]
    game_key = "2026_01_ATL_CAR"
    assert by_game_market_player.loc[(game_key, "first_td", "rb-1")] == 1
    assert by_game_market_player.loc[(game_key, "last_td", "rb-1")] == 1
    assert by_game_market_player.loc[(game_key, "two_plus_tds", "rb-1")] == 1
    assert by_game_market_player.loc[(game_key, "three_plus_tds", "rb-1")] == 1
    assert by_game_market_player.loc[(game_key, "first_td", "wr-1")] == 0
    assert by_game_market_player.loc[(game_key, "last_td", "wr-1")] == 0
    assert by_game_market_player.loc[(game_key, "two_plus_tds", "wr-1")] == 0
    assert set(novelty["market"]) == {"first_td", "last_td", "two_plus_tds", "three_plus_tds"}
    assert novelty["source"].eq("nfl_pbp").all()
