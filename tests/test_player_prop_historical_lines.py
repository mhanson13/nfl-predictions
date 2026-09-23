from __future__ import annotations

import pandas as pd

from src.player_props.historical_lines import join_historical_lines, summarize_join


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_CAR_ATL",
                "team": "ATL",
                "opponent": "CAR",
                "player_id": "qb-1",
                "player_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "market": "qb_passing_yards",
                "actual_value": 260.0,
                "actual_over_zero": True,
                "model_projection": 252.0,
                "prob_over_zero": None,
                "model_type": "ridge",
                "evaluation_method": "season_walk_forward",
                "train_rows": 100,
                "feature_count": 10,
            },
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_DEN_LV",
                "team": "DEN",
                "opponent": "LV",
                "player_id": "wr-1",
                "player_name": "C.Sutton",
                "position": "WR",
                "position_group": "WR",
                "market": "wrte_receiving_yards",
                "actual_value": 47.0,
                "actual_over_zero": True,
                "model_projection": 53.0,
                "prob_over_zero": None,
                "model_type": "ridge",
                "evaluation_method": "season_walk_forward",
                "train_rows": 100,
                "feature_count": 10,
            },
        ]
    )


def _lines() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "line_source": "odds_api_historical",
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_CAR_ATL",
                "oddsapi_event_id": "evt-1",
                "requested_date": "2025-09-07T16:50:00Z",
                "snapshot_timestamp": "2025-09-07T16:45:00Z",
                "market": "qb_passing_yards",
                "market_key": "player_pass_yds",
                "player_name": "QB One",
                "player_name_key": "q:one",
                "player_name_prefix_key": "qb:one",
                "sportsbook": "draftkings",
                "line": 245.5,
                "over_odds": -110,
                "under_odds": -110,
                "implied_probability": 0.5,
                "line_balance_distance": 0.0,
                "line_is_comparable": True,
                "line_updated_at": "2025-09-07T16:45:00Z",
            },
            {
                "line_source": "odds_api_historical",
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_DEN_LV",
                "oddsapi_event_id": "evt-2",
                "requested_date": "2025-09-07T20:15:00Z",
                "snapshot_timestamp": "2025-09-07T20:10:00Z",
                "market": "wrte_receiving_yards",
                "market_key": "player_reception_yds",
                "player_name": "Courtland Sutton",
                "player_name_key": "c:sutton",
                "player_name_prefix_key": "co:sutton",
                "sportsbook": "fanduel",
                "line": 49.5,
                "over_odds": -105,
                "under_odds": -115,
                "implied_probability": 0.477273,
                "line_balance_distance": 0.022727,
                "line_is_comparable": True,
                "line_updated_at": "2025-09-07T20:10:00Z",
            },
        ]
    )


def test_join_historical_lines_matches_strict_and_initial_last_fallback():
    joined, prepared_lines = join_historical_lines(lines=_lines(), predictions=_predictions())

    assert len(prepared_lines) == 2
    assert len(joined) == 2
    rows = {row["market"]: row for _, row in joined.iterrows()}
    qb = rows["qb_passing_yards"]
    wr = rows["wrte_receiving_yards"]

    assert qb["match_method"] == "strict_prefix"
    assert qb["line_player_name"] == "QB One"
    assert qb["model_player_name"] == "QB One"
    assert qb["model_edge"] == 6.5
    assert bool(qb["actual_over_line"]) is True
    assert qb["model_pick"] == "over"
    assert bool(qb["model_pick_correct"]) is True

    assert wr["match_method"] == "initial_last_unique"
    assert wr["line_player_name"] == "Courtland Sutton"
    assert wr["model_player_name"] == "C.Sutton"
    assert wr["player_id"] == "wr-1"
    assert wr["model_pick"] == "over"
    assert bool(wr["actual_under_line"]) is True
    assert bool(wr["model_pick_correct"]) is False


def test_summarize_join_reports_match_and_pick_accuracy():
    joined, prepared_lines = join_historical_lines(lines=_lines(), predictions=_predictions())

    summary = summarize_join(prepared_lines, joined, generated_at="2026-09-22T00:00:00+00:00")

    overall = summary[summary["scope"].eq("all")].iloc[0]
    assert overall["available_line_rows"] == 2
    assert overall["matched_line_rows"] == 2
    assert overall["matched_rate"] == 1.0
    assert overall["model_pick_rows"] == 2
    assert overall["model_pick_accuracy"] == 0.5
    assert set(summary["scope"]) == {"all", "market", "sportsbook"}
