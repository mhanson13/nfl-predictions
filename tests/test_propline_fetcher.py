from __future__ import annotations

import json

import pandas as pd

from src.data.propline import (
    DEFAULT_MARKETS,
    _build_output_path,
    _filter_to_schedule_week,
    _flatten_nested_columns,
    _flatten_odds_events,
    _write_propline_frame,
)


def test_build_output_path_uses_week_label():
    path = _build_output_path(2026, 3)

    assert path.name == "propline_player_props_2026_wk03.parquet"


def test_default_markets_include_touchdown_props():
    assert DEFAULT_MARKETS == [
        "player_pass_yds",
        "player_pass_tds",
        "player_rush_yds",
        "player_rush_tds",
        "player_reception_yds",
        "player_reception_tds",
        "player_anytime_td",
        "player_1st_td",
        "player_last_td",
        "player_2plus_td",
        "player_3plus_td",
        "player_sacks",
    ]


def test_flatten_odds_events_normalizes_player_prop_outcomes():
    payload = [
        {
            "id": "evt-1",
            "sport_key": "football_nfl",
            "home_team": "Atlanta Falcons",
            "away_team": "Carolina Panthers",
            "commence_time": "2026-09-27T17:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "player_pass_yds",
                            "last_update": "2026-09-22T12:00:00Z",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "QB One",
                                    "price": -110,
                                    "point": 245.5,
                                    "player_id": "espn:1",
                                    "outcome_id": "out-1",
                                },
                                {
                                    "name": "Under",
                                    "description": "QB One",
                                    "price": -110,
                                    "point": 245.5,
                                    "player_id": "espn:1",
                                    "outcome_id": "out-2",
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    frame = _flatten_odds_events(
        payload,
        requested_markets=["player_pass_yds"],
        requested_bookmakers=["draftkings"],
    )

    assert len(frame) == 2
    assert set(frame["outcome_name"]) == {"Over", "Under"}
    assert frame["event_id"].iloc[0] == "evt-1"
    assert frame["bookmaker_key"].iloc[0] == "draftkings"
    assert frame["market_key"].iloc[0] == "player_pass_yds"
    assert frame["player_name"].iloc[0] == "QB One"
    assert frame["requested_markets"].iloc[0] == "player_pass_yds"


def test_filter_to_schedule_week_removes_out_of_week_events_and_adds_game_id():
    odds = pd.DataFrame(
        [
            {
                "event_id": "old",
                "home_team": "Chicago Bears",
                "away_team": "Philadelphia Eagles",
                "commence_time": "2026-09-29T00:15:00Z",
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
                "outcome_name": "Over",
                "player_name": "RB One",
                "price": -110,
            },
            {
                "event_id": "target",
                "home_team": "Cleveland Browns",
                "away_team": "Pittsburgh Steelers",
                "commence_time": "2026-10-02T00:15:00Z",
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
                "outcome_name": "Over",
                "player_name": "RB Two",
                "price": -110,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_PIT_CLE",
                "home_team": "CLE",
                "away_team": "PIT",
                "gameday": "2026-10-01",
                "gametime": "20:15",
            }
        ]
    )

    filtered = _filter_to_schedule_week(odds, schedule, season=2026, week=4)

    assert filtered["event_id"].tolist() == ["target"]
    assert filtered["game_id"].iloc[0] == "2026_04_PIT_CLE"
    assert int(filtered["schedule_week"].iloc[0]) == 4
    assert filtered["event_home_abbr"].iloc[0] == "CLE"
    assert filtered["event_away_abbr"].iloc[0] == "PIT"


def test_flatten_nested_columns_stringifies_list_values_for_parquet():
    frame = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "merged_from_event_ids": ["a", "b"],
            }
        ]
    )

    flattened = _flatten_nested_columns(frame)

    assert json.loads(flattened["merged_from_event_ids"].iloc[0]) == ["a", "b"]


def test_flatten_nested_columns_keeps_all_null_columns_pyarrow_readable(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "espn_event_id": None,
                "merged_from_event_ids": None,
            }
        ]
    )
    output = tmp_path / "propline.parquet"

    _write_propline_frame(frame, output)
    reread = pd.read_parquet(output)

    assert reread["event_id"].tolist() == ["evt-1"]
    assert "espn_event_id" in reread.columns
    assert "merged_from_event_ids" in reread.columns


def test_write_propline_frame_is_readable_by_pyarrow_and_fastparquet(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "merged_from_event_ids": ["evt-a", "evt-b"],
                "espn_event_id": None,
                "book_updated_at": None,
            }
        ]
    )
    output = tmp_path / "propline.parquet"

    written = _write_propline_frame(frame, output)
    pyarrow_frame = pd.read_parquet(written, engine="pyarrow")
    fastparquet_frame = pd.read_parquet(written, engine="fastparquet")

    assert pyarrow_frame["event_id"].tolist() == ["evt-1"]
    assert fastparquet_frame["event_id"].tolist() == ["evt-1"]
    assert json.loads(pyarrow_frame["merged_from_event_ids"].iloc[0]) == ["evt-a", "evt-b"]
    assert "espn_event_id" in pyarrow_frame.columns
    assert "book_updated_at" in pyarrow_frame.columns
