from __future__ import annotations

from src.data.propline import _build_output_path, _flatten_odds_events


def test_build_output_path_uses_week_label():
    path = _build_output_path(2026, 3)

    assert path.name == "propline_player_props_2026_wk03.parquet"


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
