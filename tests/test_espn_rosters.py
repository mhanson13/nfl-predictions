from __future__ import annotations

from src.data.espn_rosters import normalize_depthchart, normalize_roster, normalize_teams


def test_normalize_teams_extracts_espn_team_ids_and_abbreviations():
    payload = {
        "sports": [
            {
                "leagues": [
                    {
                        "teams": [
                            {
                                "team": {
                                    "id": "2",
                                    "abbreviation": "BUF",
                                    "displayName": "Buffalo Bills",
                                    "shortDisplayName": "Bills",
                                    "location": "Buffalo",
                                    "name": "Bills",
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }

    teams = normalize_teams(payload)

    assert teams.loc[0, "espn_team_id"] == "2"
    assert teams.loc[0, "team"] == "BUF"
    assert teams.loc[0, "display_name"] == "Buffalo Bills"


def test_normalize_roster_handles_status_objects_and_strings():
    payload = {
        "season": {"year": 2026, "type": 2},
        "team": {"id": "2", "abbreviation": "BUF", "displayName": "Buffalo Bills"},
        "athletes": [
            {
                "position": {"abbreviation": "offense"},
                "items": [
                    {
                        "id": "3918298",
                        "displayName": "Josh Allen",
                        "shortName": "J. Allen",
                        "jersey": "17",
                        "position": {"abbreviation": "QB", "displayName": "Quarterback"},
                        "status": {"name": "Active", "type": "active", "abbreviation": "Active"},
                    }
                ],
            },
            {
                "position": "practiceSquad",
                "items": [
                    {
                        "id": "999",
                        "displayName": "Practice Player",
                        "position": {"abbreviation": "WR"},
                        "status": "Practice Squad",
                    }
                ],
            },
        ],
    }

    roster = normalize_roster(payload, season=2026, week=3)

    assert roster.shape[0] == 2
    allen = roster[roster["espn_player_id"].eq("3918298")].iloc[0]
    assert allen["team"] == "BUF"
    assert allen["position"] == "QB"
    assert allen["status_type"] == "active"
    practice = roster[roster["espn_player_id"].eq("999")].iloc[0]
    assert practice["roster_section"] == "practiceSquad"
    assert practice["status_name"] == "Practice Squad"


def test_normalize_depthchart_flattens_position_dicts_with_rank_order():
    payload = {
        "team": {"id": "2", "abbreviation": "BUF", "displayName": "Buffalo Bills"},
        "depthchart": [
            {
                "id": "offense",
                "name": "Offense",
                "positions": {
                    "qb": {
                        "position": {"abbreviation": "QB", "name": "Quarterback"},
                        "athletes": [
                            {"id": "3918298", "displayName": "Josh Allen"},
                            {"id": "3115293", "displayName": "Kyle Allen"},
                        ],
                    }
                },
            }
        ],
    }

    depth = normalize_depthchart(payload, season=2026, week=3)

    assert depth["espn_player_id"].tolist() == ["3918298", "3115293"]
    assert depth["depth_chart_position"].tolist() == ["QB", "QB"]
    assert depth["depth_rank"].tolist() == [1, 2]
