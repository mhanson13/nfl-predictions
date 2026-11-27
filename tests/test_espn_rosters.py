from __future__ import annotations

import json
from pathlib import Path


import src.data.espn_rosters as er
from src.data.espn_rosters import normalize_depthchart, normalize_roster, normalize_teams

FIXTURE = Path(__file__).parent / "fixtures" / "espn_depthchart_atl_2026_wk12.json"


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



# --- injuries from depth charts (recorded payload: tests/fixtures/espn_depthchart_atl_2026_wk12.json) ---


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fixture_injured_ids(payload: dict) -> set[str]:
    return {
        str(a["id"])
        for sec in payload["depthchart"]
        for pos in sec["positions"].values()
        for a in pos["athletes"]
        if a.get("injuries")
    }


def test_injuries_from_recorded_depthchart_one_row_per_injured_athlete():
    payload = _fixture()
    injuries = er.injuries_from_depthchart(payload, season=2026, week=12)

    assert list(injuries.columns) == er.INJURY_COLUMNS
    assert set(injuries["espn_player_id"]) == _fixture_injured_ids(payload)
    assert injuries["espn_player_id"].is_unique  # athletes at several depth slots appear once
    assert set(injuries["team"]) == {"ATL"}
    assert injuries["injury_status"].notna().all()
    assert set(injuries["injury_status"]) <= {"Injured Reserve", "Out", "Questionable", "Doubtful", "Day-To-Day"}
    hand = injuries[injuries["espn_player_id"].eq("3126352")].iloc[0]
    assert hand["injury_status"] == "Injured Reserve"
    assert hand["injury_type"] == "IR"
    assert hand["injury_date"] == "2026-09-17T00:21Z"
    assert "injured reserve" in hand["injury_detail"].lower()


def test_injuries_from_depthchart_keeps_latest_entry_and_handles_empty():
    payload = {
        "team": {"id": "2", "abbreviation": "BUF"},
        "depthchart": [
            {"positions": {"qb": {"athletes": [
                {"id": "9", "displayName": "A", "injuries": [
                    {"status": "Questionable", "type": {"abbreviation": "Q"}, "date": "2026-11-20T00:00Z"},
                    {"status": "Out", "type": {"abbreviation": "O"}, "date": "2026-11-26T00:00Z"},
                ]},
                {"id": "10", "displayName": "Healthy"},
            ]}}},
        ],
    }
    injuries = er.injuries_from_depthchart(payload, season=2026, week=12)
    assert injuries[["espn_player_id", "injury_status", "injury_type"]].values.tolist() == [["9", "Out", "O"]]

    empty = er.injuries_from_depthchart({}, season=2026, week=12)
    assert empty.empty and list(empty.columns) == er.INJURY_COLUMNS


def test_injury_rows_feed_player_availability_lookup():
    from src.player_availability import _prepare_espn_injury_lookup

    injuries = er.injuries_from_depthchart(_fixture(), season=2026, week=12)
    lookup = _prepare_espn_injury_lookup(injuries)
    assert len(lookup) == len(injuries)
    assert lookup["injury_reported"].all()


def _fake_request(payloads, calls):
    def request(session, endpoint, *, timeout=er.DEFAULT_TIMEOUT):
        calls.append(endpoint)
        return payloads.get(endpoint, {})

    return request


def test_fetch_uses_depthcharts_for_injuries_and_never_calls_injuries_endpoint(monkeypatch):
    teams_payload = {"sports": [{"leagues": [{"teams": [{"team": {"id": "1", "abbreviation": "ATL"}}]}]}]}
    payloads = {
        "teams": teams_payload,
        "teams/1/roster": {"team": {"id": "1", "abbreviation": "ATL"}, "athletes": []},
        "teams/1/depthcharts": _fixture(),
    }
    calls: list[str] = []
    monkeypatch.setattr(er, "_request_endpoint", _fake_request(payloads, calls))

    _, _, depth, injuries = er.fetch_espn_roster_snapshots(season=2026, week=12, sleep=0)
    assert calls == ["teams", "teams/1/roster", "teams/1/depthcharts"]
    assert not any(c.endswith("/injuries") for c in calls)
    assert not depth.empty
    assert set(injuries["espn_player_id"]) == _fixture_injured_ids(_fixture())

    calls.clear()
    _, _, depth, injuries = er.fetch_espn_roster_snapshots(season=2026, week=12, sleep=0, include_depthcharts=False)
    assert calls == ["teams", "teams/1/roster", "teams/1/depthcharts"]  # still fetched for the injuries
    assert depth.empty and not injuries.empty

    calls.clear()
    er.fetch_espn_roster_snapshots(season=2026, week=12, sleep=0, include_depthcharts=False, include_injuries=False)
    assert calls == ["teams", "teams/1/roster"]


def test_empty_payloads_and_outputs_print_warnings(monkeypatch, tmp_path, capsys):
    teams_payload = {"sports": [{"leagues": [{"teams": [{"team": {"id": "1", "abbreviation": "ATL"}}]}]}]}
    payloads = {"teams": teams_payload, "teams/1/roster": {}, "teams/1/depthcharts": {}}
    monkeypatch.setattr(er, "_request_endpoint", _fake_request(payloads, []))
    monkeypatch.setattr(er, "RAW_DIR", tmp_path)
    monkeypatch.setattr(er, "write_df", lambda df, path: path)

    assert er.main(["--season", "2026", "--week", "12"]) == 0
    out = capsys.readouterr().out
    assert "[espn_rosters] Warning: 2 ESPN endpoint(s) returned an empty payload: teams/1/roster, teams/1/depthcharts" in out
    for name in ("roster", "depthchart", "injury"):
        assert f"[espn_rosters] Warning: wrote 0 {name} rows for 2026 week 12" in out


def test_no_warning_when_everything_has_rows(monkeypatch, tmp_path, capsys):
    teams_payload = {"sports": [{"leagues": [{"teams": [{"team": {"id": "1", "abbreviation": "ATL"}}]}]}]}
    roster = {
        "team": {"id": "1", "abbreviation": "ATL"},
        "athletes": [{"position": "offense", "items": [{"id": "1", "displayName": "X"}]}],
    }
    payloads = {"teams": teams_payload, "teams/1/roster": roster, "teams/1/depthcharts": _fixture()}
    monkeypatch.setattr(er, "_request_endpoint", _fake_request(payloads, []))
    monkeypatch.setattr(er, "RAW_DIR", tmp_path)
    monkeypatch.setattr(er, "write_df", lambda df, path: path)

    assert er.main(["--season", "2026", "--week", "12"]) == 0
    assert "Warning" not in capsys.readouterr().out


def test_warning_lines_are_flagged_by_the_run_review():
    from src.utils.run_log import is_issue_line

    assert is_issue_line("[espn_rosters] Warning: wrote 0 injury rows for 2026 week 12; check the ESPN response")
