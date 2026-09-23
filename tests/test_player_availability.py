from __future__ import annotations

import pandas as pd

from src.player_availability import attach_player_availability, build_current_player_availability


def test_build_current_player_availability_bridges_espn_roster_to_gsis_ids(tmp_path):
    espn_rosters_path = tmp_path / "espn_rosters.parquet"
    espn_depthcharts_path = tmp_path / "espn_depthcharts.parquet"
    espn_injuries_path = tmp_path / "espn_injuries.parquet"
    nfl_rosters_path = tmp_path / "nfl_rosters.parquet"
    bdl_players_path = tmp_path / "bdl_players.parquet"

    pd.DataFrame(
        [
            {
                "espn_player_id": "100",
                "team": "ATL",
                "full_name": "QB One",
                "position": "QB",
                "roster_section": "offense",
                "status_type": "active",
                "status_name": "Active",
                "jersey_number": "1",
            },
            {
                "espn_player_id": "200",
                "team": "ATL",
                "full_name": "RB One",
                "position": "RB",
                "roster_section": "practiceSquad",
                "status_type": "active",
                "status_name": "Active",
                "jersey_number": "2",
            },
            {
                "espn_player_id": "300",
                "team": "CAR",
                "full_name": "DL One",
                "position": "DE",
                "roster_section": "injuredReserveOrOut",
                "status_type": "inactive",
                "status_name": "Injured Reserve",
                "jersey_number": "3",
            },
        ]
    ).to_parquet(espn_rosters_path, index=False)
    pd.DataFrame(
        [
            {
                "espn_player_id": "100",
                "team": "ATL",
                "depthchart_position": "QB",
                "depth_rank": 1,
            }
        ]
    ).to_parquet(espn_depthcharts_path, index=False)
    pd.DataFrame(
        [
            {
                "espn_player_id": "300",
                "team": "CAR",
                "injury_status": "Out",
            }
        ]
    ).to_parquet(espn_injuries_path, index=False)
    pd.DataFrame(
        [
            {"season": 2026, "week": 3, "gsis_id": "qb-1", "espn_id": 100, "team": "ATL"},
            {"season": 2026, "week": 3, "gsis_id": "rb-1", "espn_id": 200, "team": "ATL"},
            {"season": 2026, "week": 3, "gsis_id": "dl-1", "espn_id": 300, "team": "CAR"},
        ]
    ).to_parquet(nfl_rosters_path, index=False)
    pd.DataFrame().to_parquet(bdl_players_path, index=False)

    availability = build_current_player_availability(
        season=2026,
        week=3,
        espn_rosters_path=espn_rosters_path,
        espn_depthcharts_path=espn_depthcharts_path,
        espn_injuries_path=espn_injuries_path,
        nfl_rosters_path=nfl_rosters_path,
        balldontlie_active_players_path=bdl_players_path,
    )

    qb = availability[availability["player_id"].eq("qb-1")].iloc[0]
    rb = availability[availability["player_id"].eq("rb-1")].iloc[0]
    dl = availability[availability["player_id"].eq("dl-1")].iloc[0]
    assert qb["availability_source"] == "espn"
    assert qb["current_team"] == "ATL"
    assert bool(qb["active_roster_flag"]) is True
    assert qb["depth_chart_position"] == "QB"
    assert bool(rb["practice_squad_flag"]) is True
    assert bool(rb["active_roster_flag"]) is False
    assert bool(dl["injured_reserve_flag"]) is True
    assert dl["injury_status"] == "Out"


def test_attach_player_availability_marks_roster_validation_flags():
    predictions = pd.DataFrame(
        [
            {"player_id": "qb-1", "player_name": "QB One", "team": "ATL"},
            {"player_id": "rb-1", "player_name": "RB One", "team": "ATL"},
            {"player_id": "dl-1", "player_name": "DL One", "team": "ATL"},
            {"player_id": "wr-1", "player_name": "WR One", "team": "ATL"},
        ]
    )
    availability = pd.DataFrame(
        [
            {
                "player_id": "qb-1",
                "player_name_key": "q:one",
                "current_team": "ATL",
                "active_roster_flag": True,
                "practice_squad_flag": False,
                "injured_reserve_flag": False,
                "availability_source": "espn",
            },
            {
                "player_id": "rb-1",
                "player_name_key": "r:one",
                "current_team": "ATL",
                "active_roster_flag": False,
                "practice_squad_flag": True,
                "injured_reserve_flag": False,
                "availability_source": "espn",
            },
            {
                "player_id": "dl-1",
                "player_name_key": "d:one",
                "current_team": "CAR",
                "active_roster_flag": True,
                "practice_squad_flag": False,
                "injured_reserve_flag": False,
                "availability_source": "espn",
            },
        ]
    )

    attached = attach_player_availability(predictions, availability)

    flags = dict(zip(attached["player_id"], attached["roster_validation_flag"]))
    active = dict(zip(attached["player_id"], attached["active_current_roster"]))
    assert flags == {
        "qb-1": "ok",
        "rb-1": "practice_squad",
        "dl-1": "team_mismatch",
        "wr-1": "unknown",
    }
    assert bool(active["qb-1"]) is True
    assert bool(active["rb-1"]) is False
    assert bool(active["dl-1"]) is False
