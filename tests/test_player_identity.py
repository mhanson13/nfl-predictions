from __future__ import annotations

import pandas as pd

from src.data.player_identity import build_player_identity


def test_build_player_identity_links_roster_and_balldontlie_aliases():
    nfl_rosters = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "team": "BUF",
                "gsis_id": "gsis-ty",
                "espn_id": 1001,
                "sportradar_id": "sr-ty",
                "full_name": "Ty Johnson",
                "football_name": "Ty Johnson",
                "first_name": "Ty",
                "last_name": "Johnson",
                "position": "RB",
                "jersey_number": "26",
                "status": "ACT",
            },
            {
                "season": 2025,
                "week": 1,
                "team": "BUF",
                "gsis_id": "gsis-taron",
                "espn_id": 1002,
                "sportradar_id": "sr-taron",
                "full_name": "Taron Johnson",
                "football_name": "Taron Johnson",
                "first_name": "Taron",
                "last_name": "Johnson",
                "position": "CB",
                "jersey_number": "7",
                "status": "ACT",
            },
        ]
    )
    bdl_active = pd.DataFrame(
        [
            {
                "id": 5001,
                "first_name": "Ty",
                "last_name": "Johnson",
                "team.abbreviation": "BUF",
                "position_abbreviation": "RB",
                "jersey_number": "26",
            }
        ]
    )

    identity = build_player_identity(
        nfl_rosters=nfl_rosters,
        bdl_active_players=bdl_active,
        generated_at="2026-09-25T00:00:00+00:00",
    )

    ty = identity[
        identity["player_id"].eq("gsis-ty")
        & identity["alias_name_key"].eq("t:johnson")
        & identity["alias_prefix_key"].eq("ty:johnson")
    ]
    taron = identity[
        identity["player_id"].eq("gsis-taron")
        & identity["alias_name_key"].eq("t:johnson")
        & identity["alias_prefix_key"].eq("ta:johnson")
    ]

    assert not ty.empty
    assert not taron.empty
    assert set(ty["alias_source"]) >= {"nflverse_roster", "balldontlie_active_players"}
    assert ty["bdl_player_id"].dropna().iloc[0] == "5001"
    assert ty["position_group"].dropna().iloc[0] == "RB"
