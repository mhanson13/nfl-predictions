from __future__ import annotations

import numpy as np
import pandas as pd

from src.player_props.predict import (
    _attach_edge_publish_status,
    _attach_vacated_role_flags,
    _model_signal_reason,
)


def _availability(rows):
    return pd.DataFrame(
        rows,
        columns=["player_id", "player_name", "team", "current_team", "depth_chart_position", "depth_chart_rank",
                 "injury_status", "injured_reserve_flag"],
    )


def _injuries(rows):
    return pd.DataFrame(rows, columns=["player_id", "player_name_key", "team", "injury_status"])


def _row(name, pid, *, team="KC", position="RB", market="rb_rushing_yards", line=73.5, projection=24.4, **extra):
    base = {
        "player_id": pid, "player_name": name, "team": team, "position": position,
        "depth_chart_position": position, "market": market, "line": line, "projection": projection,
        "line_type": "canonical", "prop_quality_flag": "ok", "bettable_flag": True,
        "edge_recommendation_flag": True, "edge_recommendation_tier": "strong", "edge_policy_threshold": 0.05,
        "injury_reported": False, "injury_exclusion_flag": False, "active_current_roster": True,
    }
    base.update(extra)
    return base


KC_DEPTH = _availability([
    ["00-walker", "Kenneth Walker III", "KC", "KC", "RB", 1, None, False],
    ["00-ott", "Jaydn Ott", "KC", "KC", "RB", 3, None, False],
    ["00-rice", "Rashee Rice", "KC", "KC", "WR", 1, None, False],
])


def test_backup_flagged_when_starter_questionable_and_line_far_above_projection():
    preds = pd.DataFrame([_row("Emmett Johnson", "00-johnson")])
    injuries = _injuries([["00-walker", "k:walker", "KC", "Questionable"]])

    out = _attach_vacated_role_flags(preds, injuries=injuries, availability=KC_DEPTH)

    assert bool(out.at[0, "vacated_role_flag"])
    assert out.at[0, "vacated_role_note"] == "KC RB1 Kenneth Walker III is Questionable"

    status = _attach_edge_publish_status(out)
    assert status.at[0, "edge_publish_status"] == "review"
    assert "Kenneth Walker III is Questionable" in status.at[0, "edge_reason"]
    assert _model_signal_reason(status.assign(public_board_status="review").iloc[0]).startswith(
        "Watchlist only: a starter at this position may not play"
    )


def test_not_flagged_when_line_close_to_projection_or_starter_healthy():
    injuries = _injuries([["00-walker", "k:walker", "KC", "Out"]])
    close = _attach_vacated_role_flags(
        pd.DataFrame([_row("Emmett Johnson", "00-johnson", line=45.5, projection=30.0)]),
        injuries=injuries, availability=KC_DEPTH,
    )
    healthy = _attach_vacated_role_flags(
        pd.DataFrame([_row("Emmett Johnson", "00-johnson")]),
        injuries=_injuries([["00-walker", "k:walker", "KC", None]]), availability=KC_DEPTH,
    )
    assert not close["vacated_role_flag"].any()
    assert not healthy["vacated_role_flag"].any()


def test_starter_not_flagged_for_own_absence_and_other_positions_unaffected():
    injuries = _injuries([["00-walker", "k:walker", "KC", "Questionable"]])
    preds = pd.DataFrame([
        _row("Kenneth Walker III", "00-walker", line=80.5, projection=30.0),
        _row("Rashee Rice", "00-rice", position="WR", market="wrte_receiving_yards", line=70.5, projection=20.0),
        _row("Other Team RB", "00-x", team="SEA", line=70.5, projection=20.0),
    ])

    out = _attach_vacated_role_flags(preds, injuries=injuries, availability=KC_DEPTH)

    assert not out["vacated_role_flag"].any()


def test_status_falls_back_to_name_match_and_ir_flag():
    depth = _availability([
        ["00-walker", "Kenneth Walker III", "KC", "KC", "RB", 1, None, False],
        ["00-kelce", "Travis Kelce", "KC", "KC", "TE", 1, None, True],
    ])
    injuries = _injuries([[np.nan, "k:walker", "KC", "Doubtful"]])  # report row without a gsis id
    preds = pd.DataFrame([
        _row("Emmett Johnson", "00-johnson"),
        _row("Noah Gray", "00-gray", position="TE", market="wrte_receiving_yards", line=40.5, projection=12.0),
    ])

    out = _attach_vacated_role_flags(preds, injuries=injuries, availability=depth)

    assert out["vacated_role_note"].tolist() == ["KC RB1 Kenneth Walker III is Doubtful", "KC TE1 Travis Kelce is IR"]


def test_own_injury_reason_wins_and_exclusion_still_unavailable():
    injuries = _injuries([["00-walker", "k:walker", "KC", "Out"]])
    preds = pd.DataFrame([
        _row("Emmett Johnson", "00-johnson", injury_reported=True),
        _row("Jaydn Ott", "00-ott", injury_exclusion_flag=True, injury_reported=True),
    ])

    status = _attach_edge_publish_status(_attach_vacated_role_flags(preds, injuries=injuries, availability=KC_DEPTH))

    assert status.at[0, "edge_publish_status"] == "review"
    assert status.at[0, "edge_reason"] == "Player has an injury report flag; review status before betting."
    assert status.at[1, "edge_publish_status"] == "unavailable"


def test_no_availability_means_no_flags():
    preds = pd.DataFrame([_row("Emmett Johnson", "00-johnson")])
    out = _attach_vacated_role_flags(preds, injuries=None, availability=pd.DataFrame())
    assert not out["vacated_role_flag"].any()
    assert _attach_edge_publish_status(out).at[0, "edge_publish_status"] == "strong"
