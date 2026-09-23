from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.io import PROC_DIR, RAW_DIR, read_df, write_df
from src.utils.teams import normalize_team_abbr


DEFAULT_OUTPUT_PATH = PROC_DIR / "current_player_availability.parquet"
AVAILABILITY_COLUMNS = [
    "season",
    "week",
    "player_id",
    "espn_player_id",
    "player_name",
    "player_name_key",
    "team",
    "current_team",
    "position",
    "position_group",
    "jersey_number",
    "active_roster_flag",
    "practice_squad_flag",
    "injured_reserve_flag",
    "injury_reported",
    "injury_status",
    "depth_chart_flag",
    "depth_chart_position",
    "depth_chart_rank",
    "availability_source",
    "availability_priority",
]

ROSTER_VALIDATION_COLUMNS = [
    "current_team",
    "active_current_roster",
    "roster_validation_flag",
    "roster_validation_source",
    "availability_status",
    "depth_chart_position",
    "depth_chart_rank",
]


def _read_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return read_df(path)


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NAN", "NA", "<NA>", "NULL"}:
        return None
    return text


def _player_name_parts(value: object) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    if "," in text:
        parts = [part.strip() for part in text.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            text = f"{parts[1]} {parts[0]}"
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return text.split()


def _player_name_key(value: object) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first_initial = parts[0][0]
    last = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{first_initial}:{last}"


def _position_group(value: object) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    pos = text.upper()
    if pos in {"QB"}:
        return "QB"
    if pos in {"RB", "FB"}:
        return "RB"
    if pos in {"WR"}:
        return "WR"
    if pos in {"TE"}:
        return "TE"
    if pos in {"DE", "DT", "NT", "DL", "EDGE"}:
        return "DL"
    if pos in {"LB", "ILB", "OLB", "MLB"}:
        return "LB"
    if pos in {"CB", "DB", "S", "FS", "SS"}:
        return "DB"
    return pos


def _load_id_bridge(nfl_rosters: pd.DataFrame) -> pd.DataFrame:
    columns = ["gsis_id", "espn_id", "full_name", "football_name", "position"]
    if nfl_rosters.empty or not {"gsis_id", "espn_id"}.issubset(nfl_rosters.columns):
        return pd.DataFrame(columns=["player_id", "espn_player_id", "bridge_name", "bridge_position"])
    bridge = nfl_rosters[[col for col in columns if col in nfl_rosters.columns]].copy()
    bridge["player_id"] = _as_text(bridge["gsis_id"])
    bridge["espn_player_id"] = pd.to_numeric(bridge["espn_id"], errors="coerce").astype("Int64").astype("string")
    name = bridge.get("football_name", pd.Series(pd.NA, index=bridge.index))
    full = bridge.get("full_name", pd.Series(pd.NA, index=bridge.index))
    bridge["bridge_name"] = name.where(name.notna(), full)
    bridge["bridge_position"] = bridge.get("position", pd.Series(pd.NA, index=bridge.index))
    bridge = bridge.dropna(subset=["player_id", "espn_player_id"])
    return bridge[["player_id", "espn_player_id", "bridge_name", "bridge_position"]].drop_duplicates(
        "espn_player_id",
        keep="last",
    )


def _prepare_depth_lookup(depthcharts: pd.DataFrame) -> pd.DataFrame:
    if depthcharts.empty or "espn_player_id" not in depthcharts.columns:
        return pd.DataFrame(
            columns=["espn_player_id", "team", "depth_chart_flag", "depth_chart_position", "depth_chart_rank"]
        )
    depth = depthcharts.copy()
    depth["espn_player_id"] = _as_text(depth["espn_player_id"])
    depth["team"] = depth["team"].apply(lambda value: normalize_team_abbr(str(value)) if _clean_text(value) else None)
    depth["depth_chart_rank"] = pd.to_numeric(depth.get("depth_rank"), errors="coerce")
    depth = depth.dropna(subset=["espn_player_id", "team"])
    if depth.empty:
        return pd.DataFrame(
            columns=["espn_player_id", "team", "depth_chart_flag", "depth_chart_position", "depth_chart_rank"]
        )
    depth = depth.sort_values(["espn_player_id", "team", "depth_chart_rank"], kind="stable")
    depth = depth.drop_duplicates(["espn_player_id", "team"], keep="first")
    depth["depth_chart_flag"] = True
    return depth[
        ["espn_player_id", "team", "depth_chart_flag", "depthchart_position", "depth_chart_rank"]
    ].rename(columns={"depthchart_position": "depth_chart_position"}) if "depthchart_position" in depth.columns else depth[
        ["espn_player_id", "team", "depth_chart_flag", "depth_chart_position", "depth_chart_rank"]
    ]


def _prepare_espn_injury_lookup(injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries.empty or "espn_player_id" not in injuries.columns:
        return pd.DataFrame(columns=["espn_player_id", "team", "injury_reported", "injury_status"])
    out = injuries.copy()
    out["espn_player_id"] = _as_text(out["espn_player_id"])
    out["team"] = out["team"].apply(lambda value: normalize_team_abbr(str(value)) if _clean_text(value) else None)
    status = out.get("injury_status", pd.Series(pd.NA, index=out.index)).apply(_clean_text)
    detail = out.get("injury_detail", pd.Series(pd.NA, index=out.index)).apply(_clean_text)
    out["injury_status"] = status.where(status.notna(), detail)
    out["injury_reported"] = out["injury_status"].notna()
    out = out.dropna(subset=["espn_player_id"])
    if out.empty:
        return pd.DataFrame(columns=["espn_player_id", "team", "injury_reported", "injury_status"])
    return out[["espn_player_id", "team", "injury_reported", "injury_status"]].drop_duplicates(
        ["espn_player_id", "team"],
        keep="last",
    )


def _espn_availability(
    *,
    rosters: pd.DataFrame,
    depthcharts: pd.DataFrame,
    injuries: pd.DataFrame,
    nfl_rosters: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    if rosters.empty or "espn_player_id" not in rosters.columns:
        return pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    out = rosters.copy()
    out["season"] = int(season)
    out["week"] = int(week)
    out["espn_player_id"] = _as_text(out["espn_player_id"])
    out["team"] = out["team"].apply(lambda value: normalize_team_abbr(str(value)) if _clean_text(value) else None)
    out["current_team"] = out["team"]
    out["player_name"] = out.get("full_name", pd.Series(pd.NA, index=out.index)).where(
        out.get("full_name", pd.Series(pd.NA, index=out.index)).notna(),
        out.get("display_name", pd.Series(pd.NA, index=out.index)),
    )
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    out["position"] = out.get("position", pd.Series(pd.NA, index=out.index)).astype("string").str.upper()
    out["position_group"] = out["position"].apply(_position_group)
    section = out.get("roster_section", pd.Series("", index=out.index)).astype("string").str.lower()
    status_type = out.get("status_type", pd.Series("", index=out.index)).astype("string").str.lower()
    status_name = out.get("status_name", pd.Series("", index=out.index)).astype("string").str.lower()
    out["practice_squad_flag"] = section.str.contains("practice", na=False)
    out["injured_reserve_flag"] = section.str.contains("injured|reserve|out", na=False) | status_name.str.contains(
        "injured reserve|reserve|inactive|out",
        na=False,
    )
    out["active_roster_flag"] = (
        status_type.eq("active") | status_name.eq("active") | status_name.eq("")
    ) & ~out["practice_squad_flag"] & ~out["injured_reserve_flag"]

    bridge = _load_id_bridge(nfl_rosters)
    if not bridge.empty:
        out = out.merge(bridge, on="espn_player_id", how="left")
        out["player_name"] = out["player_name"].where(out["player_name"].notna(), out["bridge_name"])
        out["position"] = out["position"].where(out["position"].notna(), out["bridge_position"])
    else:
        out["player_id"] = pd.NA

    depth = _prepare_depth_lookup(depthcharts)
    if not depth.empty:
        out = out.merge(depth, on=["espn_player_id", "team"], how="left")
    for col, default in [
        ("depth_chart_flag", False),
        ("depth_chart_position", pd.NA),
        ("depth_chart_rank", np.nan),
    ]:
        if col not in out.columns:
            out[col] = default
    out["depth_chart_flag"] = out["depth_chart_flag"].astype("boolean").fillna(False).astype(bool)

    injury_lookup = _prepare_espn_injury_lookup(injuries)
    if not injury_lookup.empty:
        out = out.merge(injury_lookup, on=["espn_player_id", "team"], how="left")
    out["injury_reported"] = (
        out.get("injury_reported", pd.Series(False, index=out.index)).astype("boolean").fillna(False).astype(bool)
    )
    if "injury_status" not in out.columns:
        out["injury_status"] = pd.NA
    out["availability_source"] = "espn"
    out["availability_priority"] = 30
    return out.reindex(columns=AVAILABILITY_COLUMNS)


def _nflverse_availability(nfl_rosters: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    if nfl_rosters.empty or not {"season", "week", "gsis_id", "team"}.issubset(nfl_rosters.columns):
        return pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    out = nfl_rosters.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].eq(int(season)) & out["week"].le(int(week))]
    if out.empty:
        return pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    out = out.sort_values(["gsis_id", "week"], kind="stable").drop_duplicates("gsis_id", keep="last")
    out["player_id"] = _as_text(out["gsis_id"])
    espn_id = out.get("espn_id", pd.Series(pd.NA, index=out.index))
    out["espn_player_id"] = pd.to_numeric(espn_id, errors="coerce").astype("Int64").astype("string")
    out["player_name"] = out.get("football_name", pd.Series(pd.NA, index=out.index)).where(
        out.get("football_name", pd.Series(pd.NA, index=out.index)).notna(),
        out.get("full_name", pd.Series(pd.NA, index=out.index)),
    )
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    out["team"] = out["team"].apply(lambda value: normalize_team_abbr(str(value)) if _clean_text(value) else None)
    out["current_team"] = out["team"]
    out["position"] = out.get("position", pd.Series(pd.NA, index=out.index)).astype("string").str.upper()
    out["position_group"] = out["position"].apply(_position_group)
    status = out.get("status", pd.Series(pd.NA, index=out.index)).astype("string").str.upper()
    out["practice_squad_flag"] = status.str.contains("PRA|PRACTICE", na=False)
    out["injured_reserve_flag"] = status.str.contains("RES|IR|PUP|NFI|OUT|INA", na=False)
    out["active_roster_flag"] = status.eq("ACT") & ~out["practice_squad_flag"] & ~out["injured_reserve_flag"]
    out["injury_reported"] = False
    out["injury_status"] = pd.NA
    out["depth_chart_flag"] = out.get("depth_chart_position", pd.Series(pd.NA, index=out.index)).notna()
    out["depth_chart_rank"] = pd.to_numeric(out.get("depth_chart_order"), errors="coerce")
    out["availability_source"] = "nflverse"
    out["availability_priority"] = 20
    return out.reindex(columns=AVAILABILITY_COLUMNS)


def _bdl_availability(active_players: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    if active_players.empty or not {"first_name", "last_name"}.issubset(active_players.columns):
        return pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    out = active_players.copy()
    team_col = "team.abbreviation" if "team.abbreviation" in out.columns else None
    if team_col is None:
        return pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    out["season"] = int(season)
    out["week"] = int(week)
    out["player_id"] = pd.NA
    out["espn_player_id"] = pd.NA
    out["player_name"] = (
        out["first_name"].astype("string").fillna("")
        + " "
        + out["last_name"].astype("string").fillna("")
    ).str.strip()
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    out["team"] = out[team_col].apply(lambda value: normalize_team_abbr(str(value)) if _clean_text(value) else None)
    out["current_team"] = out["team"]
    out["position"] = out.get("position_abbreviation", pd.Series(pd.NA, index=out.index)).astype("string").str.upper()
    out["position_group"] = out["position"].apply(_position_group)
    out["jersey_number"] = out.get("jersey_number", pd.Series(pd.NA, index=out.index))
    out["active_roster_flag"] = True
    out["practice_squad_flag"] = False
    out["injured_reserve_flag"] = False
    out["injury_reported"] = False
    out["injury_status"] = pd.NA
    out["depth_chart_flag"] = False
    out["depth_chart_position"] = pd.NA
    out["depth_chart_rank"] = np.nan
    out["availability_source"] = "balldontlie"
    out["availability_priority"] = 10
    return out.reindex(columns=AVAILABILITY_COLUMNS)


def _deduplicate_availability(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=AVAILABILITY_COLUMNS)
    out = frame.copy()
    out["availability_priority"] = pd.to_numeric(out["availability_priority"], errors="coerce").fillna(0)
    out["depth_chart_rank"] = pd.to_numeric(out["depth_chart_rank"], errors="coerce")
    out = out.sort_values(
        ["availability_priority", "active_roster_flag", "depth_chart_flag", "depth_chart_rank"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    with_id = out.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="first")
    without_id = out[out["player_id"].isna()].drop_duplicates(["player_name_key", "current_team"], keep="first")
    combined = pd.concat([with_id, without_id], ignore_index=True, sort=False)
    return combined.reindex(columns=AVAILABILITY_COLUMNS)


def build_current_player_availability(
    *,
    season: int,
    week: int,
    espn_rosters_path: Path | None = None,
    espn_depthcharts_path: Path | None = None,
    espn_injuries_path: Path | None = None,
    nfl_rosters_path: Path | None = None,
    balldontlie_active_players_path: Path | None = None,
) -> pd.DataFrame:
    espn_rosters_path = espn_rosters_path or RAW_DIR / f"espn_rosters_{int(season)}_wk{int(week):02d}.parquet"
    espn_depthcharts_path = espn_depthcharts_path or RAW_DIR / f"espn_depthcharts_{int(season)}_wk{int(week):02d}.parquet"
    espn_injuries_path = espn_injuries_path or RAW_DIR / f"espn_injuries_{int(season)}_wk{int(week):02d}.parquet"
    nfl_rosters_path = nfl_rosters_path or RAW_DIR / "nfl_rosters.parquet"
    balldontlie_active_players_path = balldontlie_active_players_path or RAW_DIR / "balldontlie_active_players.parquet"

    nfl_rosters = _read_if_exists(nfl_rosters_path)
    frames = [
        _espn_availability(
            rosters=_read_if_exists(espn_rosters_path),
            depthcharts=_read_if_exists(espn_depthcharts_path),
            injuries=_read_if_exists(espn_injuries_path),
            nfl_rosters=nfl_rosters,
            season=season,
            week=week,
        ),
        _nflverse_availability(nfl_rosters, season=season, week=week),
        _bdl_availability(_read_if_exists(balldontlie_active_players_path), season=season, week=week),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    availability = pd.concat(frames, ignore_index=True, sort=False)
    return _deduplicate_availability(availability)


def _unique_name_lookup(availability: pd.DataFrame) -> pd.DataFrame:
    if availability.empty or "player_name_key" not in availability.columns:
        return pd.DataFrame()
    valid = availability.dropna(subset=["player_name_key"]).copy()
    counts = valid.groupby("player_name_key", dropna=False)["current_team"].nunique().reset_index(name="_team_count")
    valid = valid.merge(counts, on="player_name_key", how="left")
    valid = valid[valid["_team_count"].eq(1)]
    valid = valid.sort_values("availability_priority", ascending=False, kind="stable")
    return valid.drop_duplicates("player_name_key", keep="first").drop(columns=["_team_count"], errors="ignore")


def attach_player_availability(predictions: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    out = predictions.copy()
    out = out.drop(columns=ROSTER_VALIDATION_COLUMNS, errors="ignore")
    if availability.empty:
        out["current_team"] = pd.NA
        out["active_current_roster"] = pd.NA
        out["roster_validation_flag"] = "unknown"
        out["roster_validation_source"] = pd.NA
        out["availability_status"] = pd.NA
        out["depth_chart_position"] = pd.NA
        out["depth_chart_rank"] = np.nan
        return out

    avail_cols = [
        "player_id",
        "player_name_key",
        "current_team",
        "active_roster_flag",
        "practice_squad_flag",
        "injured_reserve_flag",
        "availability_source",
        "injury_status",
        "depth_chart_position",
        "depth_chart_rank",
        "availability_priority",
    ]
    avail = availability[[col for col in avail_cols if col in availability.columns]].copy()
    for col in avail_cols:
        if col not in avail.columns:
            avail[col] = pd.NA
    avail["availability_priority"] = pd.to_numeric(avail["availability_priority"], errors="coerce").fillna(0)
    out["_player_name_key_availability"] = out.get("player_name", pd.Series(pd.NA, index=out.index)).apply(_player_name_key)
    out["_team_key_availability"] = out.get("team", pd.Series(pd.NA, index=out.index)).apply(
        lambda value: normalize_team_abbr(str(value)) if _clean_text(value) else None
    )

    id_lookup = avail.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="first")
    if not id_lookup.empty and "player_id" in out.columns:
        out = out.merge(
            id_lookup.drop(columns=["player_name_key"], errors="ignore").rename(
                columns={
                    "active_roster_flag": "_availability_active",
                    "practice_squad_flag": "_availability_practice",
                    "injured_reserve_flag": "_availability_ir",
                    "availability_source": "roster_validation_source",
                    "injury_status": "availability_status",
                }
            ),
            on="player_id",
            how="left",
        )

    missing_mask = out.get("current_team", pd.Series(pd.NA, index=out.index)).isna()
    name_lookup = _unique_name_lookup(avail)
    if missing_mask.any() and not name_lookup.empty:
        fallback = out.loc[missing_mask, ["_player_name_key_availability"]].merge(
            name_lookup.rename(columns={"player_name_key": "_player_name_key_availability"})[
                [
                    "_player_name_key_availability",
                    "current_team",
                    "active_roster_flag",
                    "practice_squad_flag",
                    "injured_reserve_flag",
                    "availability_source",
                    "injury_status",
                    "depth_chart_position",
                    "depth_chart_rank",
                ]
            ],
            on="_player_name_key_availability",
            how="left",
        )
        for source_col, target_col in [
            ("current_team", "current_team"),
            ("active_roster_flag", "_availability_active"),
            ("practice_squad_flag", "_availability_practice"),
            ("injured_reserve_flag", "_availability_ir"),
            ("availability_source", "roster_validation_source"),
            ("injury_status", "availability_status"),
            ("depth_chart_position", "depth_chart_position"),
            ("depth_chart_rank", "depth_chart_rank"),
        ]:
            out.loc[missing_mask, target_col] = fallback[source_col].to_numpy()

    current_team = out.get("current_team", pd.Series(pd.NA, index=out.index))
    current_team_clean = current_team.apply(lambda value: normalize_team_abbr(str(value)) if _clean_text(value) else None)
    predicted_team = out["_team_key_availability"]
    matched = current_team_clean.notna()
    same_team = matched & predicted_team.eq(current_team_clean)
    active = out.get("_availability_active", pd.Series(pd.NA, index=out.index)).astype("boolean")
    practice = out.get("_availability_practice", pd.Series(False, index=out.index)).astype("boolean").fillna(False)
    injured_reserve = out.get("_availability_ir", pd.Series(False, index=out.index)).astype("boolean").fillna(False)

    flag = pd.Series("unknown", index=out.index, dtype="object")
    flag.loc[matched & ~same_team] = "team_mismatch"
    flag.loc[same_team & practice] = "practice_squad"
    flag.loc[same_team & injured_reserve] = "inactive_roster"
    flag.loc[same_team & active.fillna(False)] = "ok"
    flag.loc[same_team & active.notna() & ~active.fillna(False) & ~practice & ~injured_reserve] = "inactive_roster"

    out["current_team"] = current_team_clean
    out["active_current_roster"] = same_team & active.fillna(False)
    out["roster_validation_flag"] = flag
    if "roster_validation_source" not in out.columns:
        out["roster_validation_source"] = pd.NA
    if "availability_status" not in out.columns:
        out["availability_status"] = pd.NA
    if "depth_chart_position" not in out.columns:
        out["depth_chart_position"] = pd.NA
    if "depth_chart_rank" not in out.columns:
        out["depth_chart_rank"] = np.nan
    return out.drop(
        columns=[
            "_player_name_key_availability",
            "_team_key_availability",
            "_availability_active",
            "_availability_practice",
            "_availability_ir",
        ],
        errors="ignore",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build current-week player availability from ESPN, nflverse, and BallDontLie roster data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--espn-rosters", type=Path, default=None)
    parser.add_argument("--espn-depthcharts", type=Path, default=None)
    parser.add_argument("--espn-injuries", type=Path, default=None)
    parser.add_argument("--nfl-rosters", type=Path, default=RAW_DIR / "nfl_rosters.parquet")
    parser.add_argument("--balldontlie-active-players", type=Path, default=RAW_DIR / "balldontlie_active_players.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    availability = build_current_player_availability(
        season=args.season,
        week=args.week,
        espn_rosters_path=args.espn_rosters,
        espn_depthcharts_path=args.espn_depthcharts,
        espn_injuries_path=args.espn_injuries,
        nfl_rosters_path=args.nfl_rosters,
        balldontlie_active_players_path=args.balldontlie_active_players,
    )
    output = write_df(availability, args.output)
    if args.debug and not availability.empty:
        print("[player_availability] rows by source:")
        print(availability["availability_source"].value_counts(dropna=False).to_string())
        print("[player_availability] active/current counts:")
        print(availability["active_roster_flag"].value_counts(dropna=False).to_string())
    print(f"[player_availability] wrote rows={len(availability)} -> {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
