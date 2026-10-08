from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.utils.io import PROC_DIR, RAW_DIR, read_df, write_df
from src.utils.teams import normalize_team_abbr


DEFAULT_NFL_ROSTERS_PATH = RAW_DIR / "nfl_rosters.parquet"
DEFAULT_ESPN_ROSTERS_GLOB = "espn_rosters_*_wk*.parquet"
DEFAULT_BDL_PLAYERS_PATH = RAW_DIR / "balldontlie_players.parquet"
DEFAULT_BDL_ACTIVE_PLAYERS_PATH = RAW_DIR / "balldontlie_active_players.parquet"
DEFAULT_SPORTRADAR_ROSTERS_PATH = PROC_DIR / "sportradar" / "roster_players.parquet"
DEFAULT_OUTPUT_PATH = PROC_DIR / "player_identity.parquet"

IDENTITY_COLUMNS = [
    "season",
    "week",
    "team",
    "player_id",
    "canonical_name",
    "alias_name",
    "alias_source",
    "alias_confidence",
    "alias_name_key",
    "alias_prefix_key",
    "alias_full_key",
    "position",
    "position_group",
    "jersey_number",
    "status",
    "espn_player_id",
    "sportradar_id",
    "yahoo_id",
    "pfr_id",
    "bdl_player_id",
    "generated_at",
]


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_df(path)


def _clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NAN", "NA", "<NA>", "NULL"}:
        return None
    return text


def _as_text(value: object) -> str | None:
    return _clean_text(value)


def _as_int_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return _clean_text(value)


def _as_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _team(value: object) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return normalize_team_abbr(text)
    except Exception:
        return text.upper()


def _player_name_parts(value: object) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    if "," in text:
        parts = [part.strip() for part in text.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            text = f"{parts[1]} {parts[0]}"
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return str(text).strip().lower().split()


def _player_name_key(value: object) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first_initial = parts[0][0]
    last = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{first_initial}:{last}"


def _player_name_prefix_key(value: object, *, prefix_len: int = 2) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first = parts[0][: max(1, int(prefix_len))]
    last = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{first}:{last}"


def _player_full_key(value: object) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    return ":".join([parts[0], "".join(parts[1:]) if len(parts) > 1 else parts[0]])


def _position_group(value: object) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    pos = text.upper()
    if pos == "QB":
        return "QB"
    if pos in {"RB", "FB", "RUNNING BACK", "FULLBACK"}:
        return "RB"
    if pos in {"WR", "WIDE RECEIVER"}:
        return "WR"
    if pos in {"TE", "TIGHT END"}:
        return "TE"
    if pos in {"DE", "DT", "NT", "DL", "EDGE", "DEFENSIVE END", "DEFENSIVE TACKLE"}:
        return "DL"
    if pos in {"LB", "ILB", "OLB", "MLB", "LINEBACKER"}:
        return "LB"
    if pos in {"CB", "DB", "S", "FS", "SS", "SAF", "SAFETY", "CORNERBACK"}:
        return "DB"
    return pos


def _name_values(*values: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = " ".join(_player_name_parts(text))
        if not key or key in seen:
            continue
        out.append(text)
        seen.add(key)
    return out


def _make_alias_record(
    *,
    season: object,
    week: object,
    team: object,
    player_id: object,
    canonical_name: object,
    alias_name: object,
    alias_source: str,
    alias_confidence: float,
    position: object = None,
    jersey_number: object = None,
    status: object = None,
    espn_player_id: object = None,
    sportradar_id: object = None,
    yahoo_id: object = None,
    pfr_id: object = None,
    bdl_player_id: object = None,
    generated_at: str | None = None,
) -> dict[str, object] | None:
    alias = _clean_text(alias_name)
    if not alias:
        return None
    name_key = _player_name_key(alias)
    prefix_key = _player_name_prefix_key(alias)
    full_key = _player_full_key(alias)
    if not name_key or not prefix_key:
        return None
    season_num = _as_int(season)
    week_num = _as_int(week)
    if season_num is None or week_num is None:
        return None
    return {
        "season": season_num,
        "week": week_num,
        "team": _team(team),
        "player_id": _as_text(player_id),
        "canonical_name": _clean_text(canonical_name) or alias,
        "alias_name": alias,
        "alias_source": alias_source,
        "alias_confidence": float(alias_confidence),
        "alias_name_key": name_key,
        "alias_prefix_key": prefix_key,
        "alias_full_key": full_key,
        "position": (_clean_text(position) or "").upper() or None,
        "position_group": _position_group(position),
        "jersey_number": _as_text(jersey_number),
        "status": _clean_text(status),
        "espn_player_id": _as_int_text(espn_player_id),
        "sportradar_id": _as_text(sportradar_id),
        "yahoo_id": _as_text(yahoo_id),
        "pfr_id": _as_text(pfr_id),
        "bdl_player_id": _as_int_text(bdl_player_id),
        "generated_at": generated_at,
    }


def _records_from_nfl_rosters(rosters: pd.DataFrame, *, generated_at: str) -> list[dict[str, object]]:
    if rosters.empty or not {"season", "week", "team", "gsis_id"}.issubset(rosters.columns):
        return []
    records: list[dict[str, object]] = []
    for row in rosters.itertuples(index=False):
        data = row._asdict()
        canonical = _clean_text(data.get("football_name")) or _clean_text(data.get("full_name"))
        aliases = _name_values(
            data.get("full_name"),
            data.get("football_name"),
            " ".join(
                value
                for value in [_clean_text(data.get("first_name")), _clean_text(data.get("last_name"))]
                if value
            ),
        )
        for alias in aliases:
            record = _make_alias_record(
                season=data.get("season"),
                week=data.get("week"),
                team=data.get("team"),
                player_id=data.get("gsis_id"),
                canonical_name=canonical,
                alias_name=alias,
                alias_source="nflverse_roster",
                alias_confidence=1.0,
                position=data.get("position"),
                jersey_number=data.get("jersey_number"),
                status=data.get("status"),
                espn_player_id=data.get("espn_id"),
                sportradar_id=data.get("sportradar_id"),
                yahoo_id=data.get("yahoo_id"),
                pfr_id=data.get("pfr_id"),
                generated_at=generated_at,
            )
            if record:
                records.append(record)
    return records


def _nfl_bridge(rosters: pd.DataFrame) -> pd.DataFrame:
    if rosters.empty or "gsis_id" not in rosters.columns:
        return pd.DataFrame()
    keep = [
        col
        for col in [
            "season",
            "week",
            "team",
            "gsis_id",
            "espn_id",
            "sportradar_id",
            "full_name",
            "football_name",
            "position",
            "jersey_number",
            "status",
        ]
        if col in rosters.columns
    ]
    bridge = rosters[keep].copy()
    bridge["season"] = pd.to_numeric(bridge.get("season"), errors="coerce")
    bridge["week"] = pd.to_numeric(bridge.get("week"), errors="coerce")
    bridge["team"] = bridge.get("team", pd.Series(pd.NA, index=bridge.index)).apply(_team)
    bridge["player_id"] = bridge["gsis_id"].apply(_as_text)
    bridge["espn_player_id"] = bridge.get("espn_id", pd.Series(pd.NA, index=bridge.index)).apply(_as_int_text)
    bridge["sportradar_id"] = bridge.get("sportradar_id", pd.Series(pd.NA, index=bridge.index)).apply(_as_text)
    bridge["alias_full_key"] = bridge.get("full_name", pd.Series(pd.NA, index=bridge.index)).apply(_player_full_key)
    bridge = bridge.dropna(subset=["season", "week", "team", "player_id"])
    bridge = bridge.sort_values(["season", "week"], kind="stable")
    return bridge


def _records_from_espn_rosters(
    espn_rosters: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    generated_at: str,
) -> list[dict[str, object]]:
    if espn_rosters.empty or "espn_player_id" not in espn_rosters.columns:
        return []
    bridge = _nfl_bridge(rosters)
    espn = espn_rosters.copy()
    espn["espn_player_id"] = espn["espn_player_id"].apply(_as_int_text)
    if not bridge.empty and "espn_player_id" in bridge.columns:
        lookup = (
            bridge.dropna(subset=["espn_player_id"])
            .sort_values(["season", "week"], kind="stable")
            .drop_duplicates("espn_player_id", keep="last")
            [["espn_player_id", "player_id", "sportradar_id"]]
        )
        espn = espn.merge(lookup, on="espn_player_id", how="left")
    if "player_id" not in espn.columns:
        espn["player_id"] = pd.NA
    records: list[dict[str, object]] = []
    for row in espn.itertuples(index=False):
        data = row._asdict()
        canonical = _clean_text(data.get("full_name")) or _clean_text(data.get("display_name"))
        aliases = _name_values(data.get("full_name"), data.get("display_name"), data.get("short_name"))
        for alias in aliases:
            record = _make_alias_record(
                season=data.get("season"),
                week=data.get("week"),
                team=data.get("team"),
                player_id=data.get("player_id"),
                canonical_name=canonical,
                alias_name=alias,
                alias_source="espn_roster",
                alias_confidence=0.95 if _clean_text(data.get("player_id")) else 0.4,
                position=data.get("position"),
                jersey_number=data.get("jersey_number"),
                status=data.get("status_name"),
                espn_player_id=data.get("espn_player_id"),
                sportradar_id=data.get("sportradar_id"),
                generated_at=generated_at,
            )
            if record:
                records.append(record)
    return records


def _latest_bridge_candidates(rosters: pd.DataFrame) -> pd.DataFrame:
    bridge = _nfl_bridge(rosters)
    if bridge.empty:
        return bridge
    latest_season = int(bridge["season"].max())
    latest_week = int(bridge[bridge["season"].eq(latest_season)]["week"].max())
    out = bridge[bridge["season"].eq(latest_season) & bridge["week"].eq(latest_week)].copy()
    if "position" in out.columns:
        out["position_group"] = out["position"].apply(_position_group)
    return out


def _records_from_bdl_players(
    players: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    generated_at: str,
    source: str,
) -> list[dict[str, object]]:
    if players.empty or not {"first_name", "last_name"}.issubset(players.columns):
        return []
    out = players.copy()
    out["alias_name"] = (
        out["first_name"].astype("string").fillna("")
        + " "
        + out["last_name"].astype("string").fillna("")
    ).str.strip()
    team_col = "team.abbreviation" if "team.abbreviation" in out.columns else None
    out["team"] = out[team_col].apply(_team) if team_col else pd.NA
    out["position_group"] = out.get("position_abbreviation", out.get("position", pd.Series(pd.NA, index=out.index))).apply(
        _position_group
    )
    out["alias_full_key"] = out["alias_name"].apply(_player_full_key)
    bridge = _latest_bridge_candidates(rosters)
    latest_season = int(bridge["season"].max()) if not bridge.empty else None
    latest_week = int(bridge["week"].max()) if not bridge.empty else None
    if not bridge.empty and {"team", "alias_full_key", "player_id"}.issubset(bridge.columns):
        candidates = bridge.dropna(subset=["team", "alias_full_key", "player_id"]).copy()
        counts = (
            candidates.groupby(["team", "alias_full_key"], dropna=False)["player_id"]
            .nunique()
            .reset_index(name="_identity_candidates")
        )
        candidates = candidates.merge(counts, on=["team", "alias_full_key"], how="left")
        candidates = candidates[candidates["_identity_candidates"].eq(1)].copy()
        out = out.merge(
            candidates[
                ["team", "alias_full_key", "player_id", "espn_player_id", "sportradar_id", "position", "status"]
            ].drop_duplicates(["team", "alias_full_key"], keep="last"),
            on=["team", "alias_full_key"],
            how="left",
        )
    records: list[dict[str, object]] = []
    for row in out.itertuples(index=False):
        data = row._asdict()
        record = _make_alias_record(
            season=latest_season,
            week=latest_week,
            team=data.get("team"),
            player_id=data.get("player_id"),
            canonical_name=data.get("alias_name"),
            alias_name=data.get("alias_name"),
            alias_source=source,
            alias_confidence=0.75 if _clean_text(data.get("player_id")) else 0.35,
            position=data.get("position_abbreviation") or data.get("position"),
            jersey_number=data.get("jersey_number"),
            status=data.get("status"),
            espn_player_id=data.get("espn_player_id"),
            sportradar_id=data.get("sportradar_id"),
            bdl_player_id=data.get("id"),
            generated_at=generated_at,
        )
        if record:
            records.append(record)
    return records


def _records_from_sportradar_rosters(
    sportradar: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    generated_at: str,
) -> list[dict[str, object]]:
    if sportradar.empty or "player_id" not in sportradar.columns:
        return []
    bridge = _latest_bridge_candidates(rosters)
    sr = sportradar.copy()
    sr["sportradar_id"] = sr["player_id"].apply(_as_text)
    sr["team"] = sr.get("team_alias", pd.Series(pd.NA, index=sr.index)).apply(_team)
    sr["alias_name"] = sr.get("player_name", pd.Series(pd.NA, index=sr.index))
    sr["alias_full_key"] = sr["alias_name"].apply(_player_full_key)
    latest_season = int(bridge["season"].max()) if not bridge.empty else None
    latest_week = int(bridge["week"].max()) if not bridge.empty else None
    if not bridge.empty and "sportradar_id" in bridge.columns:
        sr_lookup = bridge.dropna(subset=["sportradar_id"]).drop_duplicates("sportradar_id", keep="last")
        sr = sr.merge(
            sr_lookup[["sportradar_id", "player_id", "espn_player_id"]],
            on="sportradar_id",
            how="left",
        )
    if "player_id_y" in sr.columns and "player_id_x" in sr.columns:
        sr["player_id"] = sr["player_id_y"].where(sr["player_id_y"].notna(), pd.NA)
    elif "player_id_y" in sr.columns:
        sr["player_id"] = sr["player_id_y"]
    elif "player_id" in sr.columns:
        sr["player_id"] = pd.NA
    records: list[dict[str, object]] = []
    for row in sr.itertuples(index=False):
        data = row._asdict()
        record = _make_alias_record(
            season=latest_season,
            week=latest_week,
            team=data.get("team"),
            player_id=data.get("player_id"),
            canonical_name=data.get("alias_name"),
            alias_name=data.get("alias_name"),
            alias_source="sportradar_roster",
            alias_confidence=0.9 if _clean_text(data.get("player_id")) else 0.35,
            position=data.get("position"),
            jersey_number=data.get("jersey"),
            status=data.get("status"),
            espn_player_id=data.get("espn_player_id"),
            sportradar_id=data.get("sportradar_id"),
            generated_at=generated_at,
        )
        if record:
            records.append(record)
    return records


def build_player_identity(
    *,
    nfl_rosters: pd.DataFrame,
    espn_rosters: pd.DataFrame | None = None,
    bdl_players: pd.DataFrame | None = None,
    bdl_active_players: pd.DataFrame | None = None,
    sportradar_rosters: pd.DataFrame | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    espn_rosters = espn_rosters if espn_rosters is not None else pd.DataFrame()
    bdl_players = bdl_players if bdl_players is not None else pd.DataFrame()
    bdl_active_players = bdl_active_players if bdl_active_players is not None else pd.DataFrame()
    sportradar_rosters = sportradar_rosters if sportradar_rosters is not None else pd.DataFrame()

    records: list[dict[str, object]] = []
    records.extend(_records_from_nfl_rosters(nfl_rosters, generated_at=generated_at))
    records.extend(_records_from_espn_rosters(espn_rosters, nfl_rosters, generated_at=generated_at))
    records.extend(
        _records_from_bdl_players(
            bdl_players,
            nfl_rosters,
            generated_at=generated_at,
            source="balldontlie_players",
        )
    )
    records.extend(
        _records_from_bdl_players(
            bdl_active_players,
            nfl_rosters,
            generated_at=generated_at,
            source="balldontlie_active_players",
        )
    )
    records.extend(_records_from_sportradar_rosters(sportradar_rosters, nfl_rosters, generated_at=generated_at))

    if not records:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)
    out = pd.DataFrame(records)
    for col in IDENTITY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[IDENTITY_COLUMNS].copy()
    out["alias_confidence"] = pd.to_numeric(out["alias_confidence"], errors="coerce").fillna(0.0)
    out = out.sort_values(
        ["season", "week", "team", "alias_name_key", "alias_prefix_key", "alias_confidence"],
        ascending=[True, True, True, True, True, False],
        kind="stable",
    )
    out = out.drop_duplicates(
        [
            "season",
            "week",
            "team",
            "player_id",
            "alias_source",
            "alias_name_key",
            "alias_prefix_key",
            "alias_full_key",
        ],
        keep="first",
    )
    return out.reset_index(drop=True)


def _load_espn_rosters(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob(pattern)):
        frame = _read_optional(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cross-source NFL player identity aliases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nfl-rosters", type=Path, default=DEFAULT_NFL_ROSTERS_PATH)
    parser.add_argument("--espn-rosters-glob", default=DEFAULT_ESPN_ROSTERS_GLOB)
    parser.add_argument("--balldontlie-players", type=Path, default=DEFAULT_BDL_PLAYERS_PATH)
    parser.add_argument("--balldontlie-active-players", type=Path, default=DEFAULT_BDL_ACTIVE_PLAYERS_PATH)
    parser.add_argument("--sportradar-rosters", type=Path, default=DEFAULT_SPORTRADAR_ROSTERS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    identity = build_player_identity(
        nfl_rosters=_read_optional(args.nfl_rosters),
        espn_rosters=_load_espn_rosters(args.espn_rosters_glob),
        bdl_players=_read_optional(args.balldontlie_players),
        bdl_active_players=_read_optional(args.balldontlie_active_players),
        sportradar_rosters=_read_optional(args.sportradar_rosters),
    )
    output_path = write_df(identity, args.output)
    if args.debug:
        print("[player_identity] aliases by source:")
        if identity.empty:
            print("<empty>")
        else:
            print(identity.groupby("alias_source").size().sort_values(ascending=False).to_string())
    print(f"[player_identity] wrote rows={len(identity)} -> {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
