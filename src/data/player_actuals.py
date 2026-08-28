# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.io import PROC_DIR, RAW_DIR, write_df
from src.utils.pydantic_schemas import validate_dataframe
from src.utils.teams import normalize_team_abbr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-game player actuals from NFLverse play-by-play.")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Seasons to include (e.g., 2002 2003 ... 2025).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROC_DIR / "player_actuals.parquet",
        help="Destination parquet file for the aggregated results.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def _log(msg: str, *, debug: bool) -> None:
    if debug:
        print(f"[player_actuals] {msg}")


def _load_pbp(columns: Iterable[str], *, debug: bool) -> pd.DataFrame:
    pbp_path = RAW_DIR / "nfl_pbp.parquet"
    if not pbp_path.exists():
        raise FileNotFoundError(f"Expected play-by-play data at {pbp_path}")
    _log(f"reading {pbp_path}", debug=debug)
    return pd.read_parquet(pbp_path, columns=list(columns), engine="fastparquet")


def _clean_team(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda v: normalize_team_abbr(v) if v and v != "NONE" else pd.NA)
    )


def _prep_players(df: pd.DataFrame, name_col: str, team_col: str, player_id_col: str) -> pd.DataFrame:
    out = df[[name_col, team_col, player_id_col, "season", "week", "game_id"]].copy()
    out = out.rename(columns={name_col: "player_name", player_id_col: "player_id", team_col: "team_alias"})
    out = out.dropna(subset=["player_name", "team_alias"])
    out["team_alias"] = _clean_team(out["team_alias"])
    return out.dropna(subset=["team_alias"])


def _aggregate_offense(
    pbp: pd.DataFrame, *, debug: bool
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    # Rushing yards
    rush = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_name"].notna() & pbp["posteam"].notna()].copy()
    if not rush.empty:
        rush = rush.rename(
            columns={"rusher_player_name": "player_name", "rusher_player_id": "player_id", "posteam": "team_alias"}
        )
        rush["team_alias"] = _clean_team(rush["team_alias"])
        rush = rush.dropna(subset=["team_alias"])
        rush_agg = (
            rush.groupby(["season", "week", "game_id", "team_alias", "player_id", "player_name"], as_index=False)[
                "rushing_yards"
            ]
            .sum(min_count=1)
            .rename(columns={"rushing_yards": "value"})
        )
        rush_agg["stat_category"] = "rushing"
        rush_agg["stats"] = rush_agg["value"].apply(lambda val: json.dumps({"yards": float(val)}))
        records.append(rush_agg)

    # Receiving yards (completed passes only)
    rec = pbp[
        (pbp["complete_pass"] == 1)
        & pbp["receiver_player_name"].notna()
        & pbp["posteam"].notna()
        & pbp["receiving_yards"].notna()
    ].copy()
    if not rec.empty:
        rec = rec.rename(
            columns={"receiver_player_name": "player_name", "receiver_player_id": "player_id", "posteam": "team_alias"}
        )
        rec["team_alias"] = _clean_team(rec["team_alias"])
        rec = rec.dropna(subset=["team_alias"])
        rec_agg = (
            rec.groupby(["season", "week", "game_id", "team_alias", "player_id", "player_name"], as_index=False)[
                "receiving_yards"
            ]
            .sum(min_count=1)
            .rename(columns={"receiving_yards": "value"})
        )
        rec_agg["stat_category"] = "receiving"
        rec_agg["stats"] = rec_agg["value"].apply(lambda val: json.dumps({"yards": float(val)}))
        records.append(rec_agg)

    # Passing yards
    pas = pbp[
        (pbp["pass_attempt"] == 1)
        & pbp["passer_player_name"].notna()
        & pbp["posteam"].notna()
        & pbp["passing_yards"].notna()
    ].copy()
    if not pas.empty:
        pas = pas.rename(
            columns={"passer_player_name": "player_name", "passer_player_id": "player_id", "posteam": "team_alias"}
        )
        pas["team_alias"] = _clean_team(pas["team_alias"])
        pas = pas.dropna(subset=["team_alias"])
        pas_agg = (
            pas.groupby(["season", "week", "game_id", "team_alias", "player_id", "player_name"], as_index=False)[
                "passing_yards"
            ]
            .sum(min_count=1)
            .rename(columns={"passing_yards": "value"})
        )
        pas_agg["stat_category"] = "passing"
        pas_agg["stats"] = pas_agg["value"].apply(lambda val: json.dumps({"yards": float(val)}))
        records.append(pas_agg)

    if not records:
        return pd.DataFrame(columns=["season", "week", "game_id", "team_alias", "player_id", "player_name", "stat_category", "stats"])

    offense = pd.concat(records, ignore_index=True)
    offense = offense.drop(columns=["value"])
    _log(f"aggregated offense rows={len(offense)}", debug=debug)
    return offense


def _aggregate_defense(pbp: pd.DataFrame, *, debug: bool) -> pd.DataFrame:
    def _extract_players(source_col: str, value_col: str, value: float) -> pd.DataFrame:
        if source_col not in pbp.columns:
            return pd.DataFrame()
        id_col = source_col.replace("_name", "_id")
        cols = ["season", "week", "game_id", "defteam", source_col]
        if id_col in pbp.columns:
            cols.append(id_col)
        sub = pbp[pbp[source_col].notna() & pbp["defteam"].notna()][cols].copy()
        if sub.empty:
            return pd.DataFrame()
        rename_map = {source_col: "player_name", "defteam": "team_alias"}
        if id_col in sub.columns:
            rename_map[id_col] = "player_id"
        sub = sub.rename(columns=rename_map)
        if "player_id" not in sub.columns:
            sub["player_id"] = pd.NA
        sub["team_alias"] = _clean_team(sub["team_alias"])
        sub = sub.dropna(subset=["team_alias", "player_name"])
        sub[value_col] = value
        return sub

    pieces = []
    pieces.append(_extract_players("sack_player_name", "sacks", 1.0))
    pieces.append(_extract_players("half_sack_1_player_name", "sacks", 0.5))
    pieces.append(_extract_players("half_sack_2_player_name", "sacks", 0.5))
    pieces.append(_extract_players("qb_hit_1_player_name", "qb_hits", 1.0))
    pieces.append(_extract_players("qb_hit_2_player_name", "qb_hits", 1.0))
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return pd.DataFrame(columns=["season", "week", "game_id", "team_alias", "player_id", "player_name", "stat_category", "stats"])

    defense = pd.concat(pieces, ignore_index=True)
    agg = (
        defense.groupby(["season", "week", "game_id", "team_alias", "player_name"], as_index=False)[["sacks", "qb_hits"]]
        .sum(min_count=1)
    )
    agg["player_id"] = pd.NA
    agg["stat_category"] = "defense"

    def _to_stats(row: pd.Series) -> str:
        payload = {}
        if pd.notna(row.get("sacks")) and row["sacks"]:
            payload["sacks"] = float(row["sacks"])
        if pd.notna(row.get("qb_hits")) and row["qb_hits"]:
            payload["qb_hits"] = float(row["qb_hits"])
        return json.dumps(payload or {"sacks": 0.0})

    agg["stats"] = agg.apply(_to_stats, axis=1)
    agg = agg[["season", "week", "game_id", "team_alias", "player_id", "player_name", "stat_category", "stats"]]
    _log(f"aggregated defense rows={len(agg)}", debug=debug)
    return agg


def build_player_actuals(seasons: list[int], *, output: Path, debug: bool) -> None:
    cols = [
        "season",
        "week",
        "game_id",
        "posteam",
        "defteam",
        "rush_attempt",
        "rusher_player_name",
        "rusher_player_id",
        "rushing_yards",
        "receiving_yards",
        "receiver_player_name",
        "receiver_player_id",
        "complete_pass",
        "pass_attempt",
        "passer_player_name",
        "passer_player_id",
        "passing_yards",
        "sack",
        "sack_player_name",
        "sack_player_id",
        "half_sack_1_player_name",
        "half_sack_1_player_id",
        "half_sack_2_player_name",
        "half_sack_2_player_id",
        "qb_hit",
        "qb_hit_1_player_name",
        "qb_hit_1_player_id",
        "qb_hit_2_player_name",
        "qb_hit_2_player_id",
    ]
    pbp = _load_pbp(cols, debug=debug)
    pbp = pbp[pbp["season"].isin(seasons)].copy()
    pbp["week"] = pd.to_numeric(pbp["week"], errors="coerce").astype("Int64")

    offense = _aggregate_offense(pbp, debug=debug)
    defense = _aggregate_defense(pbp, debug=debug)

    frames = [df for df in (offense, defense) if not df.empty]
    if not frames:
        raise RuntimeError("No player actuals produced from play-by-play data.")

    result = pd.concat(frames, ignore_index=True)
    result["season"] = result["season"].astype(int)
    keep_cols = ["season", "week", "game_id", "team_alias", "player_id", "player_name", "stat_category", "stats"]
    result = result[keep_cols].sort_values(["season", "week", "game_id", "team_alias"])
    output.parent.mkdir(parents=True, exist_ok=True)
    report = validate_dataframe(result, "player_actuals", log_errors=False)
    _log(f"schema validation: {report}", debug=debug)
    _log(f"writing {len(result)} rows to {output}", debug=debug)
    write_df(result, output)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_player_actuals(args.seasons, output=args.output, debug=args.debug)
    except Exception as exc:
        print(f"[player_actuals] failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
