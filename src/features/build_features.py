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

"""Feature engineering pipeline that merges schedule, weather, roster, and team statistics."""



from __future__ import annotations



import argparse

import json

import re

import uuid

from datetime import timedelta

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple



import numpy as np

import pandas as pd



from src.features.adjusted_efficiency import build_adjusted_efficiency_features
from src.features.passing_epa_features import build_passing_epa_features
from src.features.drive_epa_features import build_drive_epa_features
from src.features.pressure_features import build_pressure_features
from src.features.qb_health import build_qb_health_features
from src.features.redzone_features import build_redzone_features
from src.features.volatility import engineer_volatility_inputs
from src.data.weather_fallback import get_fallback_weather
from src.utils.io import RAW_DIR, PROC_DIR, write_df, read_df
from src.utils.logging_config import setup_logging
from src.utils.teams import normalize_team_abbr, TEAM_NAME_TO_ABBR

EARTH_RADIUS_KM = 6371.0

# Alias for backward compatibility with existing code
_norm_abbr = normalize_team_abbr



# =========================== Utilities ===========================



def _coalesce(cols: list[str], df: pd.DataFrame, default=None):

    for c in cols:

        if c in df.columns:

            return df[c]

    return default





def _to_int(val: Any) -> Optional[int]:

    try:

        if pd.isna(val):

            return None

        return int(val)

    except Exception:

        return None





def _detect_team_col(df: pd.DataFrame) -> Optional[str]:

    for c in [

        "team",

        "team_abbr",

        "team_abbreviation",

        "abbr",

        "club",

        "club_code",

        "team_code",

    ]:

        if c in df.columns:

            return c

    return None



def _safe_json_load(v: Any) -> Any:

    if isinstance(v, str) and v and v[0] in "{[":

        try:

            return json.loads(v)

        except Exception:

            return v

    return v



def _extract_abbr_from_jsonlike(series: pd.Series) -> pd.Series:

    """

    Extract 'abbreviation' from JSON-like strings/objects in schedule columns.

    Falls back to original scalar when already an abbr (e.g., 'KC').

    """

    def _extract(v: Any) -> Any:

        v = _safe_json_load(v)

        if isinstance(v, str) and 2 <= len(v) <= 4 and v.isupper():

            return v

        if isinstance(v, dict):

            if "abbreviation" in v and isinstance(v["abbreviation"], str):

                return v["abbreviation"]

            t = v.get("team") if isinstance(v.get("team"), dict) else None

            if t and "abbreviation" in t and isinstance(t["abbreviation"], str):

                return t["abbreviation"]

        return v

    return series.apply(_extract)





def _resolve_sportradar_abbr(alias: Optional[str], name: Optional[str]) -> Optional[str]:

    if isinstance(alias, str) and alias:

        return alias.upper()

    if isinstance(name, str):

        return TEAM_NAME_TO_ABBR.get(name.strip())

    return None





# =========================== Schedule / weather helpers ===========================



def _load_schedule_data(path: Path, seasons: List[int]) -> pd.DataFrame:

    """Read the schedule parquet and filter to the requested seasons if the column exists."""

    sched = read_df(path)

    if "season" in sched.columns and seasons:

        sched = sched[sched["season"].isin(seasons)].copy()

    return sched





def _mk_uid(df: pd.DataFrame) -> pd.Series:

    """Deterministic UUID keyed by season/week/home/away for joining disparate feeds."""

    def _one(r: pd.Series) -> Optional[str]:

        try:

            s_val = int(r.get("season")) if pd.notna(r.get("season")) else -1

            w_val = int(r.get("week")) if pd.notna(r.get("week")) else -1

            ht = str(r.get("home_team") or "").upper()

            at = str(r.get("away_team") or "").upper()

            key = f"{s_val}|{w_val}|{ht}|{at}"

            return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{key}"))

        except Exception:

            return None



    return df.apply(_one, axis=1)





def _mk_uid_relaxed(df: pd.DataFrame) -> pd.Series:

    """Variant of _mk_uid that collapses LA franchises to a single token for noisy feeds."""

    def _relax(ab: str) -> str:

        ab = (ab or "").upper()

        return "LA" if ab in ("LA", "LAR", "LAC") else ab



    def _one(r: pd.Series) -> Optional[str]:

        try:

            s_val = int(r.get("season")) if pd.notna(r.get("season")) else -1

            w_val = int(r.get("week")) if pd.notna(r.get("week")) else -1

            ht = _relax(str(r.get("home_team")))

            at = _relax(str(r.get("away_team")))

            key = f"{s_val}|{w_val}|{ht}|{at}"

            return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{key}"))

        except Exception:

            return None



    return df.apply(_one, axis=1)





def _load_weather_sources() -> pd.DataFrame:

    """

    Load all available weather parquet sources (Visual Crossing, Tomorrow.io, NOAA) and

    collapse them to a single frame with a priority ordering.

    """

    weather_frames: List[Tuple[int, pd.DataFrame]] = []



    visual_path = PROC_DIR / "visualcrossing_weather.parquet"

    if visual_path.exists():

        vx_weather = read_df(visual_path)

        if not vx_weather.empty:

            vx_weather = vx_weather.copy()

            for col in ("season", "week"):

                if col in vx_weather.columns:

                    vx_weather[col] = pd.to_numeric(vx_weather[col], errors="coerce").astype("Int64")

            if "game_id" in vx_weather.columns:

                vx_weather["game_id"] = vx_weather["game_id"].astype(str)

            for team_col in ("home_team", "away_team"):

                if team_col in vx_weather.columns:

                    vx_weather[team_col] = vx_weather[team_col].astype(str).str.upper().str.strip()

            if "weather_source" not in vx_weather.columns:

                vx_weather["weather_source"] = "visualcrossing"

            weather_frames.append((-1, vx_weather))



    weather_path = PROC_DIR / "weather_games.parquet"

    if weather_path.exists():

        base_weather = read_df(weather_path)

        if not base_weather.empty:

            if "weather_source" not in base_weather.columns:

                base_weather = base_weather.copy()

                base_weather["weather_source"] = "tomorrow"

            weather_frames.append((0, base_weather))



    noaa_path = PROC_DIR / "noaa_weather.parquet"

    if noaa_path.exists():

        noaa_weather = read_df(noaa_path)

        if not noaa_weather.empty:

            noaa_weather = noaa_weather.copy()

            if "game_id" in noaa_weather.columns:

                noaa_weather["game_id"] = noaa_weather["game_id"].astype(str)

            for col in ("season", "week"):

                if col in noaa_weather.columns:

                    noaa_weather[col] = pd.to_numeric(noaa_weather[col], errors="coerce").astype("Int64")

            for team_col in ("home_team", "away_team"):

                if team_col in noaa_weather.columns:

                    noaa_weather[team_col] = (

                        noaa_weather[team_col].astype(str).str.upper().str.strip()

                    )

            rename_map = {

                "noaa_temp_f": "weather_temp_f",

                "noaa_wind_mph": "weather_wind_mph",

                "noaa_wind_gust_mph": "weather_windgust_mph",

                "noaa_relative_humidity_pct": "weather_humidity_pct",

                "noaa_is_windy": "weather_is_windy",

                "noaa_is_cold": "weather_is_cold",

                "noaa_is_hot": "weather_is_hot",

                "noaa_is_precip": "weather_is_precip",

            }

            for src, dest in rename_map.items():

                if src in noaa_weather.columns and dest not in noaa_weather.columns:

                    noaa_weather[dest] = pd.to_numeric(noaa_weather[src], errors="coerce")

            if "noaa_precip_mm_total" in noaa_weather.columns and "weather_precip_mm_total" not in noaa_weather.columns:

                noaa_weather["weather_precip_mm_total"] = pd.to_numeric(

                    noaa_weather["noaa_precip_mm_total"], errors="coerce"

                )

            if "noaa_precip_mm" in noaa_weather.columns and "weather_precip_mm" not in noaa_weather.columns:

                noaa_weather["weather_precip_mm"] = pd.to_numeric(

                    noaa_weather["noaa_precip_mm"], errors="coerce"

                )

            if "weather_precip_mm" in noaa_weather.columns and "weather_precip_intensity_inph" not in noaa_weather.columns:

                noaa_weather["weather_precip_intensity_inph"] = (

                    pd.to_numeric(noaa_weather["weather_precip_mm"], errors="coerce") / 25.4

                )

            if "start_utc" in noaa_weather.columns and "kickoff" not in noaa_weather.columns:

                noaa_weather["kickoff"] = pd.to_datetime(

                    noaa_weather["start_utc"], errors="coerce", utc=True

                )

            if "weather_temp_f" in noaa_weather.columns and "weather_temp_kickoff_f" not in noaa_weather.columns:

                noaa_weather["weather_temp_kickoff_f"] = noaa_weather["weather_temp_f"]

            if "weather_source" not in noaa_weather.columns:

                noaa_weather["weather_source"] = "noaa"

            weather_frames.append((1, noaa_weather))



    if not weather_frames:

        return pd.DataFrame()



    weather_frames.sort(key=lambda item: item[0])

    weather_parts = [frame.dropna(axis=1, how="all") for _, frame in weather_frames if not frame.empty]
    if not weather_parts:
        return pd.DataFrame()
    weather_df = pd.concat(weather_parts, ignore_index=True, sort=False)

    weather_df["_weather_rank"] = (

        weather_df.get("weather_source")

        .map({"visualcrossing": -1, "tomorrow": 0, "noaa": 1})

        .fillna(0)

    )

    if "game_id" in weather_df.columns:

        weather_df = weather_df.sort_values(["game_id", "_weather_rank"])

        weather_df = weather_df.drop_duplicates(subset=["game_id"], keep="last")

    elif all(col in weather_df.columns for col in ["game_uid", "season", "week", "home_team", "away_team"]):

        weather_df = weather_df.sort_values(["game_uid", "_weather_rank"])

        weather_df = weather_df.drop_duplicates(subset=["game_uid"], keep="last")

    elif all(col in weather_df.columns for col in ["season", "week", "home_team", "away_team"]):

        weather_df = weather_df.sort_values(["season", "week", "home_team", "away_team", "_weather_rank"])

        weather_df = weather_df.drop_duplicates(subset=["season", "week", "home_team", "away_team"], keep="last")

    weather_df = weather_df.drop(columns=["_weather_rank"], errors="ignore")

    return weather_df





def _merge_weather_features(feats: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:

    """

    Merge consolidated weather data into the matchup features using cascading keys.

    Mirrors the original multi-step join strategy (game_id, game_uid, relaxed UID, fallback season/week).

    """

    if feats.empty or weather_df.empty:

        return feats



    w = weather_df.copy()



    # Diagnostics: summarize key coverage without expensive set operations
    try:
        wx_key_cols = [
            c
            for c in [
                "season",
                "week",
                "home_team",
                "away_team",
                "game_id",
                "game_uid",
                "game_uid_relaxed",
                "kickoff",
                "venue_lat",
                "venue_lon",
            ]
            if c in w.columns
        ]
        print(f"[weather][diag] weather rows={len(w)} cols={len(w.columns)}")
        print("[weather][diag] key columns:", wx_key_cols)
        print("[weather][diag] weather head:\n", w[wx_key_cols].head())
        for key in ("game_id", "game_uid", "game_uid_relaxed"):
            if key in w.columns and key in feats.columns:
                overlap = w[key].astype(str).isin(feats[key].astype(str)).sum()
                print(f"[weather][diag] overlap via {key}: {overlap}")
        if "season" in w.columns and "week" in w.columns:
            recent = (
                w.groupby(["season", "week"])
                .size()
                .reset_index(name="wx_games")
                .sort_values(["season", "week"], ascending=[False, False])
                .head(3)
            )
            print("[weather][diag] recent season/week coverage:\n", recent)
    except Exception as _e_diag:
        print("[weather][diag] diagnostics error:", _e_diag)


    key_cols = [c for c in ["game_id", "season", "week", "home_team", "away_team"] if c in feats.columns and c in w.columns]

    if "game_id" in feats.columns and "game_id" in w.columns:

        feats["game_id"] = feats["game_id"].astype(str)

        w["game_id"] = w["game_id"].astype(str)



    if "game_uid" in w.columns and "game_uid" not in feats.columns:

        try:

            feats["game_uid"] = _mk_uid(feats)

        except Exception:

            pass

    if "game_uid_relaxed" in w.columns and "game_uid_relaxed" not in feats.columns:

        try:

            feats["game_uid_relaxed"] = _mk_uid_relaxed(feats)

        except Exception:

            pass



    if "season" in feats.columns and "season" in w.columns:

        feats["season"] = pd.to_numeric(feats["season"], errors="coerce").astype("Int64")

        w["season"] = pd.to_numeric(w["season"], errors="coerce").astype("Int64")

    if "week" in feats.columns and "week" in w.columns:

        feats["week"] = pd.to_numeric(feats["week"], errors="coerce").astype("Int64")

        w["week"] = pd.to_numeric(w["week"], errors="coerce").astype("Int64")

    for tcol in ("home_team", "away_team"):

        if tcol in feats.columns:

            feats[tcol] = feats[tcol].astype(str).str.upper().str.strip()

        if tcol in w.columns:

            w[tcol] = w[tcol].astype(str).str.upper().str.strip()



    wx_cols = [c for c in w.columns if c.startswith("weather_") or c in ("kickoff", "roof", "venue_lat", "venue_lon")]

    missing_cols = [c for c in wx_cols if c not in feats.columns]

    if missing_cols:

        filler = pd.DataFrame(pd.NA, index=feats.index, columns=missing_cols)

        feats = pd.concat([feats, filler], axis=1)



    def _wx_mask(df: pd.DataFrame) -> pd.Series:

        cols = [x for x in ("weather_temp_kickoff_f", "kickoff") if x in df.columns]

        if not cols:

            return pd.Series([False] * len(df), index=df.index)

        m = pd.Series([False] * len(df), index=df.index)

        for x in cols:

            m = m | df[x].notna()

        return m



    hit_before = _wx_mask(feats)

    if "game_id" in key_cols:

        feats = feats.merge(

            w.drop_duplicates(subset=["game_id"])[["game_id"] + wx_cols],

            on="game_id",

            how="left",

            suffixes=("", "_wx"),

        )

        for c in wx_cols:

            if (c + "_wx") in feats.columns:

                feats[c] = feats[c].where(feats[c].notna(), feats[c + "_wx"])

        hit_after_gid = _wx_mask(feats)

        print(f"[weather] merge game_id filled: {(hit_after_gid & ~hit_before).sum()} rows")



    if "game_uid" in feats.columns and "game_uid" in w.columns:

        feats = feats.merge(

            w.drop_duplicates(subset=["game_uid"])[["game_uid"] + wx_cols],

            on="game_uid",

            how="left",

            suffixes=("", "_wx3"),

        )

        for c in wx_cols:

            if c in feats.columns and (c + "_wx3") in feats.columns:

                feats[c] = feats[c].where(feats[c].notna(), feats[c + "_wx3"])

        hit_after_guid = _wx_mask(feats)

        print(f"[weather] merge game_uid filled: {(hit_after_guid & ~hit_before).sum()} rows (cumulative)")



    if "game_uid_relaxed" in feats.columns and "game_uid_relaxed" in w.columns:

        feats = feats.merge(

            w.drop_duplicates(subset=["game_uid_relaxed"])[["game_uid_relaxed"] + wx_cols],

            on="game_uid_relaxed",

            how="left",

            suffixes=("", "_wx4"),

        )

        for c in wx_cols:

            if c in feats.columns and (c + "_wx4") in feats.columns:

                feats[c] = feats[c].where(feats[c].notna(), feats[c + "_wx4"])

        hit_after_rel = _wx_mask(feats)

        print(f"[weather] merge game_uid_relaxed filled: {(hit_after_rel & ~hit_before).sum()} rows (cumulative)")



    can_alt = all(c in feats.columns and c in w.columns for c in ["season", "week", "home_team", "away_team"])

    if can_alt:

        alt = feats.merge(

            w.drop_duplicates(subset=["season", "week", "home_team", "away_team"])[

                ["season", "week", "home_team", "away_team"] + wx_cols

            ],

            on=["season", "week", "home_team", "away_team"],

            how="left",

            suffixes=("", "_wx2"),

        )

        for c in wx_cols:

            if c in feats.columns and (c + "_wx2") in alt.columns:

                feats[c] = feats[c].where(feats[c].notna(), alt[c + "_wx2"])

        drop_wx2 = [c for c in alt.columns if c.endswith("_wx2")]

        if drop_wx2:

            alt = alt.drop(columns=drop_wx2)

        feats = alt

        hit_after_swha = _wx_mask(feats)

        print(f"[weather] merge season/week/home/away filled: {(hit_after_swha & ~hit_before).sum()} rows (cumulative)")



    drop_tmp = [c for c in feats.columns if c.endswith("_wx") or c.endswith("_wx2") or c.endswith("_wx3") or c.endswith("_wx4")]

    if drop_tmp:

        feats.drop(columns=drop_tmp, inplace=True, errors="ignore")



    final_hit = _wx_mask(feats)

    total_rows = len(feats)

    print(f"[weather] final weather rows with data: {final_hit.sum()}/{total_rows}")

    return feats



# =========================== Schedule normalization ===========================



def _normalize_schedule_columns(sched: pd.DataFrame) -> pd.DataFrame:

    s = sched.copy()



    # Team columns — many variants possible from ESPN/sdv

    cand_home = [

        "homeTeam", "home_team", "home", "home_abbr", "home_abbreviation",

        "homeTeam.abbreviation", "home_team_abbr", "competitions.home.team.abbreviation",

        "competitions.home.team", "competitions.home"

    ]

    cand_away = [

        "awayTeam", "away_team", "away", "away_abbr", "away_abbreviation",

        "awayTeam.abbreviation", "away_team_abbr", "competitions.away.team.abbreviation",

        "competitions.away.team", "competitions.away"

    ]

    s["home_team"] = _coalesce(cand_home, s)

    s["away_team"] = _coalesce(cand_away, s)



    # Extract abbreviations from JSON-like or nested

    for side in ["home_team", "away_team"]:

        if side in s.columns and s[side].dtype == "object":

            sample = s[side].dropna().head(20).astype(str)

            if sample.str.startswith(("{", "[")).any():

                s[side] = _extract_abbr_from_jsonlike(s[side])



    # Keep display names if present (for resolving 'LA')

    s["home_name"] = _coalesce(["homeTeam.displayName", "homeTeam.name", "home_name", "competitions.home.team.displayName"], s)

    s["away_name"] = _coalesce(["awayTeam.displayName", "awayTeam.name", "away_name", "competitions.away.team.displayName"], s)



    # Normalize abbreviations (handle alternates; resolve 'LA' via name when possible)

    s["home_team"] = s.apply(lambda r: _norm_abbr(r.get("home_team"), r.get("home_name")), axis=1)

    s["away_team"] = s.apply(lambda r: _norm_abbr(r.get("away_team"), r.get("away_name")), axis=1)



    # IDs / week / season

    s["game_id"] = _coalesce(["id", "game_id", "event_id", "eventId", "gameId", "competitions.id"], s)

    s["week"] = _coalesce(["week", "week_number", "round.number"], s)

    s["season"] = _coalesce(["season", "year"], s)



    # Scores

    s["home_score"] = _coalesce(["home_score", "homeScore", "competitions.home.score", "homeTeam.score", "score_home"], s)

    s["away_score"] = _coalesce(["away_score", "awayScore", "competitions.away.score", "awayTeam.score", "score_away"], s)



    # Clean casing/whitespace on team codes

    s["home_team"] = s["home_team"].map(
        lambda v: v.upper().strip() if isinstance(v, str) else np.nan
    )

    s["away_team"] = s["away_team"].map(
        lambda v: v.upper().strip() if isinstance(v, str) else np.nan
    )



    if "roof_is_dome" not in s.columns and "roof" in s.columns:
        roof = s["roof"].astype("string").str.lower().str.strip()
        s["roof_is_dome"] = roof.isin({"closed", "dome", "fixed", "indoor", "indoors"}).astype(int)

    return s



def _dedupe_schedule_games(sched: pd.DataFrame, label: str = "schedule") -> pd.DataFrame:
    """Keep one row per scheduled game after source feeds have been merged."""
    if sched is None or sched.empty:
        return sched

    if "game_id" in sched.columns:
        key = ["game_id"]
    elif {"season", "week", "home_team", "away_team"}.issubset(sched.columns):
        key = ["season", "week", "home_team", "away_team"]
    else:
        return sched

    dup_mask = sched.duplicated(subset=key, keep=False)
    if dup_mask.any():
        examples = sched.loc[dup_mask, key].drop_duplicates().head(5).to_dict("records")
        print(f"[features] warning: dropping duplicate {label} rows on {key}: {examples}")
        return sched.drop_duplicates(subset=key, keep="last").reset_index(drop=True)

    return sched


# =========================== PBP helpers ===========================



def _derive_scores_from_pbp(pbp: pd.DataFrame) -> Optional[pd.DataFrame]:

    if pbp is None or pbp.empty:

        return None



    df = pbp.copy()

    h = "home_score" if "home_score" in df.columns else None

    a = "away_score" if "away_score" in df.columns else None



    if h is None or a is None:

        score_like = [c for c in df.columns if c.lower().endswith("_score")]

        if len(score_like) >= 2:

            for c in score_like:

                lc = c.lower()

                if "home" in lc and h is None:

                    h = c

                if "away" in lc and a is None:

                    a = c

    if h is None or a is None:

        return None



    cand_gid = "game_id" if "game_id" in df.columns else ("id" if "id" in df.columns else None)

    if cand_gid is None:

        return None



    out = (

        df.groupby(cand_gid, as_index=False)[[h, a]]

        .max()

        .rename(columns={h: "home_score", a: "away_score", cand_gid: "game_id"})

    )

    return out





def _standardize_team_cols_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:

    """

    Produce: 'team' (offense), 'opp' (defense), 'yards', 'is_pass', 'is_rush', 'season', 'play_id'.

    Works across multiple possible column name conventions.

    """

    df = pbp.copy()



    cand_team = ["posteam", "possessionTeam.abbreviation", "offenseTeam.abbreviation", "team"]

    cand_opp  = ["defteam", "defenseTeam.abbreviation", "opponentTeam.abbreviation", "opp"]

    cand_yards = ["yards_gained", "yards", "stat.yards", "play.yards", "distance.gained", "statYards", "gain"]

    cand_pass_bool = ["pass", "is_pass", "play.pass"]

    cand_rush_bool = ["rush", "is_rush", "play.rush"]

    cand_epa = ["epa", "stat.epa", "play.epa", "EPA"]

    cand_playtype_txt = ["play_type", "playType", "type.text", "play.type", "playType.text"]



    def _abbr_series(x: Optional[pd.Series]) -> Optional[pd.Series]:

        if x is None:

            return None

        if x.dtype == "object":

            sample = x.dropna().head(20).astype(str)

            if sample.str.startswith(("{", "[")).any():

                x = _extract_abbr_from_jsonlike(x)

        # normalize alternates in pbp as well

        return x.apply(lambda v: _norm_abbr(v))



    t = _abbr_series(_coalesce(cand_team, df))

    o = _abbr_series(_coalesce(cand_opp, df))

    if t is None or o is None:

        return pd.DataFrame(columns=["season", "abbr", "plays", "yards", "pass_plays", "rush_plays", "yards_per_play", "pass_rate"])



    df["team"] = t

    df["opp"] = o



    y = _coalesce(cand_yards, df, default=0)

    if y is not None:

        df["yards"] = pd.to_numeric(y, errors="coerce").fillna(0.0)

    else:

        df["yards"] = 0.0



    p = _coalesce(cand_pass_bool, df)

    r = _coalesce(cand_rush_bool, df)

    if p is not None:

        df["is_pass"] = (p.astype(str).str.lower().isin(["1", "true", "t"])).astype(int)

    if r is not None:

        df["is_rush"] = (r.astype(str).str.lower().isin(["1", "true", "t"])).astype(int)



    if "is_pass" not in df.columns or "is_rush" not in df.columns:

        pt = _coalesce(cand_playtype_txt, df)

        if pt is not None:

            pt_l = pt.astype(str).str.lower()

            if "is_pass" not in df.columns:

                df["is_pass"] = pt_l.str.contains("pass").astype(int)

            if "is_rush" not in df.columns:

                df["is_rush"] = pt_l.str.contains("rush|run|rushing").astype(int)

        else:

            df["is_pass"] = df.get("is_pass", 0)

            df["is_rush"] = df.get("is_rush", 0)



    # EPA (optional)

    e = _coalesce(cand_epa, df)

    if e is not None:

        df["epa"] = pd.to_numeric(e, errors="coerce").fillna(0.0)

    else:

        df["epa"] = 0.0



    if "season" not in df.columns:

        df["season"] = np.nan

    if "play_id" not in df.columns:

        df["play_id"] = np.arange(len(df))



    keep = ["season", "team", "opp", "yards", "is_pass", "is_rush", "epa", "play_id"]

    return df[keep]





def team_season_agg_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate offensive play-by-play into team-season summary stats.

    Parameters
    ----------
    pbp : pd.DataFrame
        Raw play-level data including `team`, `yards`, `is_pass`, `is_rush`, `epa`.

    Returns
    -------
    pd.DataFrame
        One row per (season, abbr) with tempo, yards, pass/rush mix, and EPA metrics.
    """
    if pbp is None or pbp.empty:

        return pd.DataFrame(columns=[

            "season", "abbr", "plays", "yards", "pass_plays", "rush_plays", "yards_per_play", "pass_rate",

            "pass_yards", "rush_yards", "pass_ypp", "rush_ypc", "epa", "epa_per_play"

        ])



    df = _standardize_team_cols_from_pbp(pbp)

    if df.empty or "team" not in df.columns:

        return pd.DataFrame(columns=[

            "season", "abbr", "plays", "yards", "pass_plays", "rush_plays", "yards_per_play", "pass_rate",

            "pass_yards", "rush_yards", "pass_ypp", "rush_ypc", "epa", "epa_per_play"

        ])



    df = df.dropna(subset=["team"])

    # Compute component yards for pass/rush

    df["pass_yards"] = df["yards"] * (df["is_pass"].astype(int))

    df["rush_yards"] = df["yards"] * (df["is_rush"].astype(int))



    g = (

        df.groupby(["season", "team"], as_index=False)

        .agg(

            plays=("play_id", "count"),

            yards=("yards", "sum"),

            pass_plays=("is_pass", "sum"),

            rush_plays=("is_rush", "sum"),

            pass_yards=("pass_yards", "sum"),

            rush_yards=("rush_yards", "sum"),

            epa=("epa", "sum"),

        )

    )

    g = g.dropna(subset=["season"]).copy()

    g["yards_per_play"] = g["yards"] / g["plays"].replace(0, np.nan)

    g["pass_rate"] = g["pass_plays"] / g["plays"].replace(0, np.nan)

    g["pass_ypp"] = g["pass_yards"] / g["pass_plays"].replace(0, np.nan)

    g["rush_ypc"] = g["rush_yards"] / g["rush_plays"].replace(0, np.nan)

    g["epa_per_play"] = g["epa"] / g["plays"].replace(0, np.nan)

    g = g.rename(columns={"team": "abbr"})

    g["abbr"] = g["abbr"].apply(_norm_abbr)

    return g



def team_season_def_allowed_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate opponent (defensive allowed) metrics using play-by-play data.

    Groups by the offensive opponent to derive allowed plays, yards, and pass/rush splits.
    """
    if pbp is None or pbp.empty:

        return pd.DataFrame(columns=[

            "season", "abbr", "plays_allowed", "yards_allowed", "pass_plays_allowed", "rush_plays_allowed",

            "ypp_allowed", "pass_rate_allowed"

        ])



    df = _standardize_team_cols_from_pbp(pbp)

    if df.empty or "opp" not in df.columns:

        return pd.DataFrame(columns=[

            "season", "abbr", "plays_allowed", "yards_allowed", "pass_plays_allowed", "rush_plays_allowed",

            "ypp_allowed", "pass_rate_allowed"

        ])



    df = df.dropna(subset=["opp"]).copy()

    # Pre-compute pass/rush yards per play flags

    df["pass_yards"] = df["yards"] * (df["is_pass"].astype(int))

    df["rush_yards"] = df["yards"] * (df["is_rush"].astype(int))



    g = (

        df.groupby(["season", "opp"], as_index=False)

        .agg(

            plays_allowed=("play_id", "count"),

            yards_allowed=("yards", "sum"),

            pass_plays_allowed=("is_pass", "sum"),

            rush_plays_allowed=("is_rush", "sum"),

            pass_yards_allowed=("pass_yards", "sum"),

            rush_yards_allowed=("rush_yards", "sum"),

            epa_allowed=("epa", "sum"),

        )

    )

    g = g.dropna(subset=["season"]).copy()

    g["ypp_allowed"] = g["yards_allowed"] / g["plays_allowed"].replace(0, np.nan)

    g["pass_rate_allowed"] = g["pass_plays_allowed"] / g["plays_allowed"].replace(0, np.nan)

    g["pass_ypp_allowed"] = g["pass_yards_allowed"] / g["pass_plays_allowed"].replace(0, np.nan)

    g["rush_ypc_allowed"] = g["rush_yards_allowed"] / g["rush_plays_allowed"].replace(0, np.nan)

    g["epa_per_play_allowed"] = g["epa_allowed"] / g["plays_allowed"].replace(0, np.nan)

    g = g.rename(columns={"opp": "abbr"})

    g["abbr"] = g["abbr"].apply(_norm_abbr)

    return g



# =========================== NFL.com team stats loader (robust numeric parsing) ===========================



NFLCOM_FILES = {

    "passing": RAW_DIR / "nflcom_teamstats_passing.parquet",

    "rushing": RAW_DIR / "nflcom_teamstats_rushing.parquet",

    "receiving": RAW_DIR / "nflcom_teamstats_receiving.parquet",

    "scoring": RAW_DIR / "nflcom_teamstats_scoring.parquet",

    "downs": RAW_DIR / "nflcom_teamstats_downs.parquet",

}



_TIME_RE = re.compile(r"^\s*(\d{1,3}):([0-5]?\d)\s*$")  # e.g., 31:45



def _norm_colname(name: str) -> str:

    c = str(name).strip()

    c = c.replace("%", "pct")

    c = c.replace("/", "_per_")

    c = c.replace("-", "_")

    c = c.replace(" ", "")  # Remove all spaces

    return c.lower()



def _clean_numeric_value(x: Any) -> Any:

    if x is None:

        return np.nan

    if isinstance(x, (int, float, np.number)):

        return float(x)

    s = str(x).strip()

    if s == "" or s == "--" or s.lower() == "na":

        return np.nan

    # time mm:ss -> seconds (float)

    m = _TIME_RE.match(s)

    if m:

        mm, ss = int(m.group(1)), int(m.group(2))

        return float(mm * 60 + ss)

    # normalize symbols: unicode minus, commas, leading plus

    s = s.replace("−", "-").replace(",", "")

    if s.startswith("+"):

        s = s[1:]

    # percentages like "57.8%"

    if s.endswith("%"):

        try:

            return float(s[:-1])  # keep as 57.8; divide by 100 later if you prefer

        except Exception:

            return np.nan

    # ranks like "#3" or "T-7"

    if s.startswith("#"):

        s = s[1:]

    if s.upper().startswith("T-"):

        s = s[2:]

    try:

        return float(s)

    except Exception:

        return np.nan



def _load_nflcom_table(path: Path, category: str) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    df = read_df(path)



    # Standardize column names early

    df.columns = [_norm_colname(c) for c in df.columns]



    # Map 'team' or 't e a m' to 'team', 'year' or 'y e a r' to 'season'

    if 'team' not in df.columns and 't e a m' in df.columns:

        df = df.rename(columns={'t e a m': 'team'})

    if 'season' not in df.columns:

        if 'year' in df.columns:

            df = df.rename(columns={'year': 'season'})

        elif 'y e a r' in df.columns:

            df = df.rename(columns={'y e a r': 'season'})



    team_col = 'team' if 'team' in df.columns else None

    if team_col is None:

        return pd.DataFrame()





    df = df.rename(columns={team_col: 'team_name'})

    # Clean up team_name: take first word if repeated, strip whitespace

    def clean_team_name(val):

        if isinstance(val, str):

            # If value is like 'Commanders  Commanders', split and take first

            parts = val.split()

            if len(parts) > 1 and parts[0] == parts[1]:

                return parts[0]

            return val.strip()

        return val

    df['team_name'] = df['team_name'].apply(clean_team_name)

    # Try direct match, then fallback to partial match

    def map_abbr(name):

        if name in TEAM_NAME_TO_ABBR:

            return TEAM_NAME_TO_ABBR[name]

        # Try to find a key that contains the name

        for k in TEAM_NAME_TO_ABBR:

            if name in k:

                return TEAM_NAME_TO_ABBR[k]

        return None

    df['abbr'] = df['team_name'].apply(map_abbr)

    df['abbr'] = df['abbr'].apply(_norm_abbr)



    # Drop obvious non-metrics and coerce numerics

    drop_like = {'category', 'page'}

    base_keys = {'season', 'abbr', 'team_name'}

    candidate_cols = [c for c in df.columns if c not in base_keys | drop_like]



    for c in candidate_cols:

        df[c] = df[c].apply(_clean_numeric_value)



    # Keep keys + numeric columns that have data

    numeric_cols = [c for c in candidate_cols if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0]

    if not numeric_cols:

        return pd.DataFrame(columns=['season', 'abbr'])



    df = df[['season', 'abbr'] + numeric_cols].dropna(subset=['season', 'abbr'])



    # Prefix metrics by category to avoid collisions

    prefix = f'{category}_'

    df = df.rename(columns={c: prefix + c for c in numeric_cols})



    return df



def load_nflcom_team_features() -> pd.DataFrame:
    """Aggregate NFL.com team tables (passing/rushing/etc.) into a single season-level frame."""
    frames = []

    for cat, path in NFLCOM_FILES.items():

        t = _load_nflcom_table(path, cat)

        if not t.empty:

            value_cols = [c for c in t.columns if c not in ("season", "abbr")]

            if value_cols:

                t = t.groupby(["season", "abbr"], as_index=False)[value_cols].sum()

            frames.append(t)

    if not frames:

        return pd.DataFrame(columns=["season", "abbr"])

    out = frames[0]

    for t in frames[1:]:

        out = out.merge(t, on=["season", "abbr"], how="outer")

    return out



# =========================== ESPN player stats (aggregate to team-season) ===========================

def _load_all_espn_player_tables() -> list[pd.DataFrame]:

    frames = []

    try:

        for p in RAW_DIR.glob("espn_playerstats_*.parquet"):

            try:

                df = read_df(p)

                if df is not None and not df.empty:

                    frames.append(df)

            except Exception:

                continue

    except Exception:

        pass

    return frames



def _norm_team_series_for_espn(s: pd.Series) -> pd.Series:

    return s.astype(str).str.strip().str.upper().apply(_norm_abbr)



def load_espn_player_team_agg() -> pd.DataFrame:

    """Aggregate ESPN player stats to team-season totals.

    Robust to varying column names; sums numeric columns per (season, team).

    """

    frames = _load_all_espn_player_tables()

    if not frames:

        return pd.DataFrame(columns=["season", "abbr"])

    out_frames: list[pd.DataFrame] = []

    for df in frames:

        if df is None or df.empty:

            continue

        dfc = df.copy()

        # Team column candidates commonly found on ESPN tables

        team_col = None

        for c in [

            "team", "tm", "team_abbr", "team_abbreviation", "player_team", "team_1", "team_name"

        ]:

            if c in dfc.columns:

                team_col = c

                break

        # If no plausible team column exists, skip this ESPN frame (player leaderboards are league-wide and not team-tagged)

        if team_col is None:

            continue

        if "season" not in dfc.columns:

            # Expect season was added in fetcher; skip otherwise

            continue

        dfc = dfc.rename(columns={team_col: "abbr"})

        dfc["abbr"] = _norm_team_series_for_espn(dfc["abbr"])

        # Keep only (season, abbr) + numeric columns, excluding non-informative metadata

        def _is_meta(col: str) -> bool:

            lc = str(col).lower()

            return lc.endswith("_page") or lc.endswith("_season_type") or lc == "page" or lc == "season_type"

        num_cols = [

            c for c in dfc.columns

            if c not in {"season", "abbr"}

            and not _is_meta(c)

            and pd.api.types.is_numeric_dtype(dfc[c])

        ]

        # Try coercion for likely numeric

        for c in list(dfc.columns):

            if c not in {"season", "abbr"} and c not in num_cols and not _is_meta(c):

                coerced = pd.to_numeric(dfc[c], errors="coerce")

                if coerced.notna().sum() > 0:

                    dfc[c] = coerced

                    num_cols.append(c)

        if not num_cols:

            continue

        agg = (

            dfc.groupby(["season", "abbr"], as_index=False)[num_cols].sum(min_count=1)

        )

        # Prefix by category if present

        cat = None

        if "category" in dfc.columns:

            cat = str(dfc["category"].iloc[0]).strip().lower()

        if cat:

            ren = {c: f"espn_{cat}_{c}" for c in num_cols}

            agg = agg.rename(columns=ren)

        else:

            ren = {c: f"espn_{c}" for c in num_cols}

            agg = agg.rename(columns=ren)

        out_frames.append(agg)



    if not out_frames:

        return pd.DataFrame(columns=["season", "abbr"])

    res = out_frames[0]

    for t in out_frames[1:]:

        res = res.merge(t, on=["season", "abbr"], how="outer")

    # Collapse duplicate columns introduced by merges: *_x / *_y -> base with first non-null

    cols = list(res.columns)

    for c in cols:

        if c.endswith("_x"):

            base = c[:-2]

            ycol = base + "_y"

            if ycol in res.columns:

                res[base] = res[c].combine_first(res[ycol])

                res.drop(columns=[c, ycol], inplace=True)

    return res





PBP_SR_COLUMNS = [

    "season",

    "game_id",

    "posteam",

    "defteam",

    "pass_attempt",

    "complete_pass",

    "passing_yards",

    "yards_gained",

    "pass_touchdown",

    "interception",

    "sack",

    "air_yards",

    "yards_after_catch",

    "rush_attempt",

    "rushing_yards",

    "rush_touchdown",

    "tackled_for_loss",

    "qb_hit",

    "success",

    "first_down",

    "first_down_pass",

    "first_down_rush",

    "epa",

    "fumble_forced",

    "fumble_recovery_1_team",

    "fumble_recovery_2_team",

    "return_touchdown",

]





def _safe_numeric_series(df: pd.DataFrame, col: str, *, default: float = 0.0) -> pd.Series:

    if col not in df.columns:

        return pd.Series(default, index=df.index, dtype="float64")

    series = pd.to_numeric(df[col], errors="coerce")

    return series.fillna(default)





def _add_per_game_means(df: pd.DataFrame, sum_cols: list[str], denom_col: str) -> None:

    if df.empty or denom_col not in df.columns:

        return

    denom = pd.to_numeric(df[denom_col], errors="coerce").replace(0, np.nan)

    for col in sum_cols:

        if col not in df.columns:

            continue

        base = col[:-4] if col.endswith("_sum") else f"{col}_per_game"

        df[f"{base}_mean"] = df[col] / denom





def _load_sr_pbp_subset() -> pd.DataFrame:
    """Load Sportradar play-by-play data with column subset."""
    pbp_path = RAW_DIR / "nfl_pbp.parquet"
    return read_df(pbp_path, columns=PBP_SR_COLUMNS)





def _aggregate_sr_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:

    if pbp is None or pbp.empty:

        return pd.DataFrame(columns=["season", "abbr"])



    df = pbp.copy()

    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    df = df.dropna(subset=["season"])

    df["season"] = df["season"].astype(int)

    df["game_id"] = df["game_id"].astype(str)



    offense = df[df["posteam"].notna()].copy()

    offense_stats = pd.DataFrame()

    if not offense.empty:

        offense["abbr"] = offense["posteam"].astype(str).str.upper().str.strip().apply(_norm_abbr)

        offense = offense.dropna(subset=["abbr"])

        pass_attempts = _safe_numeric_series(offense, "pass_attempt")

        rush_attempts = _safe_numeric_series(offense, "rush_attempt")

        yards_gained = _safe_numeric_series(offense, "yards_gained")

        passing_yards = pd.to_numeric(offense.get("passing_yards"), errors="coerce")

        rushing_yards = pd.to_numeric(offense.get("rushing_yards"), errors="coerce")

        passing_yards = passing_yards.fillna(yards_gained)

        rushing_yards = rushing_yards.fillna(yards_gained)

        mask_pass = pass_attempts > 0

        mask_rush = rush_attempts > 0

        passing_yards = passing_yards.where(mask_pass, 0.0)

        rushing_yards = rushing_yards.where(mask_rush, 0.0)



        sacks = _safe_numeric_series(offense, "sack")

        sack_yards = (-yards_gained).where(sacks > 0, 0.0).clip(lower=0)

        tfl = _safe_numeric_series(offense, "tackled_for_loss")

        rush_tfl = tfl.where(mask_rush, 0.0)

        rush_tfl_yards = (-yards_gained).where(rush_tfl > 0, 0.0).clip(lower=0)



        offense = offense.assign(

            pass_attempts=pass_attempts,

            completions=_safe_numeric_series(offense, "complete_pass"),

            pass_yards=passing_yards,

            pass_touchdowns=_safe_numeric_series(offense, "pass_touchdown"),

            interceptions=_safe_numeric_series(offense, "interception"),

            sacks_taken=sacks,

            sack_yards=sack_yards,

            air_yards_clean=_safe_numeric_series(offense, "air_yards").where(mask_pass, 0.0),

            yac_clean=_safe_numeric_series(offense, "yards_after_catch").where(mask_pass, 0.0),

            rush_attempts=rush_attempts,

            rush_yards=rushing_yards,

            rush_touchdowns=_safe_numeric_series(offense, "rush_touchdown"),

            rush_tfl=rush_tfl,

            rush_tfl_yards=rush_tfl_yards,

            first_down_pass=_safe_numeric_series(offense, "first_down_pass"),

            first_down_rush=_safe_numeric_series(offense, "first_down_rush"),

            success_metric=_safe_numeric_series(offense, "success").where(

                (pass_attempts + rush_attempts) > 0, 0.0

            ),

            epa_metric=_safe_numeric_series(offense, "epa"),

        )



        off_cols = {

            "pass_attempts": "sr_passing_attempts_sum",

            "completions": "sr_passing_completions_sum",

            "pass_yards": "sr_passing_yards_sum",

            "pass_touchdowns": "sr_passing_touchdowns_sum",

            "interceptions": "sr_passing_interceptions_sum",

            "sacks_taken": "sr_passing_sacks_sum",

            "sack_yards": "sr_passing_sack_yards_sum",

            "air_yards_clean": "sr_passing_air_yards_sum",

            "yac_clean": "sr_passing_yards_after_catch_sum",

            "rush_attempts": "sr_rushing_attempts_sum",

            "rush_yards": "sr_rushing_yards_sum",

            "rush_touchdowns": "sr_rushing_touchdowns_sum",

            "rush_tfl": "sr_rushing_tlost_sum",

            "rush_tfl_yards": "sr_rushing_tlost_yards_sum",

            "first_down_pass": "sr_passing_first_downs_sum",

            "first_down_rush": "sr_rushing_first_downs_sum",

            "success_metric": "sr_offense_successful_plays_sum",

            "epa_metric": "sr_offense_epa_sum",

        }



        offense_stats = (

            offense.groupby(["season", "abbr"], as_index=False)

            .agg({col: "sum" for col in off_cols})

            .rename(columns={col: dest for col, dest in off_cols.items()})

        )



    defense = df[df["defteam"].notna()].copy()

    defense_stats = pd.DataFrame()

    if not defense.empty:

        defense["abbr"] = defense["defteam"].astype(str).str.upper().str.strip().apply(_norm_abbr)

        defense = defense.dropna(subset=["abbr"])

        yards_allowed = _safe_numeric_series(defense, "yards_gained")

        frec = pd.Series(0.0, index=defense.index, dtype="float64")

        for col in ("fumble_recovery_1_team", "fumble_recovery_2_team"):

            if col in defense.columns:

                matches = (

                    defense[col]

                    .astype(str)

                    .str.upper()

                    .str.strip()

                ) == defense["abbr"]

                frec = frec + matches.fillna(False).astype(float)

        defense = defense.assign(

            sacks=_safe_numeric_series(defense, "sack"),

            qb_hits=_safe_numeric_series(defense, "qb_hit"),

            interceptions=_safe_numeric_series(defense, "interception"),

            forced_fumbles=_safe_numeric_series(defense, "fumble_forced"),

            fumble_recoveries=frec,

            tfl=_safe_numeric_series(defense, "tackled_for_loss"),

            tfl_yards=(-yards_allowed).where(_safe_numeric_series(defense, "tackled_for_loss") > 0, 0.0).clip(lower=0),

            return_tds=_safe_numeric_series(defense, "return_touchdown"),

        )



        def_cols = {

            "sacks": "sr_defense_sacks_sum",

            "qb_hits": "sr_defense_qb_hits_sum",

            "interceptions": "sr_defense_interceptions_sum",

            "forced_fumbles": "sr_defense_forced_fumbles_sum",

            "fumble_recoveries": "sr_defense_fumble_recoveries_sum",

            "tfl": "sr_defense_tloss_sum",

            "tfl_yards": "sr_defense_tloss_yards_sum",

            "return_tds": "sr_defense_touchdowns_sum",

        }



        defense_stats = (

            defense.groupby(["season", "abbr"], as_index=False)

            .agg({col: "sum" for col in def_cols})

            .rename(columns={col: dest for col, dest in def_cols.items()})

        )



    game_frames: list[pd.DataFrame] = []

    if not offense.empty:

        off_games = (

            offense[["season", "abbr", "game_id"]]

            .dropna()

            .drop_duplicates()

        )

        game_frames.append(off_games)

    if not defense.empty:

        def_games = (

            defense[["season", "abbr", "game_id"]]

            .dropna()

            .drop_duplicates()

        )

        game_frames.append(def_games)



    if not game_frames:

        return pd.DataFrame(columns=["season", "abbr"])



    games = (

        pd.concat(game_frames, ignore_index=True)

        .groupby(["season", "abbr"], as_index=False)["game_id"]

        .count()

        .rename(columns={"game_id": "sr_games_sampled"})

    )

    games["sr_games_sampled"] = games["sr_games_sampled"].astype(int)



    result = games.copy()

    if not offense_stats.empty:

        offense_stats = offense_stats.merge(games, on=["season", "abbr"], how="left")

        off_sum_cols = [col for col in offense_stats.columns if col.startswith("sr_") and col.endswith("_sum")]

        _add_per_game_means(offense_stats, off_sum_cols, "sr_games_sampled")

        if {"sr_passing_completions_sum", "sr_passing_attempts_sum"}.issubset(offense_stats.columns):

            offense_stats["sr_passing_cmp_pct_mean"] = (

                offense_stats["sr_passing_completions_sum"]

                / offense_stats["sr_passing_attempts_sum"].replace(0, np.nan)

            )

        if {"sr_passing_yards_sum", "sr_passing_attempts_sum"}.issubset(offense_stats.columns):

            offense_stats["sr_passing_avg_yards_mean"] = (

                offense_stats["sr_passing_yards_sum"]

                / offense_stats["sr_passing_attempts_sum"].replace(0, np.nan)

            )

        if {"sr_rushing_yards_sum", "sr_rushing_attempts_sum"}.issubset(offense_stats.columns):

            offense_stats["sr_rushing_avg_yards_mean"] = (

                offense_stats["sr_rushing_yards_sum"]

                / offense_stats["sr_rushing_attempts_sum"].replace(0, np.nan)

            )

        if "sr_passing_attempts_sum" in offense_stats.columns and "sr_rushing_attempts_sum" in offense_stats.columns:

            offense_stats["sr_offense_plays_sum"] = (

                offense_stats["sr_passing_attempts_sum"] + offense_stats["sr_rushing_attempts_sum"]

            )

            offense_stats["sr_offense_yards_sum"] = (

                offense_stats.get("sr_passing_yards_sum", 0) + offense_stats.get("sr_rushing_yards_sum", 0)

            )

            _add_per_game_means(

                offense_stats,

                ["sr_offense_plays_sum", "sr_offense_yards_sum"],

                "sr_games_sampled",

            )

            offense_stats["sr_offense_pass_rate_mean"] = (

                offense_stats["sr_passing_attempts_sum"]

                / offense_stats["sr_offense_plays_sum"].replace(0, np.nan)

            )

        if {"sr_offense_successful_plays_sum", "sr_offense_plays_sum"}.issubset(offense_stats.columns):

            offense_stats["sr_offense_success_rate_mean"] = (

                offense_stats["sr_offense_successful_plays_sum"]

                / offense_stats["sr_offense_plays_sum"].replace(0, np.nan)

            )

        if {"sr_offense_epa_sum", "sr_offense_plays_sum"}.issubset(offense_stats.columns):

            offense_stats["sr_offense_epa_per_play_mean"] = (

                offense_stats["sr_offense_epa_sum"]

                / offense_stats["sr_offense_plays_sum"].replace(0, np.nan)

            )

        result = result.merge(

            offense_stats,

            on=["season", "abbr"],

            how="left",

            suffixes=("", "_off"),

        )



    if not defense_stats.empty:

        defense_stats = defense_stats.merge(games, on=["season", "abbr"], how="left")

        def_sum_cols = [col for col in defense_stats.columns if col.startswith("sr_defense_") and col.endswith("_sum")]

        _add_per_game_means(defense_stats, def_sum_cols, "sr_games_sampled")

        result = result.merge(

            defense_stats,

            on=["season", "abbr"],

            how="left",

            suffixes=("", "_def"),

        )



    if "sr_games_sampled" not in result.columns:

        result["sr_games_sampled"] = pd.NA

    for suffix in ("_x", "_y", "_off", "_def"):

        col = f"sr_games_sampled{suffix}"

        if col in result.columns:

            result["sr_games_sampled"] = result["sr_games_sampled"].combine_first(result[col])

            result.drop(columns=[col], inplace=True)



    return result.sort_values(["season", "abbr"]).reset_index(drop=True)





def load_sportradar_team_features() -> pd.DataFrame:

    """Build the legacy Sportradar feature block from local PBP (preferred) with legacy fallback."""

    pbp = _load_sr_pbp_subset()

    derived = _aggregate_sr_from_pbp(pbp)

    if not derived.empty:

        print(f"[sr_substitute] using play-by-play derived team features ({len(derived)} rows)")

        return derived

    print("[sr_substitute] falling back to legacy Sportradar aggregates (no PBP data available)")

    return _load_sportradar_team_features_legacy()





def _load_sportradar_team_features_legacy() -> pd.DataFrame:

    """Legacy fallback: aggregate historical Sportradar parquet files if present."""

    sr_dir = PROC_DIR / "sportradar"

    team_path = sr_dir / "game_team_stats.parquet"

    sched_path = sr_dir / "schedule.parquet"

    if not team_path.exists() or not sched_path.exists():

        return pd.DataFrame(columns=["season", "abbr"])



    try:

        team_df = read_df(team_path)

        sched_df = read_df(sched_path)

    except Exception:

        return pd.DataFrame(columns=["season", "abbr"])



    if team_df is None or team_df.empty or sched_df is None or sched_df.empty:

        return pd.DataFrame(columns=["season", "abbr"])



    sched_subset = sched_df[

        [

            "game_id",

            "season_year",

            "week",

            "home_id",

            "home_alias",

            "home_name",

            "away_id",

            "away_alias",

            "away_name",

        ]

    ]



    merged = team_df.merge(sched_subset, on="game_id", how="left")



    def _resolve_abbr_row(row: pd.Series) -> Optional[str]:

        if row.get("team_id") == row.get("home_id"):

            return _resolve_sportradar_abbr(row.get("home_alias"), row.get("home_name"))

        if row.get("team_id") == row.get("away_id"):

            return _resolve_sportradar_abbr(row.get("away_alias"), row.get("away_name"))

        return None



    merged["abbr"] = merged.apply(_resolve_abbr_row, axis=1)

    merged["season"] = merged["season_year"]

    merged = merged.dropna(subset=["abbr", "season"])

    if merged.empty:

        return pd.DataFrame(columns=["season", "abbr"])



    merged["abbr"] = merged["abbr"].astype(str).str.upper().str.strip().apply(_norm_abbr)



    def _expand_stats(row: pd.Series) -> Dict[str, Any]:

        stats_obj = _safe_json_load(row.get("stats"))

        if not isinstance(stats_obj, dict):

            return {}

        out: Dict[str, Any] = {}

        for key, value in stats_obj.items():

            if isinstance(value, (int, float)) and pd.notna(value):

                out[f"sr_{row.get('stat_category')}_{key}"] = float(value)

        return out



    expanded_stats = merged.apply(_expand_stats, axis=1)

    expanded_df = pd.DataFrame(expanded_stats.tolist())

    expanded_df["season"] = merged["season"].values

    expanded_df["abbr"] = merged["abbr"].values

    expanded_df["games_played"] = 1



    stats_cols = [c for c in expanded_df.columns if c not in {"season", "abbr", "games_played"}]

    if stats_cols:

        grouped_mean = (

            expanded_df.groupby(["season", "abbr"], as_index=False)[stats_cols]

            .mean()

            .rename(columns={c: f"{c}_mean" for c in stats_cols})

        )

        grouped_sum = (

            expanded_df.groupby(["season", "abbr"], as_index=False)[stats_cols]

            .sum()

            .rename(columns={c: f"{c}_sum" for c in stats_cols})

        )

        games = (

            expanded_df.groupby(["season", "abbr"], as_index=False)["games_played"]

            .sum()

            .rename(columns={"games_played": "sr_games_sampled"})

        )

        out = games.merge(grouped_mean, on=["season", "abbr"], how="left").merge(

            grouped_sum, on=["season", "abbr"], how="left"

        )

    else:

        out = pd.DataFrame(columns=["season", "abbr"])



    seasonal_path = sr_dir / "seasonal_team_stats.parquet"

    if seasonal_path.exists():

        try:

            seasonal_df = read_df(seasonal_path)

        except Exception:

            seasonal_df = pd.DataFrame()

        if seasonal_df is not None and not seasonal_df.empty:

            seasonal_df = seasonal_df.copy()

            seasonal_df["season"] = pd.to_numeric(seasonal_df.get("season"), errors="coerce")

            seasonal_df = seasonal_df.dropna(subset=["season"])

            seasonal_df["season"] = seasonal_df["season"].astype(int)

            seasonal_df["abbr"] = seasonal_df.get("team_alias", "").astype(str).str.upper().str.strip().apply(_norm_abbr)

            seasonal_features = [

                c

                for c in seasonal_df.columns

                if c not in {"team_id", "team_name", "team_alias", "team_market", "source_path", "abbr"}

            ]

            seasonal_df = seasonal_df[["season", "abbr"] + seasonal_features]

            if out.empty:

                out = seasonal_df

            else:

                out = out.merge(seasonal_df, on=["season", "abbr"], how="outer")



    roster_summary_path = sr_dir / "roster_status_summary.parquet"

    if roster_summary_path.exists():

        try:

            roster_df = read_df(roster_summary_path)

        except Exception:

            roster_df = pd.DataFrame()

        if roster_df is not None and not roster_df.empty:

            roster_df = roster_df.copy()

            roster_df["season"] = pd.to_numeric(roster_df.get("season"), errors="coerce")

            roster_df = roster_df.dropna(subset=["season"])

            roster_df["season"] = roster_df["season"].astype(int)

            roster_df["abbr"] = roster_df["team_alias"].astype(str).str.upper().str.strip().apply(_norm_abbr)

            roster_features = [

                c

                for c in roster_df.columns

                if c not in {"team_id", "team_name", "team_alias", "source_path", "abbr", "season"}

            ]

            roster_df = roster_df[["season", "abbr"] + roster_features]

            if out.empty:

                out = roster_df

            else:

                out = out.merge(roster_df, on=["season", "abbr"], how="outer")

    return out





# =========================== Matchup feature builder ===========================



def build_matchup_features(schedule: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:

    """

    Normalize schedule rows and join them with team-season aggregates to form per-game features.



    Parameters

    ----------

    schedule

        Raw schedule DataFrame that may include mixed team identifiers.

    team_stats

        Team-season metrics keyed by (season, abbr) covering offense/defense signals.



    Returns

    -------

    pd.DataFrame

        One row per game with home/away feature suffixes plus derived targets.

    """

    # Ensure team_stats is unique for each (season, abbr)
    if 'season' in team_stats.columns and 'abbr' in team_stats.columns:
        team_stats = team_stats.drop_duplicates(subset=['season', 'abbr'])
    
    s = _normalize_schedule_columns(schedule)
    s = _dedupe_schedule_games(s, "matchup schedule")
    
    # Only keep rows where both teams are real NFL teams
    valid_abbrs = set(team_stats['abbr'].unique()) if 'abbr' in team_stats.columns else set()
    s = s[s['home_team'].isin(valid_abbrs) & s['away_team'].isin(valid_abbrs)].copy()



    try:

        s["game_uid"] = _mk_uid(s)

    except Exception:

        pass

    try:

        s["game_uid_relaxed"] = _mk_uid_relaxed(s)

    except Exception:

        pass



    # Normalize abbr in team_stats

    if "abbr" in team_stats.columns:

        team_stats = team_stats.copy()

        team_stats["abbr"] = team_stats["abbr"].astype(str).str.upper().str.strip().apply(_norm_abbr)



    if team_stats.empty:

        print("[ERROR] team_stats is empty. Skipping merge to avoid cartesian product.")

        return s

    feats = s.merge(
        team_stats.add_suffix("_home"),
        left_on=["season", "home_team"],
        right_on=["season_home", "abbr_home"],
        how="left",
    ).merge(
        team_stats.add_suffix("_away"),
        left_on=["season", "away_team"],
        right_on=["season_away", "abbr_away"],
        how="left",
    )



    # Merge report

    try:

        total = len(feats)

        matched_home = feats["abbr_home"].notna().sum()

        matched_away = feats["abbr_away"].notna().sum()

        print(f"[merge] matched home: {matched_home}/{total}  matched away: {matched_away}/{total}")

        if matched_home < total or matched_away < total:

            unmatched_home = feats.loc[feats["abbr_home"].isna(), "home_team"].dropna().unique()[:10]

            unmatched_away = feats.loc[feats["abbr_away"].isna(), "away_team"].dropna().unique()[:10]

            print(f"[merge] sample unmatched home_team codes: {list(unmatched_home)}")

            print(f"[merge] sample unmatched away_team codes: {list(unmatched_away)}")

    except Exception:

        pass



    # Optional quick diffs (trainer will also auto-diff generically)

    if "yards_per_play_home" in feats.columns and "yards_per_play_away" in feats.columns:

        feats["ypp_diff"] = feats["yards_per_play_home"] - feats["yards_per_play_away"]

    if "pass_rate_home" in feats.columns and "pass_rate_away" in feats.columns:

        feats["passrate_diff"] = feats["pass_rate_home"] - feats["pass_rate_away"]



    # Targets

    need_scores = feats["home_score"].isna() | feats["away_score"].isna()

    if need_scores.any():

        pbp_path = RAW_DIR / "espn_pbp.parquet"

        if pbp_path.exists():

            pbp = read_df(pbp_path)

            scores = _derive_scores_from_pbp(pbp)

            if scores is not None and not scores.empty:

                feats = feats.merge(scores, on="game_id", how="left", suffixes=("", "_frompbp"))

                feats["home_score"] = feats["home_score"].fillna(feats["home_score_frompbp"])

                feats["away_score"] = feats["away_score"].fillna(feats["away_score_frompbp"])

                feats.drop(columns=[c for c in feats.columns if c.endswith("_frompbp")], inplace=True)



    if "home_score" in feats.columns and "away_score" in feats.columns:

        feats["home_score"] = pd.to_numeric(feats["home_score"], errors="coerce")

        feats["away_score"] = pd.to_numeric(feats["away_score"], errors="coerce")

        feats["home_margin"] = feats["home_score"] - feats["away_score"]

        feats["home_win"] = np.select(
            [
                feats["home_margin"] > 0,
                feats["home_margin"] < 0,
            ],
            [1.0, 0.0],
            default=np.nan,
        )

    else:

        feats["home_margin"] = np.nan

        feats["home_win"] = np.nan



    feats = _add_travel_features(feats)



    return feats



# =========================== CLI ===========================



def main():
    """CLI entry point to build matchup features for specified seasons."""
    ap = argparse.ArgumentParser()

    ap.add_argument("--season", type=int, nargs="+", required=True)

    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    args = ap.parse_args()



    setup_logging("nfl_predictions", level="DEBUG" if args.debug else "INFO")

    seasons = [int(s) for s in args.season]



    sched_path = RAW_DIR / "nfl_schedules.parquet"

    if not sched_path.exists():

        sched_path = RAW_DIR / "espn_schedule.parquet"

    pbp_path = RAW_DIR / "nfl_pbp.parquet"

    if not pbp_path.exists():

        pbp_path = RAW_DIR / "espn_pbp.parquet"






    if not sched_path.exists():

        raise FileNotFoundError(

            f"Expected schedule at {sched_path}. Run: python -m src.data.nflsdv --season ... --schedules"

        )



    sched = _load_schedule_data(sched_path, seasons)

    pbp = read_df(pbp_path) if pbp_path.exists() else pd.DataFrame()

    if not pbp.empty and "season" in pbp.columns:

        pbp = pbp[pbp["season"].isin(seasons)].copy()



    # 1) PBP-derived team features (may be empty if schema mismatch)

    pbp_team_off = team_season_agg_from_pbp(pbp)

    pbp_team_def = team_season_def_allowed_from_pbp(pbp)



    # 2) NFL.com team features (robust numeric parsing)

    nflcom_team = load_nflcom_team_features()

    # 3) ESPN player totals aggregated to team-season

    espn_team = load_espn_player_team_agg()

    # 4) Sportradar advanced team metrics

    sportradar_team = load_sportradar_team_features()



    # Merge sources on (season, abbr)

    merged = None

    if not nflcom_team.empty:

        merged = nflcom_team.copy()

    if not pbp_team_off.empty:

        merged = (merged.merge(pbp_team_off, on=["season", "abbr"], how="outer") if merged is not None else pbp_team_off)

    if not pbp_team_def.empty:

        merged = (merged.merge(pbp_team_def, on=["season", "abbr"], how="outer") if merged is not None else pbp_team_def)

    if not espn_team.empty:

        merged = (merged.merge(espn_team, on=["season", "abbr"], how="outer") if merged is not None else espn_team)

    if not sportradar_team.empty:

        merged = (

            merged.merge(sportradar_team, on=["season", "abbr"], how="outer")

            if merged is not None

            else sportradar_team

        )

    # ESPN team defense removed



    team_stats = merged if merged is not None else pd.DataFrame(columns=["season", "abbr"])

    if not team_stats.empty and "season" in team_stats.columns:

        team_stats = team_stats[team_stats["season"].isin(seasons)].copy()



    if not team_stats.empty:

        write_df(team_stats, PROC_DIR / "team_season_agg.parquet")



    feats = build_matchup_features(

        sched,

        team_stats if not team_stats.empty else pd.DataFrame(columns=["season", "abbr"]),

    )

    if not feats.empty and "season" in feats.columns:

        feats = feats[feats["season"].isin(seasons)].copy()



        # ==== Integrate additional nflverse datasets at team-season-week level ====

    def _load_raw(name: str) -> pd.DataFrame:

        try:

            p = RAW_DIR / name

            return read_df(p) if p.exists() else pd.DataFrame()

        except Exception:

            return pd.DataFrame()



    feats = _augment_team_week_features(feats, seasons, _load_raw, sched)



    # ==== Merge game-level weather if available ====

    try:

        weather_df = _load_weather_sources()

        if not weather_df.empty:

            feats = _merge_weather_features(feats, weather_df)

    except Exception as e:

        print(f"[weather] merge skipped: {e}")



    # Static weather fallback: fill NaN weather rows using climatological normals.
    # Applies only when all three live sources (Visual Crossing, NOAA, Tomorrow.io)
    # failed to provide data for a game. Does not overwrite valid live data.

    try:

        wx_cols = ["weather_temp_f", "weather_wind_mph", "weather_precip_prob"]

        # Only fill rows that are missing ALL three primary weather columns
        if "home_team" in feats.columns:

            missing_wx = feats[wx_cols].isna().all(axis=1) if all(c in feats.columns for c in wx_cols) else pd.Series(True, index=feats.index)

            if missing_wx.any():

                game_dates = pd.to_datetime(feats.get("game_date", feats.get("gameday")), errors="coerce") if ("game_date" in feats.columns or "gameday" in feats.columns) else pd.Series([None] * len(feats), index=feats.index)

                for idx in feats.index[missing_wx]:

                    team = str(feats.at[idx, "home_team"]).strip().upper()

                    gdate = game_dates.get(idx)

                    fb = get_fallback_weather(team, game_date=gdate.date() if pd.notna(gdate) else None)

                    for col, val in fb.items():

                        if col not in feats.columns or pd.isna(feats.at[idx, col]):

                            feats.at[idx, col] = val

                n_filled = int(missing_wx.sum())

                print(f"[weather][fallback] filled {n_filled} games with climatological normals")

    except Exception as e:

        print(f"[weather][fallback] skipped: {e}")



    # Weather deltas for modeling (reuse volatility engineering for consistency)

    try:

        vol_inputs = engineer_volatility_inputs(feats)

        feats["wx_temp_delta"] = pd.to_numeric(vol_inputs.get("weather_temp_delta"), errors="coerce")

        feats["wx_wind_delta"] = pd.to_numeric(vol_inputs.get("wind_mph_delta"), errors="coerce")

        if "weather_rain_index_diff" in feats.columns:

            rain_delta = pd.to_numeric(feats["weather_rain_index_diff"], errors="coerce")

        else:

            rain_delta = pd.Series(np.nan, index=feats.index)

        feats["wx_rain_index_delta"] = rain_delta

        indoor_mask = vol_inputs.get("indoor_game")

        if indoor_mask is not None:

            indoor_mask = indoor_mask.astype(bool)

            for col in ["wx_temp_delta", "wx_wind_delta", "wx_rain_index_delta"]:

                if col in feats.columns:

                    feats.loc[indoor_mask, col] = 0.0

        feats["wx_temp_delta"] = pd.to_numeric(feats["wx_temp_delta"], errors="coerce")

        feats["wx_wind_delta"] = pd.to_numeric(feats["wx_wind_delta"], errors="coerce")

        feats["wx_rain_index_delta"] = pd.to_numeric(feats["wx_rain_index_delta"], errors="coerce")

    except Exception as e:

        print(f"[weather] delta computation skipped: {e}")



    # Stadium context (noise, altitude, weather risk)

    try:

        stadium_ctx = load_stadium_context()

        if not stadium_ctx.empty:

            context_cols = [c for c in stadium_ctx.columns if c != "team_abbr"]

            for prefix in ("home", "away"):

                ctx = stadium_ctx.rename(columns={"team_abbr": f"{prefix}_team"}).copy()

                rename_map = {col: f"{prefix}_{col}" for col in context_cols}

                ctx.rename(columns=rename_map, inplace=True)

                feats = feats.merge(ctx, on=f"{prefix}_team", how="left")

            for col in ("altitude_ft", "crowd_noise_score", "fan_hostility_score", "weather_snow_index", "weather_rain_index", "indoor"):

                hcol = f"home_{col}"

                acol = f"away_{col}"

                if hcol in feats.columns and acol in feats.columns:

                    feats[f"{col}_diff"] = pd.to_numeric(feats[hcol], errors="coerce") - pd.to_numeric(feats[acol], errors="coerce")

    except Exception as e:

        print(f"[context] stadium context merge skipped: {e}")



    # Altitude × rest interaction: penalise away teams travelling to high-altitude venues on short rest.
    # altitude_ft_diff = home_altitude_ft - away_altitude_ft (already computed above).
    # A sea-level team flying into Denver (altitude_diff ≈ +5280 ft) on 3 days rest scores highest.
    # Clipped to 0 — visiting a lower-altitude venue carries no equivalent penalty.

    try:

        if "altitude_ft_diff" in feats.columns and "sched_rest_days_away" in feats.columns:

            alt_delta = pd.to_numeric(feats["altitude_ft_diff"], errors="coerce").fillna(0.0)

            rest_away = pd.to_numeric(feats["sched_rest_days_away"], errors="coerce").fillna(7.0)

            feats["altitude_rest_interaction"] = (
                alt_delta / (np.log1p(rest_away) + 1e-6)
            ).clip(lower=0.0)

    except Exception as e:

        print(f"[context] altitude_rest_interaction skipped: {e}")



    # Rivalry indicators

    try:

        rivalry_df = load_rivalry_table()

        if not rivalry_df.empty:

            rivalry_df["pair_key"] = rivalry_df["team_abbr"] + "_" + rivalry_df["rival_abbr"]

            agg = (

                rivalry_df.groupby("pair_key")

                .agg(

                    rivalry_intensity=("intensity", "max"),

                    rivalry_types=("rivalry_type", lambda x: "|".join(sorted(set(map(str, x))))),

                )

                .reset_index()

            )

            agg["rivalry_is_divisional"] = agg["rivalry_types"].str.contains("divisional").astype(int)

            agg["rivalry_has_historic_component"] = agg["rivalry_types"].apply(

                lambda s: int(any(part != "divisional" for part in s.split("|") if part))

            )

            feats["pair_key"] = feats["home_team"].astype(str).str.upper() + "_" + feats["away_team"].astype(str).str.upper()

            feats = feats.merge(agg.drop(columns=["rivalry_types"]), on="pair_key", how="left")

            feats["rivalry_intensity"] = pd.to_numeric(feats["rivalry_intensity"], errors="coerce").fillna(0.0)

            feats["rivalry_is_divisional"] = feats["rivalry_is_divisional"].fillna(0).astype(int)

            feats.drop(columns=["pair_key"], inplace=True, errors="ignore")

            feats["rivalry_has_historic_component"] = feats["rivalry_has_historic_component"].fillna(0).astype(int)

    except Exception as e:

        print(f"[context] rivalry merge skipped: {e}")



    # ==== Build persistent defensive allowed proxies and snaps proxies (when PBP missing) ====

    try:

        f = feats  # alias

        def _num_series(col: str) -> pd.Series:

            if col in f.columns:

                return pd.to_numeric(f[col], errors="coerce")

            # empty numeric series aligned to index

            return pd.Series([pd.NA] * len(f), index=f.index, dtype="float")

        # Offense proxies from NFL.com totals where PBP missing

        def _mk_off_proxy(side: str):

            # Use NFL.com downs/passing/rushing totals

            plays = _num_series(f"downs_scrmplys_{side}")

            pass_att = _num_series(f"passing_att_{side}")

            rush_att = _num_series(f"rushing_att_{side}")

            pass_yds = _num_series(f"passing_passyds_{side}")

            rush_yds = _num_series(f"rushing_rushyds_{side}")



            # Persist proxies on feats for potential downstream use

            # plays proxy prefers actual plays if present, else downs

            if f"plays_{side}" in f.columns:

                f[f"plays_proxy_{side}"] = _num_series(f"plays_{side}").fillna(plays)

            else:

                f[f"plays_proxy_{side}"] = plays

            f[f"pass_plays_proxy_{side}"] = pass_att

            f[f"rush_plays_proxy_{side}"] = rush_att

            f[f"pass_yards_proxy_{side}"] = pass_yds

            f[f"rush_yards_proxy_{side}"] = rush_yds

            f[f"yards_proxy_{side}"] = f[f"pass_yards_proxy_{side}"] + f[f"rush_yards_proxy_{side}"]



        for s in ("home", "away"):

            _mk_off_proxy(s)



        # Helper: prefer pbp allowed if present; else use opponent offense proxies

        def _fill_allowed(base: str, src_pass: bool = False, src_rush: bool = False):

            for side, opp in (("home", "away"), ("away", "home")):

                # Names

                pbp_name = f"{base}_allowed_{side}"

                def_name = f"def_{base}_allowed_{side}"

                if base in ("ypp", "pass_rate", "pass_ypp", "rush_ypc", "epa_per_play"):

                    # Derived per-play metrics will be handled after totals

                    continue

                if pbp_name in f.columns:

                    # Copy PBP allowed first

                    if def_name in f.columns:

                        f[def_name] = f[def_name].fillna(_num_series(pbp_name))

                    else:

                        f[def_name] = _num_series(pbp_name)

                # If still null, use opponent offense proxies

                mask = f[def_name].isna() if def_name in f.columns else pd.Series([True]*len(f), index=f.index)

                if base == "plays":

                    f.loc[mask, def_name] = _num_series(f"plays_proxy_{opp}").loc[mask]

                elif base == "pass_plays":

                    f.loc[mask, def_name] = _num_series(f"pass_plays_proxy_{opp}").loc[mask]

                elif base == "rush_plays":

                    f.loc[mask, def_name] = _num_series(f"rush_plays_proxy_{opp}").loc[mask]

                elif base == "yards":

                    f.loc[mask, def_name] = _num_series(f"yards_proxy_{opp}").loc[mask]

                elif base == "pass_yards":

                    f.loc[mask, def_name] = _num_series(f"pass_yards_proxy_{opp}").loc[mask]

                elif base == "rush_yards":

                    f.loc[mask, def_name] = _num_series(f"rush_yards_proxy_{opp}").loc[mask]

                elif base == "epa":

                    # No proxy for EPA; leave as-is

                    pass



        for base in ["plays", "pass_plays", "rush_plays", "yards", "pass_yards", "rush_yards", "epa"]:

            _fill_allowed(base)



        # Derived allowed per-play metrics from totals if missing

        def _derive_rate(name: str, num: str, den: str):

            for side in ("home", "away"):

                cname = f"def_{name}_allowed_{side}"

                if cname not in f.columns or f[cname].isna().any():

                    n = _num_series(f"def_{num}_allowed_{side}")

                    d = _num_series(f"def_{den}_allowed_{side}")

                    val = n / d.replace(0, pd.NA)

                    if cname in f.columns:

                        f[cname] = f[cname].fillna(val)

                    else:

                        f[cname] = val



        _derive_rate("ypp", "yards", "plays")

        _derive_rate("pass_rate", "pass_plays", "plays")

        _derive_rate("pass_ypp", "pass_yards", "pass_plays")

        _derive_rate("rush_ypc", "rush_yards", "rush_plays")

        # epa_per_play from epa/plays if available

        _derive_rate("epa_per_play", "epa", "plays")



        # Diffs for def_* allowed metrics

        def_bases = [

            "ypp_allowed", "pass_rate_allowed", "pass_ypp_allowed", "rush_ypc_allowed", "epa_per_play_allowed",

            "plays_allowed", "yards_allowed", "pass_plays_allowed", "rush_plays_allowed", "pass_yards_allowed", "rush_yards_allowed", "epa_allowed"

        ]

        for base in def_bases:

            hcol = f"def_{base}_home" if base.endswith("allowed") or base.startswith("ypp") else f"def_{base}_home"

            acol = f"def_{base}_away" if base.endswith("allowed") or base.startswith("ypp") else f"def_{base}_away"

            # The above names already include 'def_' prefix in our construction

            hcol = f"def_{base}_home"

            acol = f"def_{base}_away"

            dcol = f"def_{base}_diff"

            if hcol in f.columns and acol in f.columns:

                f[dcol] = pd.to_numeric(f[hcol], errors="coerce") - pd.to_numeric(f[acol], errors="coerce")



        # Source flags: 1 if any key def totals came from proxy

        for side, opp in (("home", "away"), ("away", "home")):

            flag = pd.Series(0, index=f.index, dtype="int64")

            for tot in ("plays", "yards", "pass_plays", "rush_plays"):

                target = f"def_{tot}_allowed_{side}"

                pbp = f"{tot}_allowed_{side}"

                if target in f.columns:

                    need = f[target].isna() if pbp not in f.columns else (f[target].notna() & f.get(pbp, pd.Series(index=f.index)).isna())

                    flag = flag | need.fillna(0).astype(int)

            f[f"def_allowed_from_proxy_{side}"] = flag



        # Snap proxies if missing

        for side, opp in (("home", "away"), ("away", "home")):

            if f"snaps_offense_snaps_{side}" not in f.columns and f"downs_scrmplys_{side}" in f.columns:

                f[f"snaps_offense_snaps_{side}"] = pd.to_numeric(f[f"downs_scrmplys_{side}"], errors="coerce")

            if f"snaps_defense_snaps_{side}" not in f.columns and f"downs_scrmplys_{opp}" in f.columns:

                f[f"snaps_defense_snaps_{side}"] = pd.to_numeric(f[f"downs_scrmplys_{opp}"], errors="coerce")

        # Snap diffs

        for base in ("snaps_offense_snaps", "snaps_defense_snaps"):

            h, a, d = f"{base}_home", f"{base}_away", f"{base}_diff"

            if h in f.columns and a in f.columns:

                f[d] = pd.to_numeric(f[h], errors="coerce") - pd.to_numeric(f[a], errors="coerce")



        feats = f

    except Exception as e:

        print(f"[features] proxy def/snaps build skipped: {e}")

    print(f"[features] writing matchup_features with shape {feats.shape}")

    write_df(feats, PROC_DIR / "matchup_features.parquet")





STADIUM_CONTEXT_PATH = Path("data/reference/stadium_advantages.csv")

RIVALRIES_PATH = Path("data/reference/rivalries.csv")

TEAM_LOCATIONS_PATH = Path("data/reference/team_locations.csv")





def load_stadium_context() -> pd.DataFrame:

    if not STADIUM_CONTEXT_PATH.exists():

        return pd.DataFrame()

    df = pd.read_csv(STADIUM_CONTEXT_PATH)

    if df.empty:

        return df

    df["team_abbr"] = df["team_abbr"].astype(str).str.upper()

    numeric_cols = [

        "altitude_ft",

        "crowd_noise_score",

        "fan_hostility_score",

        "weather_snow_index",

        "weather_rain_index",

        "indoor",

    ]

    keep_cols = ["team_abbr"] + [c for c in numeric_cols if c in df.columns]

    return df[keep_cols].copy()





def load_rivalry_table() -> pd.DataFrame:

    if not RIVALRIES_PATH.exists():

        return pd.DataFrame()

    df = pd.read_csv(RIVALRIES_PATH)

    if df.empty:

        return df

    df["team_abbr"] = df["team_abbr"].astype(str).str.upper()

    df["rival_abbr"] = df["rival_abbr"].astype(str).str.upper()

    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce").fillna(0.0)

    df["rivalry_type"] = df["rivalry_type"].astype(str)

    df["notes"] = df.get("notes", "").astype(str)

    return df[["team_abbr", "rival_abbr", "rivalry_type", "intensity"]].copy()





def load_team_locations() -> pd.DataFrame:

    if not TEAM_LOCATIONS_PATH.exists():

        return pd.DataFrame()

    df = pd.read_csv(TEAM_LOCATIONS_PATH)

    if df.empty:

        return df

    df["team_abbr"] = df["team_abbr"].astype(str).str.upper()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["timezone_offset_hours"] = pd.to_numeric(df["timezone_offset_hours"], errors="coerce")

    return df





def _haversine_km(lat1, lon1, lat2, lon2):

    lat1 = pd.to_numeric(lat1, errors="coerce")

    lon1 = pd.to_numeric(lon1, errors="coerce")

    lat2 = pd.to_numeric(lat2, errors="coerce")

    lon2 = pd.to_numeric(lon2, errors="coerce")



    lat1_rad = np.deg2rad(lat1)

    lat2_rad = np.deg2rad(lat2)

    dlat = lat2_rad - lat1_rad

    dlon = np.deg2rad(lon2 - lon1)



    sin_dlat = np.sin(dlat / 2.0)

    sin_dlon = np.sin(dlon / 2.0)

    a = sin_dlat ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * sin_dlon ** 2

    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c





def _add_travel_features(feats: pd.DataFrame) -> pd.DataFrame:

    locs = load_team_locations()

    if locs.empty:

        return feats



    enriched = feats.copy()



    home_loc = locs.rename(

        columns={

            "team_abbr": "home_team",

            "latitude": "home_base_lat",

            "longitude": "home_base_lon",

            "timezone_offset_hours": "home_timezone_offset",

            "city": "home_base_city",

            "state": "home_base_state",

        }

    )

    away_loc = locs.rename(

        columns={

            "team_abbr": "away_team",

            "latitude": "away_base_lat",

            "longitude": "away_base_lon",

            "timezone_offset_hours": "away_timezone_offset",

            "city": "away_base_city",

            "state": "away_base_state",

        }

    )



    enriched = enriched.merge(home_loc, on="home_team", how="left")

    enriched = enriched.merge(away_loc, on="away_team", how="left")



    if "venue_lat" in enriched.columns:

        venue_lat = pd.to_numeric(enriched["venue_lat"], errors="coerce")

    else:

        venue_lat = pd.Series(np.nan, index=enriched.index)

    if "venue_lon" in enriched.columns:

        venue_lon = pd.to_numeric(enriched["venue_lon"], errors="coerce")

    else:

        venue_lon = pd.Series(np.nan, index=enriched.index)

    venue_lat = venue_lat.fillna(enriched.get("home_base_lat"))

    venue_lon = venue_lon.fillna(enriched.get("home_base_lon"))



    enriched["away_travel_distance_km"] = _haversine_km(

        enriched.get("away_base_lat"),

        enriched.get("away_base_lon"),

        venue_lat,

        venue_lon,

    )

    enriched["home_travel_distance_km"] = _haversine_km(

        enriched.get("home_base_lat"),

        enriched.get("home_base_lon"),

        venue_lat,

        venue_lon,

    )

    enriched["away_travel_distance_miles"] = enriched["away_travel_distance_km"] * 0.621371

    enriched["home_travel_distance_miles"] = enriched["home_travel_distance_km"] * 0.621371



    enriched["home_timezone_offset"] = pd.to_numeric(enriched.get("home_timezone_offset"), errors="coerce")

    enriched["away_timezone_offset"] = pd.to_numeric(enriched.get("away_timezone_offset"), errors="coerce")

    enriched["timezone_diff_hours"] = enriched["home_timezone_offset"] - enriched["away_timezone_offset"]

    enriched["timezone_diff_hours_abs"] = enriched["timezone_diff_hours"].abs()

    enriched["away_travel_east"] = (enriched["timezone_diff_hours"] > 0).astype("Int64")

    enriched["away_travel_west"] = (enriched["timezone_diff_hours"] < 0).astype("Int64")



    if "sched_short_rest_flag_away" in enriched.columns:

        short_rest_flag = pd.to_numeric(enriched["sched_short_rest_flag_away"], errors="coerce").fillna(0)

    else:

        short_rest_flag = pd.Series(0, index=enriched.index)

    if "sched_back_to_back_travel_away" in enriched.columns:

        back_to_back_flag = pd.to_numeric(enriched["sched_back_to_back_travel_away"], errors="coerce").fillna(0)

    else:

        back_to_back_flag = pd.Series(0, index=enriched.index)

    if "sched_rest_days_away" in enriched.columns:

        rest_days = pd.to_numeric(enriched["sched_rest_days_away"], errors="coerce")

    else:

        rest_days = pd.Series(np.nan, index=enriched.index)



    enriched["travel_km_short_rest"] = enriched["away_travel_distance_km"] * short_rest_flag

    enriched["travel_km_back_to_back"] = enriched["away_travel_distance_km"] * back_to_back_flag

    enriched["travel_km_per_rest_day"] = np.where(

        rest_days > 0,

        enriched["away_travel_distance_km"] / rest_days,

        np.nan,

    )

    enriched["timezone_diff_short_rest"] = enriched["timezone_diff_hours_abs"] * short_rest_flag



    return enriched





def _aggregate_team_week(df: pd.DataFrame, prefix: str) -> pd.DataFrame:

    if df is None or df.empty:

        return pd.DataFrame()

    team_col = _detect_team_col(df)

    if team_col is None:

        return pd.DataFrame()

    wk_col = "week" if "week" in df.columns else ("gameday" if "gameday" in df.columns else None)

    if wk_col is None:

        return pd.DataFrame()

    agg = df.copy()

    agg[team_col] = agg[team_col].astype(str).str.upper().str.strip().apply(_norm_abbr)

    if "season" not in agg.columns:

        return pd.DataFrame()

    numeric_cols = [

        c

        for c in agg.columns

        if c not in {"season", wk_col, team_col} and pd.api.types.is_numeric_dtype(agg[c])

    ]

    if not numeric_cols:

        return pd.DataFrame()

    grouped = (

        agg.groupby(["season", wk_col, team_col], as_index=False)[numeric_cols]

        .sum(min_count=1)

    )

    grouped = grouped.rename(columns={wk_col: "week", team_col: "team"})

    grouped = grouped.rename(columns={c: f"{prefix}_{c}" for c in numeric_cols})

    return grouped





def _add_weekly_deltas(

    df: pd.DataFrame,

    value_cols: Optional[List[str]] = None,

    rolling_windows: tuple[int, ...] = (3,),

) -> pd.DataFrame:

    if df is None or df.empty:

        return df

    required = {"season", "week", "team"}

    if not required.issubset(df.columns):

        return df



    out = df.copy()

    out["season"] = pd.to_numeric(out["season"], errors="coerce")

    out["week"] = pd.to_numeric(out["week"], errors="coerce")

    out = out.dropna(subset=["season", "week"])

    if out.empty:

        return df



    out = out.sort_values(["season", "team", "week"]).reset_index(drop=True)



    if value_cols is None:

        value_cols = [

            c

            for c in out.columns

            if c not in {"season", "week", "team"} and pd.api.types.is_numeric_dtype(out[c])

        ]

    else:

        value_cols = [c for c in value_cols if c in out.columns]



    if not value_cols:

        return out



    grouped = out.groupby(["season", "team"], group_keys=False)
    derived_cols: dict[str, pd.Series] = {}

    for col in value_cols:

        prev = grouped[col].shift(1)

        prev2 = grouped[col].shift(2)

        delta = prev - prev2

        with np.errstate(divide="ignore", invalid="ignore"):

            pct_change = delta / prev2.replace(0, np.nan)

        pct_change = pct_change.replace([np.inf, -np.inf], np.nan)

        derived_cols[f"{col}_prev"] = prev

        derived_cols[f"{col}_delta"] = delta

        derived_cols[f"{col}_pct_change"] = pct_change

        for window in rolling_windows:

            derived_cols[f"{col}_rolling{window}"] = grouped[col].transform(

                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()

            )

    if derived_cols:

        out = pd.concat([out, pd.DataFrame(derived_cols, index=out.index)], axis=1).copy()



    out["season"] = out["season"].astype("Int64")

    out["week"] = out["week"].astype("Int64")

    return out





def _moneyline_to_prob(odds: pd.Series | np.ndarray | Any) -> np.ndarray:

    arr = pd.to_numeric(odds, errors="coerce")

    if isinstance(arr, pd.Series):

        values = arr.to_numpy(dtype=float, copy=False)

    else:

        values = np.asarray(arr, dtype=float)

    result = np.full_like(values, np.nan, dtype=float)

    if result.size == 0:

        return result

    pos_mask = values > 0

    neg_mask = values < 0

    result[pos_mask] = 100.0 / (values[pos_mask] + 100.0)

    result[neg_mask] = (-values[neg_mask]) / ((-values[neg_mask]) + 100.0)

    return result





def _consecutive_counts(series: pd.Series) -> pd.Series:

    values = series.fillna(0).astype(int).tolist()

    out: list[int] = []

    run = 0

    for val in values:

        if val:

            run += 1

        else:

            run = 0

        out.append(run)

    return pd.Series(out, index=series.index, dtype="Int64")





def _build_schedule_team_features(

    sched_df: pd.DataFrame,

    seasons: List[int],

) -> pd.DataFrame:

    if sched_df is None or sched_df.empty:

        return pd.DataFrame()



    sched = _normalize_schedule_columns(sched_df)

    required = {"season", "week", "home_team", "away_team"}

    if not required.issubset(sched.columns):

        return pd.DataFrame()



    sched = sched[sched["season"].isin(seasons)].copy()

    if sched.empty:

        return pd.DataFrame()



    if "gameday" in sched.columns:
        sched["gameday_dt"] = pd.to_datetime(sched["gameday"], errors="coerce")
    else:
        sched["gameday_dt"] = pd.Series(pd.NaT, index=sched.index, dtype="datetime64[ns]")

    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")

    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")

    sched = sched.dropna(subset=["season", "week", "home_team", "away_team"])

    if sched.empty:

        return pd.DataFrame()



    records: list[dict[str, Any]] = []

    for _, row in sched.iterrows():

        season = int(row["season"])

        week = int(row["week"])

        gameday = row.get("gameday_dt")

        spread_line = pd.to_numeric(row.get("spread_line"), errors="coerce")

        total_line = pd.to_numeric(row.get("total_line"), errors="coerce")

        home_ml = pd.to_numeric(row.get("home_moneyline"), errors="coerce")

        away_ml = pd.to_numeric(row.get("away_moneyline"), errors="coerce")

        neutral = bool(row.get("neutral_site", False))

        for side in ("home", "away"):

            team = row.get(f"{side}_team")

            if not isinstance(team, str):

                continue

            team = team.upper().strip()

            rest_val = pd.to_numeric(row.get(f"{side}_rest"), errors="coerce")

            is_home = 1 if side == "home" else 0

            is_away = 1 - is_home



            spread_team = spread_line

            if pd.notna(spread_team) and side == "away":

                spread_team = -spread_team



            ml_team = home_ml if side == "home" else away_ml

            implied_prob = _moneyline_to_prob([ml_team])[0] if ml_team is not None else np.nan



            record = {

                "season": season,

                "week": week,

                "team": team,

                "sched_gameday": gameday,

                "sched_rest_days": rest_val,

                "sched_is_home": is_home,

                "sched_is_away": is_away,

                "sched_is_neutral_site": int(neutral),

                "sched_spread_close": spread_team,

                "sched_spread_close_abs": abs(spread_team) if pd.notna(spread_team) else np.nan,

                "sched_total_line": total_line,

                "sched_moneyline": ml_team,

                "sched_implied_prob": implied_prob,

            }

            records.append(record)



    if not records:

        return pd.DataFrame()



    team_df = pd.DataFrame(records)

    team_df = team_df.dropna(subset=["season", "week", "team"])

    if team_df.empty:

        return pd.DataFrame()



    team_df["season"] = team_df["season"].astype(int)

    team_df["week"] = team_df["week"].astype(int)
    team_df["sched_gameday"] = pd.to_datetime(team_df["sched_gameday"], errors="coerce")

    team_df.sort_values(["season", "team", "sched_gameday", "week"], inplace=True)



    grouped = team_df.groupby(["season", "team"], group_keys=False)

    prev_date = grouped["sched_gameday"].shift(1)

    team_df["sched_days_since_last_game"] = (

        (team_df["sched_gameday"] - prev_date).dt.days

    )

    team_df["sched_prev_is_away"] = grouped["sched_is_away"].shift(1).fillna(0).astype("Int64")

    team_df["sched_prev_is_home"] = grouped["sched_is_home"].shift(1).fillna(0).astype("Int64")



    team_df["sched_rest_days"] = team_df["sched_rest_days"].fillna(team_df["sched_days_since_last_game"])

    team_df["sched_rest_days"] = pd.to_numeric(team_df["sched_rest_days"], errors="coerce")



    team_df["sched_short_rest_flag"] = (

        (team_df["sched_rest_days"] <= 6).astype("Int64")

    )

    team_df["sched_long_rest_flag"] = (

        (team_df["sched_rest_days"] >= 9).astype("Int64")

    )



    team_df["sched_consecutive_away"] = grouped["sched_is_away"].transform(_consecutive_counts)

    team_df["sched_consecutive_home"] = grouped["sched_is_home"].transform(_consecutive_counts)

    team_df["sched_back_to_back_travel"] = (

        ((team_df["sched_is_away"] == 1) & (team_df["sched_prev_is_away"] == 1))

        .astype("Int64")

    )



    numeric_cols = [

        c

        for c in team_df.columns

        if c.startswith("sched_") and pd.api.types.is_numeric_dtype(team_df[c])

    ]

    team_df = _add_weekly_deltas(team_df, value_cols=numeric_cols)

    team_df.drop(columns=["sched_gameday"], inplace=True, errors="ignore")



    return team_df





def _build_player_news_features(

    news_df: pd.DataFrame,

    sched_df: pd.DataFrame,

    roster_df: pd.DataFrame,

    seasons: List[int],

) -> pd.DataFrame:

    if news_df is None or news_df.empty:

        return pd.DataFrame()



    news = news_df.copy()



    def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:

        if column in frame.columns:

            return pd.Series(frame[column], copy=True)

        return pd.Series([default] * len(frame), index=frame.index)



    if "season" in news.columns and seasons:

        news = news[news["season"].isin(seasons)]

    news["player_id"] = news["player_id"].apply(_to_int)

    news = news.dropna(subset=["player_id"])

    published_series = _series_or_default(news, "published", pd.NaT)

    last_modified_series = _series_or_default(news, "last_modified", pd.NaT)

    news["published_dt"] = pd.to_datetime(published_series, errors="coerce", utc=True)

    news["last_modified_dt"] = pd.to_datetime(last_modified_series, errors="coerce", utc=True)

    news = news.dropna(subset=["published_dt"])

    if news.empty:

        return pd.DataFrame()



    headline = _series_or_default(news, "headline", "").fillna("").astype(str)

    description = _series_or_default(news, "description", "").fillna("").astype(str)

    story = _series_or_default(news, "story", "").fillna("").astype(str)

    news["text"] = (headline + " " + description + " " + story).str.strip()

    news["text_lower"] = news["text"].str.lower()

    type_series = _series_or_default(news, "type", "").fillna("").astype(str)

    news["type_lower"] = type_series.str.lower()



    sched = sched_df[sched_df["season"].isin(seasons)].copy()

    if sched.empty:

        return pd.DataFrame()

    sched["game_start"] = pd.to_datetime(sched.get("start_date", sched.get("gameday")), errors="coerce", utc=True)

    home = (

        sched[["season", "week", "home_team", "game_start"]]

        .rename(columns={"home_team": "team"})

        .dropna(subset=["team", "game_start"])

    )

    away = (

        sched[["season", "week", "away_team", "game_start"]]

        .rename(columns={"away_team": "team"})

        .dropna(subset=["team", "game_start"])

    )

    team_games = pd.concat([home, away], ignore_index=True)

    team_games["team"] = team_games["team"].astype(str).str.upper().str.strip()



    roster = roster_df[roster_df["season"].isin(seasons)].copy()

    roster["espn_id"] = roster["espn_id"].apply(_to_int)

    roster = roster.dropna(subset=["espn_id"])

    roster["team"] = roster["team"].astype(str).str.upper().str.strip()

    roster = roster.dropna(subset=["team"])

    player_week = (

        roster[["season", "week", "team", "espn_id", "status"]]

        .drop_duplicates()

        .merge(team_games, on=["season", "week", "team"], how="left")

    )

    player_week = player_week.dropna(subset=["game_start"])

    if player_week.empty:

        return pd.DataFrame()



    news_join = player_week.merge(

        news,

        left_on="espn_id",

        right_on="player_id",

        how="left",

        suffixes=("", "_news"),

    )

    news_join = news_join.dropna(subset=["published_dt"])

    if news_join.empty:

        return pd.DataFrame()



    news_join["hours_before_game"] = (

        (news_join["game_start"] - news_join["published_dt"]).dt.total_seconds() / 3600.0

    )

    news_join = news_join[news_join["hours_before_game"].notna() & (news_join["hours_before_game"] >= 0)]

    if news_join.empty:

        return pd.DataFrame()



    window7 = 7 * 24

    window3 = 3 * 24

    news_join["within7"] = news_join["hours_before_game"] <= window7

    news_join["within3"] = news_join["hours_before_game"] <= window3

    news_join = news_join[news_join["within7"]]

    if news_join.empty:

        return pd.DataFrame()



    injury_pattern = re.compile(

        r"\b(?:ir|injur|questionable|doubtful|out|inactive|groin|ankle|knee|hamstring|concussion|shoulder|limited|practice|dnp)\b",

        re.IGNORECASE,

    )

    news_join["is_rotowire"] = (news_join["type_lower"] == "rotowire").astype(int)

    news_join["has_injury_keyword"] = news_join["text_lower"].str.contains(injury_pattern, na=False).astype(int)



    news_join["count7"] = news_join["within7"].astype(int)

    news_join["count3"] = news_join["within3"].astype(int)

    news_join["rotowire7"] = news_join["is_rotowire"] * news_join["count7"]

    news_join["injury_kw7"] = news_join["has_injury_keyword"] * news_join["count7"]



    agg = (

        news_join.groupby(["season", "week", "team"], as_index=False)

        .agg(

            news_count7=("count7", "sum"),

            news_count3=("count3", "sum"),

            news_rotowire7=("rotowire7", "sum"),

            news_injury_kw7=("injury_kw7", "sum"),

            news_hours_before_game_min=("hours_before_game", "min"),

        )

    )

    for col in ["news_count7", "news_count3", "news_rotowire7", "news_injury_kw7"]:

        agg[col] = agg[col].astype("Int64")

    agg["news_hours_before_game_min"] = agg["news_hours_before_game_min"].astype(float)

    return agg





def _build_team_news_features(

    news_df: pd.DataFrame,

    sched_df: pd.DataFrame,

    seasons: List[int],

) -> pd.DataFrame:

    if news_df is None or news_df.empty:

        return pd.DataFrame()



    news = news_df.copy()

    if "season" in news.columns and seasons:

        news = news[news["season"].isin(seasons)]

    if news.empty:

        return pd.DataFrame()



    news["team"] = news["team_abbr"].astype(str).str.upper().str.strip()

    news = news[news["team"].notna() & (news["team"] != "None")]

    if news.empty:

        return pd.DataFrame()



    news["headline"] = news.get("headline", "").fillna("").astype(str)

    news["description"] = news.get("description", "").fillna("").astype(str)

    news["text"] = (news["headline"] + " " + news["description"]).str.strip()

    news["text_lower"] = news["text"].str.lower()

    news["article_type"] = news.get("article_type", "").fillna("").astype(str).str.lower()

    news["is_story"] = (news["article_type"] == "story").astype(int)



    news["published_dt"] = pd.to_datetime(news.get("published"), errors="coerce", utc=True)

    news = news.dropna(subset=["published_dt"])

    if news.empty:

        return pd.DataFrame()



    sched = sched_df[sched_df["season"].isin(seasons)].copy()

    if sched.empty:

        return pd.DataFrame()

    if "start_date" in sched.columns:

        sched["game_start"] = pd.to_datetime(sched["start_date"], errors="coerce", utc=True)

    elif "gameday" in sched.columns:

        sched["game_start"] = pd.to_datetime(sched["gameday"], errors="coerce", utc=True)

    else:

        return pd.DataFrame()

    home_col = "home_abbreviation" if "home_abbreviation" in sched.columns else ("home_team" if "home_team" in sched.columns else None)

    away_col = "away_abbreviation" if "away_abbreviation" in sched.columns else ("away_team" if "away_team" in sched.columns else None)

    if home_col is None or away_col is None:

        return pd.DataFrame()

    home = (

        sched[["season", "week", home_col, "game_start"]]

        .rename(columns={home_col: "team"})

        .dropna(subset=["team", "game_start"])

    )

    away = (

        sched[["season", "week", away_col, "game_start"]]

        .rename(columns={away_col: "team"})

        .dropna(subset=["team", "game_start"])

    )

    team_games = pd.concat([home, away], ignore_index=True)

    team_games["team"] = team_games["team"].astype(str).str.upper().str.strip()

    team_games = team_games.dropna(subset=["team", "game_start"])

    if team_games.empty:

        return pd.DataFrame()



    joined = team_games.merge(

        news,

        on=["season", "team"],

        how="left",

        suffixes=("", "_news"),

    )

    joined = joined.dropna(subset=["published_dt"])

    if joined.empty:

        return pd.DataFrame()



    joined["hours_before_game"] = (

        (joined["game_start"] - joined["published_dt"]).dt.total_seconds() / 3600.0

    )

    joined = joined[joined["hours_before_game"].notna()]

    if joined.empty:

        return pd.DataFrame()



    window7 = 7 * 24

    window3 = 3 * 24

    joined["abs_hours"] = joined["hours_before_game"].abs()

    joined["within7"] = joined["abs_hours"] <= window7

    joined["within3"] = joined["abs_hours"] <= window3

    joined = joined[joined["within7"]]

    if joined.empty:

        return pd.DataFrame()



    joined["is_pre_game"] = (joined["hours_before_game"] >= 0).astype(int)

    injury_pattern = re.compile(

        r"\b(?:ir|injur|questionable|doubtful|out|inactive|groin|ankle|knee|hamstring|concussion|shoulder|limited|practice|dnp)\b",

        re.IGNORECASE,

    )

    joined["has_injury_keyword"] = joined["text_lower"].str.contains(injury_pattern, na=False).astype(int)



    joined["count7"] = joined["within7"].astype(int)

    joined["count3"] = joined["within3"].astype(int)

    joined["story7"] = joined["is_story"] * joined["count7"]

    joined["injury_kw7"] = joined["has_injury_keyword"] * joined["count7"]

    joined["count7_pre"] = joined["count7"] * joined["is_pre_game"]

    joined["count7_post"] = joined["count7"] * (1 - joined["is_pre_game"])



    agg = (

        joined.groupby(["season", "week", "team"], as_index=False)

        .agg(

            team_news_count7=("count7", "sum"),

            team_news_count3=("count3", "sum"),

            team_news_story7=("story7", "sum"),

            team_news_injury_kw7=("injury_kw7", "sum"),

            team_news_count7_pre=("count7_pre", "sum"),

            team_news_count7_post=("count7_post", "sum"),

            team_news_hours_before_game_min=("hours_before_game", "min"),

        )

    )

    for col in ["team_news_count7", "team_news_count3", "team_news_story7", "team_news_injury_kw7", "team_news_count7_pre", "team_news_count7_post"]:

        agg[col] = agg[col].astype("Int64")

    agg["team_news_hours_before_game_min"] = agg["team_news_hours_before_game_min"].astype(float)

    return agg





def _augment_team_week_features(

    feats: pd.DataFrame,

    seasons: List[int],

    load_raw: callable,

    sched_df: pd.DataFrame,

) -> pd.DataFrame:

    out = feats.copy()

    pbp_sources: list[pd.DataFrame] = []

    for source in ("nfl_pbp.parquet", "espn_pbp.parquet"):

        raw = load_raw(source)

        if not raw.empty:

            pbp_sources.append(raw)

    pbp_df = pd.concat(pbp_sources, ignore_index=True) if pbp_sources else pd.DataFrame()

    if not pbp_df.empty and "season" in pbp_df.columns:

        pbp_df = pbp_df[pbp_df["season"].isin(seasons)].copy()



    def _merge_home_away(base_df: pd.DataFrame) -> pd.DataFrame:

        if base_df is None or base_df.empty:

            return out

        base_df = base_df.drop_duplicates(subset=["season", "week", "team"]).copy()

        key_cols = ["season", "week", "team"]

        value_cols = [c for c in base_df.columns if c not in key_cols]

        home_df = base_df[key_cols + value_cols].rename(columns={c: f"{c}_home" for c in value_cols})

        merged = out.merge(

            home_df,

            left_on=["season", "week", "home_team"],

            right_on=["season", "week", "team"],

            how="left",

        )

        merged.drop(columns=["team"], inplace=True, errors="ignore")

        away_df = base_df[key_cols + value_cols].rename(columns={c: f"{c}_away" for c in value_cols})

        merged = merged.merge(

            away_df,

            left_on=["season", "week", "away_team"],

            right_on=["season", "week", "team"],

            how="left",

        )

        merged.drop(columns=["team"], inplace=True, errors="ignore")

        return merged



    def _add_feature_diffs(df: pd.DataFrame, bases: List[str]) -> pd.DataFrame:

        for base in bases:

            home_col, away_col = f"{base}_home", f"{base}_away"

            if home_col in df.columns and away_col in df.columns:

                df[f"{base}_diff"] = (

                    pd.to_numeric(df[home_col], errors="coerce")

                    - pd.to_numeric(df[away_col], errors="coerce")

                )

        return df



    sched_team_features = _build_schedule_team_features(sched_df, seasons)

    if not sched_team_features.empty:

        out = _merge_home_away(sched_team_features)

        sched_cols = [c for c in out.columns if c.startswith("sched_")]

        for col in sched_cols:

            out[col] = pd.to_numeric(out[col], errors="coerce")

        diff_candidates = [

            "sched_rest_days",

            "sched_spread_close",

            "sched_spread_close_abs",

            "sched_implied_prob",

            "sched_consecutive_away",

            "sched_consecutive_home",

            "sched_back_to_back_travel",

            "sched_short_rest_flag",

            "sched_long_rest_flag",

            "sched_rest_days_delta",

            "sched_spread_close_delta",

            "sched_spread_close_prev",

            "sched_implied_prob_delta",

        ]

        diff_data = {}

        for base in diff_candidates:

            h_col, a_col = f"{base}_home", f"{base}_away"

            if h_col in out.columns and a_col in out.columns:

                diff_data[f"{base}_diff"] = pd.to_numeric(out[h_col], errors="coerce") - pd.to_numeric(out[a_col], errors="coerce")

        if diff_data:

            out = pd.concat([out, pd.DataFrame(diff_data, index=out.index)], axis=1)



    # Injuries

    inj = load_raw("nfl_injuries.parquet")

    if not inj.empty and "season" in inj.columns:

        inj = inj[inj["season"].isin(seasons)].copy()

    inj_agg = pd.DataFrame()

    if not inj.empty:

        team_col = _detect_team_col(inj)

        has_season_week = {"season", "week"}.issubset(inj.columns)

        status_like = next((c for c in ["status", "report_status"] if c in inj.columns), None)

        practice_like = next((c for c in ["practice_status", "practice_primary_injury"] if c in inj.columns), None)

        if team_col and has_season_week and status_like:

            injc = inj.copy()

            injc[team_col] = injc[team_col].astype(str).str.upper().str.strip().apply(_norm_abbr)

            injc = injc.dropna(subset=[team_col, "season", "week"])

            status_series = injc[status_like].astype(str).str.lower()

            injc["inj_out"] = status_series.str.contains("out", na=False).astype(int)

            injc["inj_doubtful"] = status_series.str.contains("doubt", na=False).astype(int)

            injc["inj_questionable"] = status_series.str.contains("question", na=False).astype(int)

            injc["inj_reserve"] = status_series.str.contains("injured|reserve", na=False).astype(int)

            if practice_like:

                practice_series = injc[practice_like].astype(str).str.lower()

                injc["inj_practice_dnp"] = practice_series.str.contains("did not practice|dnp", na=False).astype(int)

                injc["inj_practice_limited"] = practice_series.str.contains("limited", na=False).astype(int)

                injc["inj_practice_full"] = practice_series.str.contains("full", na=False).astype(int)

            else:

                injc["inj_practice_dnp"] = 0

                injc["inj_practice_limited"] = 0

                injc["inj_practice_full"] = 0

            if "position" in injc.columns:

                pos_series = injc["position"].astype(str).str.upper()

            else:

                pos_series = pd.Series([""] * len(injc), index=injc.index)

            is_qb = pos_series.str.contains("QB", na=False)

            injc["inj_qb_out"] = injc["inj_out"] * is_qb.astype(int)

            injc["inj_qb_doubtful"] = injc["inj_doubtful"] * is_qb.astype(int)

            injc["inj_qb_questionable"] = injc["inj_questionable"] * is_qb.astype(int)

            injc["inj_qb_reserve"] = injc["inj_reserve"] * is_qb.astype(int)

            injc["inj_listed_total"] = (

                injc[["inj_out", "inj_doubtful", "inj_questionable", "inj_reserve"]]

                .sum(axis=1, min_count=1)

                .fillna(0)

            )

            agg_cols = [

                "inj_out",

                "inj_doubtful",

                "inj_questionable",

                "inj_reserve",

                "inj_practice_dnp",

                "inj_practice_limited",

                "inj_practice_full",

                "inj_listed_total",

                "inj_qb_out",

                "inj_qb_doubtful",

                "inj_qb_questionable",

                "inj_qb_reserve",

            ]

            inj_agg = (

                injc.groupby(["season", "week", team_col], as_index=False)[agg_cols]

                .sum(min_count=1)

                .rename(columns={team_col: "team"})

            )

            inj_agg = _add_weekly_deltas(inj_agg, value_cols=agg_cols)

            inj_agg = inj_agg[inj_agg["season"].isin(seasons)].copy()

            out_local = _merge_home_away(inj_agg)

            for col in agg_cols:

                h, a = f"{col}_home", f"{col}_away"

                if h in out_local.columns and a in out_local.columns:

                    out_local[f"{col}_diff"] = pd.to_numeric(out_local[h], errors="coerce") - pd.to_numeric(out_local[a], errors="coerce")

            out = out_local



    dataset_specs = [

        ("nfl_snap_counts.parquet", "snaps"),

        ("nfl_depth_charts.parquet", "depth"),

        ("nfl_rosters.parquet", "roster"),

        ("nfl_player_stats.parquet", "pstats"),

    ]



    rosters_raw = pd.DataFrame()

    for name, prefix in dataset_specs:

        raw = load_raw(name)

        if raw.empty:

            continue

        if "season" in raw.columns:

            raw = raw[raw["season"].isin(seasons)].copy()

        if prefix == "roster":

            rosters_raw = raw.copy()

        agg = _aggregate_team_week(raw, prefix)

        if agg.empty:

            continue

        agg = agg[agg["season"].isin(seasons)].copy()

        value_cols = [c for c in agg.columns if c.startswith(f"{prefix}_")]

        agg = _add_weekly_deltas(agg, value_cols=value_cols)

        out = _merge_home_away(agg)



    # QB health/injury context (requires roster + injury inputs)

    try:

        qb_features = build_qb_health_features(

            sched_df=sched_df,

            injuries_df=inj if not inj.empty else pd.DataFrame(),

            roster_df=rosters_raw,

            seasons=seasons,

            norm_team=_norm_abbr,

        )

        if not qb_features.empty:

            out = _merge_home_away(qb_features)

            out = _add_feature_diffs(

                out,

                [

                    "qb_status_flag",

                    "qb_status_delta_rolling3",

                    "qb_missed_last_game",

                    "qb_games_started_rolling5",

                    "qb_recovery_score_rolling3",

                    "qb_weeks_since_injury",

                ],

            )

    except Exception as exc:

        print(f"[features] qb health merge skipped: {exc}")



    # Passing EPA, adjusted efficiency, red-zone, and pressure metrics (PBP derived)

    if not pbp_df.empty:

        try:

            epa_features = build_passing_epa_features(pbp_df, seasons, _norm_abbr)

            if not epa_features.empty:

                out = _merge_home_away(epa_features)

                out = _add_feature_diffs(

                    out,

                    [

                        "pass_epa_per_db",

                        "pass_epa_per_db_rolling3",

                        "pass_epa_per_db_rolling5",

                        "pass_epa_per_db_rolling3_diff_lg",

                    ],

                )

        except Exception as exc:

            print(f"[features] passing EPA merge skipped: {exc}")



        try:

            drive_epa_feats = build_drive_epa_features(pbp_df, seasons, _norm_abbr)

            if not drive_epa_feats.empty:

                out = _merge_home_away(drive_epa_feats)

                out = _add_feature_diffs(

                    out,

                    [

                        "drive_epa_mean",

                        "drive_epa_mean_rolling3",

                        "drive_epa_first_drive_rolling3",

                        "drive_epa_q4_rolling3",

                        "drive_epa_red_zone_rolling3",

                        "drive_completion_rate_rolling3",

                        "drives_per_game_rolling3",

                        "drive_epa_mean_rolling3_diff_lg",

                    ],

                )

        except Exception as exc:

            print(f"[features] drive EPA merge skipped: {exc}")





        try:

            adj_features = build_adjusted_efficiency_features(pbp_df, sched_df, seasons, _norm_abbr)

            if not adj_features.empty:

                out = _merge_home_away(adj_features)

                out = _add_feature_diffs(

                    out,

                    [

                        "off_adj_eff_raw",

                        "off_adj_eff_rolling3",

                        "def_adj_eff_raw",

                        "def_adj_eff_rolling3",

                    ],

                )

        except Exception as exc:

            print(f"[features] adjusted efficiency merge skipped: {exc}")



        try:

            rz_features = build_redzone_features(pbp_df, seasons, _norm_abbr)

            if not rz_features.empty:

                out = _merge_home_away(rz_features)

                out = _add_feature_diffs(

                    out,

                    [

                        "rz_trips_for",

                        "rz_tds_for",

                        "rz_trips_against",

                        "rz_tds_against",

                        "rz_td_rate_for",

                        "rz_td_rate_against",

                        "rz_td_rate_for_rolling3",

                        "rz_td_rate_against_rolling3",

                        "rz_trips_for_rolling3",

                        "rz_trips_against_rolling3",

                    ],

                )

        except Exception as exc:

            print(f"[features] red-zone merge skipped: {exc}")



        try:

            pressure_features = build_pressure_features(pbp_df, seasons, _norm_abbr)

            if not pressure_features.empty:

                out = _merge_home_away(pressure_features)

                out = _add_feature_diffs(

                    out,

                    [

                        "pressures_allowed_per_db",

                        "sack_rate_allowed",

                        "pressures_allowed_per_db_rolling3",

                        "sack_rate_allowed_rolling3",

                        "pressures_generated_per_db",

                        "sack_rate_generated",

                        "pressures_generated_per_db_rolling3",

                        "sack_rate_generated_rolling3",

                    ],

                )

        except Exception as exc:

            print(f"[features] pressure merge skipped: {exc}")



    # Player news

    try:

        news_path = RAW_DIR / "espn_player_news.parquet"

        if news_path.exists():

            news_raw = read_df(news_path)

            if "season" in news_raw.columns:

                news_raw = news_raw[news_raw["season"].isin(seasons)].copy()

            rosters_source = rosters_raw if not rosters_raw.empty else load_raw("nfl_rosters.parquet")

            sched_for_news = sched_df[sched_df["season"].isin(seasons)].copy() if "season" in sched_df.columns else sched_df.copy()

            news_features = _build_player_news_features(

                news_raw,

                sched_for_news,

                rosters_source if not rosters_source.empty else pd.DataFrame(),

                seasons,

            )

            if not news_features.empty:

                out = _merge_home_away(news_features)

                for base in ["news_count7", "news_count3", "news_rotowire7", "news_injury_kw7"]:

                    h, a = f"{base}_home", f"{base}_away"

                    if h in out.columns:

                        out[h] = pd.to_numeric(out[h], errors="coerce").fillna(0)

                    if a in out.columns:

                        out[a] = pd.to_numeric(out[a], errors="coerce").fillna(0)

                        out[f"{base}_diff"] = out[h] - out[a]

    except Exception as e:

        print(f"[features] player news merge skipped: {e}")



    # Team news

    try:

        team_news_path = RAW_DIR / "espn_news.parquet"

        if team_news_path.exists():

            team_news_raw = read_df(team_news_path)

            if "season" in team_news_raw.columns:

                team_news_raw = team_news_raw[team_news_raw["season"].isin(seasons)].copy()

            sched_for_news = sched_df[sched_df["season"].isin(seasons)].copy() if "season" in sched_df.columns else sched_df.copy()

            team_news_features = _build_team_news_features(

                team_news_raw,

                sched_for_news,

                seasons,

            )

            if not team_news_features.empty:

                out = _merge_home_away(team_news_features)

                for base in ["team_news_count7", "team_news_count3", "team_news_story7", "team_news_injury_kw7", "team_news_count7_pre", "team_news_count7_post"]:

                    h, a = f"{base}_home", f"{base}_away"

                    if h in out.columns:

                        out[h] = pd.to_numeric(out[h], errors="coerce").fillna(0)

                    if a in out.columns:

                        out[a] = pd.to_numeric(out[a], errors="coerce").fillna(0)

                        out[f"{base}_diff"] = out[h] - out[a]

    except Exception as e:

        print(f"[features] team news merge skipped: {e}")



    for prefix in ("inj_", "snaps_", "depth_", "roster_", "pstats_", "news_", "team_news_"):

        cols = [c for c in out.columns if c.startswith(prefix)]

        for col in cols:

            out[col] = pd.to_numeric(out[col], errors="coerce")



    return out





if __name__ == "__main__":

    main()

EARTH_RADIUS_KM = 6371.0

