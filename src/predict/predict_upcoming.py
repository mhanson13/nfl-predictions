
from __future__ import annotations
import argparse
import os
import re
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from src.utils.io import RAW_DIR, PROC_DIR
from src.utils.logging import configure as configure_logging
from src.utils.odds import fetch_odds

TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Redskins": "WAS",
    "Washington Football Team": "WAS",
    "Arizona": "ARI",
    "Atlanta": "ATL",
    "Baltimore": "BAL",
    "Buffalo": "BUF",
    "Carolina": "CAR",
    "Chicago": "CHI",
    "Cincinnati": "CIN",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Green Bay": "GB",
    "Houston": "HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "Kansas City": "KC",
    "Las Vegas": "LV",
    "LA Chargers": "LAC",
    "Los Angeles Chargers": "LAC",
    "LA Rams": "LAR",
    "Los Angeles Rams": "LAR",
    "Miami": "MIA",
    "Minnesota": "MIN",
    "New England": "NE",
    "New Orleans": "NO",
    "NY Giants": "NYG",
    "New York Giants": "NYG",
    "NY Jets": "NYJ",
    "New York Jets": "NYJ",
    "Philadelphia": "PHI",
    "Pittsburgh": "PIT",
    "San Francisco": "SF",
    "Seattle": "SEA",
    "Tampa Bay": "TB",
    "Tennessee": "TEN",
    "Washington": "WAS",
}

def _make_feature_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Create *_diff = *_home - *_away features to mirror training preprocessing."""
    feat_df = df.copy()
    cols = set(feat_df.columns)
    diffs: Dict[str, pd.Series] = {}
    for c in list(cols):
        if not isinstance(c, str) or not c.endswith("_home"):
            continue
        base = c[:-5]
        away_col = base + "_away"
        if away_col not in cols:
            continue
        s_home = pd.to_numeric(feat_df[c], errors="coerce")
        s_away = pd.to_numeric(feat_df[away_col], errors="coerce")
        diffs[f"{base}_diff"] = s_home.astype(float) - s_away.astype(float)
    if diffs:
        feat_df = pd.concat([feat_df, pd.DataFrame(diffs, index=feat_df.index)], axis=1)
    return feat_df.copy()


def _cleanup_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate, all-null, and all-zero numeric columns."""
    cleaned = df.copy()
    idx = pd.Index(cleaned.columns)
    if idx.duplicated().any():
        dup_mask = idx.duplicated(keep="last")
        if dup_mask.any():
            cleaned = cleaned.loc[:, ~dup_mask]
    all_na = [c for c in cleaned.columns if cleaned[c].isna().all()]
    if all_na:
        cleaned = cleaned.drop(columns=all_na)
    numeric_cols = cleaned.select_dtypes(include="number").columns
    all_zero = []
    for col in numeric_cols:
        series = pd.to_numeric(cleaned[col], errors="coerce")
        if not series.notna().any():
            continue
        if series.fillna(0).ne(0).any():
            continue
        all_zero.append(col)
    if all_zero:
        cleaned = cleaned.drop(columns=all_zero)
    return cleaned


def _normalize_team_name(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    candidate = name.strip()
    if not candidate:
        return None
    abbr = TEAM_NAME_TO_ABBR.get(candidate)
    if abbr:
        return abbr
    lowered = candidate.lower()
    if lowered.startswith("the "):
        abbr = TEAM_NAME_TO_ABBR.get(candidate[4:])
        if abbr:
            return abbr
    for full, code in TEAM_NAME_TO_ABBR.items():
        if candidate.lower() == full.lower():
            return code
    return None

def _load_odds_api_key() -> Optional[str]:
    key = os.getenv("ODDS_API_KEY")
    if key:
        return key
    secrets_path = Path(__file__).resolve().parents[2] / "secrets.env"
    if not secrets_path.exists():
        return None
    try:
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "ODDS_API_KEY":
                val = v.strip().strip('"').strip("'")
                if val:
                    os.environ.setdefault("ODDS_API_KEY", val)
                    return val
    except Exception:
        return None
    return os.getenv("ODDS_API_KEY")

def _slugify_bookmaker(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "bookmaker"

def _attach_odds_spreads(preds: pd.DataFrame, odds: Optional[list[dict]], debug: bool = False) -> List[str]:
    if odds is None or preds.empty:
        return []
    cols_added: Set[str] = set()
    for event in odds:
        try:
            home_abbr = _normalize_team_name(event.get("home_team"))
            away_abbr = _normalize_team_name(event.get("away_team"))
            if not home_abbr or not away_abbr:
                continue
            mask = (preds["home_team"].astype(str).str.upper() == home_abbr) & (
                preds["away_team"].astype(str).str.upper() == away_abbr
            )
            if not mask.any():
                continue
            commence = event.get("commence_time")
            for book in event.get("bookmakers", []):
                book_key = book.get("key") or book.get("title") or "bookmaker"
                slug = _slugify_bookmaker(book_key)
                market = None
                for mk in book.get("markets", []):
                    if mk.get("key") == "spreads":
                        market = mk
                        break
                if market is None:
                    continue
                outcomes = market.get("outcomes", [])
                home_line = home_price = away_line = away_price = None
                for outcome in outcomes:
                    name = outcome.get("name")
                    abbr = _normalize_team_name(name)
                    if abbr == home_abbr:
                        home_line = outcome.get("point")
                        home_price = outcome.get("price")
                    elif abbr == away_abbr:
                        away_line = outcome.get("point")
                        away_price = outcome.get("price")
                columns = {
                    f"odds_{slug}_home_spread": home_line,
                    f"odds_{slug}_home_price": home_price,
                    f"odds_{slug}_away_spread": away_line,
                    f"odds_{slug}_away_price": away_price,
                }
                if commence:
                    columns[f"odds_{slug}_last_update"] = book.get("last_update") or commence
                for col in columns:
                    if col not in preds.columns:
                        preds[col] = pd.NA
                    cols_added.add(col)
                for col, value in columns.items():
                    preds.loc[mask, col] = value
        except Exception:
            if debug:
                print("[predict][odds] failed to merge an odds event")
            continue
    return sorted(cols_added)





def _load_player_stats_frame() -> pd.DataFrame:
    path = RAW_DIR / "nfl_player_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.copy()
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce")
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce")
    df = df[pd.notna(df["season"]) & pd.notna(df["week"])]
    df["season"] = df["season"].astype("int64")
    df["week"] = df["week"].astype("int64")
    df["season_type"] = pd.to_numeric(df.get("season_type"), errors="coerce")
    df = df[df["season_type"].fillna(2) == 2]
    if "recent_team" not in df.columns:
        df["recent_team"] = pd.NA
    df["recent_team"] = df["recent_team"].fillna("").astype(str).str.upper()
    if "player_id" not in df.columns:
        df["player_id"] = pd.NA
    df["player_id"] = df["player_id"].fillna("").astype(str)
    if "player_name" not in df.columns:
        if "player_display_name" in df.columns:
            df["player_name"] = df["player_display_name"].astype(str)
        else:
            df["player_name"] = ""
    else:
        df["player_name"] = df["player_name"].astype(str)
    return df


def _load_roster_frame() -> pd.DataFrame:
    path = RAW_DIR / "nfl_rosters.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.copy()
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce")
    df = df[pd.notna(df["season"])]
    df["season"] = df["season"].astype("int64")
    if "team" not in df.columns:
        df["team"] = pd.NA
    df["team"] = df["team"].fillna("").astype(str).str.upper()
    if "gsis_id" not in df.columns:
        df["gsis_id"] = pd.NA
    df["gsis_id"] = df["gsis_id"].fillna("").astype(str)
    if "jersey_number" not in df.columns:
        df["jersey_number"] = pd.NA
    if "full_name" not in df.columns:
        df["full_name"] = pd.NA
    return df[["season", "team", "gsis_id", "jersey_number", "full_name"]]


def _attach_roster_info(
    agg: pd.DataFrame,
    roster: pd.DataFrame,
    season: int,
    team_col: str,
    fill_name: bool = False,
) -> pd.DataFrame:
    agg = agg.copy()
    if "jersey_number" not in agg.columns:
        agg["jersey_number"] = pd.NA
    if roster is None or roster.empty:
        agg[team_col] = agg.get(team_col, "").fillna("").astype(str).str.upper()
        return agg

    roster_all = roster
    roster_season = roster_all[roster_all["season"] == season]

    def _apply(subset: pd.DataFrame, mask: pd.Series, update_team: bool = True) -> None:
        subset = subset.dropna(subset=["gsis_id"]).drop_duplicates(subset=["gsis_id"], keep="last")
        if subset.empty:
            return
        jersey_map = subset.set_index("gsis_id")["jersey_number"]
        team_map = subset.set_index("gsis_id")["team"]
        name_map = subset.set_index("gsis_id")["full_name"]

        idx = agg.index[mask]
        if not len(idx):
            return

        jerseys = agg.loc[idx, "player_id"].map(jersey_map)
        jersey_fill_mask = agg.loc[idx, "jersey_number"].isna() & jerseys.notna()
        if jersey_fill_mask.any():
            agg.loc[idx[jersey_fill_mask], "jersey_number"] = jerseys.loc[idx[jersey_fill_mask]]

        if update_team:
            teams = agg.loc[idx, "player_id"].map(team_map)
            team_current = agg.loc[idx, team_col].fillna("")
            team_fill_mask = teams.notna() & (
                team_current.eq("") | team_current.eq("TOT") | team_current.eq("ALL")
            )
            if team_fill_mask.any():
                agg.loc[idx[team_fill_mask], team_col] = teams.loc[idx[team_fill_mask]].astype(str).str.upper()

        if fill_name and "player_name" in agg.columns:
            names = agg.loc[idx, "player_id"].map(name_map)
            name_current = agg.loc[idx, "player_name"].fillna("")
            name_fill_mask = names.notna() & name_current.eq("")
            if name_fill_mask.any():
                agg.loc[idx[name_fill_mask], "player_name"] = names.loc[idx[name_fill_mask]].astype(str)

    all_mask = pd.Series(True, index=agg.index)
    if not roster_season.empty:
        _apply(roster_season, all_mask, update_team=True)
    missing_mask = agg["jersey_number"].isna()
    if missing_mask.any():
        _apply(roster_all, missing_mask, update_team=False)

    agg[team_col] = agg.get(team_col, "").fillna("").astype(str).str.upper()
    return agg


def _select_stats_window(stats: pd.DataFrame, season: int, week: Optional[int]) -> pd.DataFrame:
    current = pd.DataFrame()
    if week is not None:
        try:
            week_int = int(week)
        except Exception:
            week_int = None
        if week_int is not None:
            current = stats[(stats["season"] == season) & (stats["week"] < week_int)].copy()
    else:
        current = stats[stats["season"] == season].copy()

    history = stats[stats["season"] < season].copy()
    subset = pd.concat([history, current], ignore_index=True)
    return subset


def _player_qb_predictions(
    pred_games: pd.DataFrame,
    save_path: Optional[str],
    stats: pd.DataFrame,
    roster: pd.DataFrame,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    if not save_path:
        return None
    if pred_games is None or pred_games.empty:
        return None
    if stats is None or stats.empty:
        if debug:
            print("[predict][players] no player stats available for QB projections")
        return None

    qb_stats = stats[stats.get("position_group").isin(["QB"])].copy()
    if qb_stats.empty:
        if debug:
            print("[predict][players] no QB stats available")
        return None

    for col in [
        "passing_yards",
        "passing_tds",
        "rushing_tds",
        "attempts",
        "interceptions",
        "completions",
        "rushing_yards",
    ]:
        if col not in qb_stats.columns:
            qb_stats[col] = 0.0

    results: List[Dict[str, Any]] = []
    grouped = pred_games.groupby(["season", "week"])
    for (season, week), games in grouped:
        subset = _select_stats_window(qb_stats, int(season), int(week))
        if subset.empty:
            continue
        subset = subset[subset["recent_team"].notna() & (subset["recent_team"] != "")]
        if subset.empty:
            continue
        subset["season_week"] = subset["season"].astype(str) + "_" + subset["week"].astype(str)
        agg = subset.groupby(["player_id", "recent_team"], as_index=False).agg(
            player_name=("player_name", "last"),
            games_played=("season_week", "nunique"),
            passing_yards=("passing_yards", "sum"),
            passing_tds=("passing_tds", "sum"),
            rushing_tds=("rushing_tds", "sum"),
            passing_attempts=("attempts", "sum"),
            interceptions=("interceptions", "sum"),
            completions=("completions", "sum"),
            rushing_yards=("rushing_yards", "sum"),
        )
        if agg.empty:
            continue
        agg["season"] = season
        agg = _attach_roster_info(agg, roster, int(season), team_col="recent_team")
        gp = agg["games_played"].replace({0: pd.NA})
        agg["projected_passing_yards"] = (agg["passing_yards"] / gp).astype(float)
        agg["projected_passing_tds"] = (agg["passing_tds"] / gp).astype(float)
        agg["projected_rushing_tds"] = (agg["rushing_tds"] / gp).astype(float)
        agg["projected_passing_attempts"] = (agg["passing_attempts"] / gp).astype(float)
        agg["projected_interceptions"] = (agg["interceptions"] / gp).astype(float)
        agg["projected_completions"] = (agg["completions"] / gp).astype(float)
        agg["projected_rushing_yards"] = (agg["rushing_yards"] / gp).astype(float)

        for _, game in games.iterrows():
            game_id = game.get("game_id")
            for side in ("home", "away"):
                team = str(game.get(f"{side}_team", "")).upper()
                if not team:
                    continue
                players = agg[agg["recent_team"] == team].copy()
                if players.empty:
                    continue
                players = players.sort_values(
                    ["projected_passing_yards", "projected_passing_attempts"],
                    ascending=[False, False]
                )
                for rank, row in enumerate(players.itertuples(index=False), start=1):
                    results.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": game_id,
                            "team": team,
                            "team_side": side,
                            "player_rank": rank,
                            "player_id": row.player_id,
                            "player_name": row.player_name,
                            "jersey_number": row.jersey_number,
                            "games_sampled": row.games_played,
                            "projected_passing_yards": None if pd.isna(row.projected_passing_yards) else round(float(row.projected_passing_yards), 1),
                            "projected_passing_tds": None if pd.isna(row.projected_passing_tds) else round(float(row.projected_passing_tds), 2),
                            "projected_rushing_tds": None if pd.isna(row.projected_rushing_tds) else round(float(row.projected_rushing_tds), 2),
                            "projected_passing_attempts": None if pd.isna(row.projected_passing_attempts) else round(float(row.projected_passing_attempts), 2),
                            "projected_interceptions": None if pd.isna(row.projected_interceptions) else round(float(row.projected_interceptions), 2),
                            "projected_completions": None if pd.isna(row.projected_completions) else round(float(row.projected_completions), 2),
                            "projected_rushing_yards": None if pd.isna(row.projected_rushing_yards) else round(float(row.projected_rushing_yards), 1),
                        }
                    )
    if not results:
        return None
    df_qb = pd.DataFrame(results)
    df_qb.to_csv(save_path, index=False)
    if debug:
        print(f"[predict][players] saved QB projections -> {save_path} (rows={len(df_qb)})")
    return df_qb



def _player_offense_predictions(
    pred_games: pd.DataFrame,
    save_path: Optional[str],
    stats: pd.DataFrame,
    roster: pd.DataFrame,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    if not save_path:
        return None
    if pred_games is None or pred_games.empty or stats is None or stats.empty:
        return None

    offense_stats = stats[stats.get("position_group").isin(["RB", "WR", "TE"])].copy()
    if offense_stats.empty:
        if debug:
            print("[predict][players] no offensive skill-position stats available")
        return None

    needed_cols = [
        "rushing_yards",
        "rushing_tds",
        "carries",
        "receiving_yards",
        "receiving_tds",
        "receptions",
        "targets",
    ]
    for col in needed_cols:
        if col not in offense_stats.columns:
            offense_stats[col] = 0.0

    results: List[Dict[str, Any]] = []
    grouped = pred_games.groupby(["season", "week"])
    for (season, week), games in grouped:
        subset = _select_stats_window(offense_stats, int(season), int(week))
        if subset.empty:
            continue
        subset = subset[subset["recent_team"].notna() & (subset["recent_team"] != "")]
        if subset.empty:
            continue
        subset["season_week"] = subset["season"].astype(str) + "_" + subset["week"].astype(str)
        agg = subset.groupby(["player_id", "recent_team"], as_index=False).agg(
            player_name=("player_name", "last"),
            games_played=("season_week", "nunique"),
            rushing_yards=("rushing_yards", "sum"),
            rushing_tds=("rushing_tds", "sum"),
            carries=("carries", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_tds=("receiving_tds", "sum"),
            receptions=("receptions", "sum"),
            targets=("targets", "sum"),
        )
        if agg.empty:
            continue
        agg["season"] = season
        agg = _attach_roster_info(agg, roster, int(season), team_col="recent_team")
        gp = agg["games_played"].replace({0: pd.NA})
        agg["projected_rushing_yards"] = (agg["rushing_yards"] / gp).astype(float)
        agg["projected_rushing_tds"] = (agg["rushing_tds"] / gp).astype(float)
        agg["projected_carries"] = (agg["carries"] / gp).astype(float)
        agg["projected_receiving_yards"] = (agg["receiving_yards"] / gp).astype(float)
        agg["projected_receiving_tds"] = (agg["receiving_tds"] / gp).astype(float)
        agg["projected_receptions"] = (agg["receptions"] / gp).astype(float)
        agg["projected_targets"] = (agg["targets"] / gp).astype(float)
        agg["projected_total_yards"] = agg["projected_rushing_yards"].fillna(0) + agg["projected_receiving_yards"].fillna(0)
        agg["projected_total_tds"] = agg["projected_rushing_tds"].fillna(0) + agg["projected_receiving_tds"].fillna(0)

        for _, game in games.iterrows():
            game_id = game.get("game_id")
            for side in ("home", "away"):
                team = str(game.get(f"{side}_team", "")).upper()
                if not team:
                    continue
                players = agg[agg["recent_team"] == team].copy()
                if players.empty:
                    continue
                players = players.sort_values(
                    ["projected_total_yards", "projected_total_tds", "projected_receptions"],
                    ascending=[False, False, False]
                ).head(10)
                for rank, row in enumerate(players.itertuples(index=False), start=1):
                    results.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": game_id,
                            "team": team,
                            "team_side": side,
                            "player_rank": rank,
                            "player_id": row.player_id,
                            "player_name": row.player_name,
                            "jersey_number": row.jersey_number,
                            "games_sampled": row.games_played,
                                            "projected_rushing_yards": None if pd.isna(row.projected_rushing_yards) else round(float(row.projected_rushing_yards), 1),
                            "projected_rushing_tds": None if pd.isna(row.projected_rushing_tds) else round(float(row.projected_rushing_tds), 2),
                            "projected_carries": None if pd.isna(row.projected_carries) else round(float(row.projected_carries), 2),
                            "projected_receiving_yards": None if pd.isna(row.projected_receiving_yards) else round(float(row.projected_receiving_yards), 1),
                            "projected_receiving_tds": None if pd.isna(row.projected_receiving_tds) else round(float(row.projected_receiving_tds), 2),
                            "projected_receptions": None if pd.isna(row.projected_receptions) else round(float(row.projected_receptions), 2),
                            "projected_targets": None if pd.isna(row.projected_targets) else round(float(row.projected_targets), 2),
                            "projected_total_yards": None if pd.isna(row.projected_total_yards) else round(float(row.projected_total_yards), 1),
                            "projected_total_tds": None if pd.isna(row.projected_total_tds) else round(float(row.projected_total_tds), 2),
                        }
                    )
    if not results:
        return None
    df_off = pd.DataFrame(results)
    df_off.to_csv(save_path, index=False)
    if debug:
        print(f"[predict][players] saved offensive projections -> {save_path} (rows={len(df_off)})")
    return df_off


def _select_pbp_window(pbp: pd.DataFrame, season: int, week: Optional[int]) -> pd.DataFrame:
    current = pd.DataFrame()
    if week is not None:
        try:
            week_int = int(week)
        except Exception:
            week_int = None
        if week_int is not None:
            current = pbp[(pbp["season"] == season) & (pbp["week"] < week_int)].copy()
    else:
        current = pbp[pbp["season"] == season].copy()

    history = pbp[pbp["season"] < season].copy()
    subset = pd.concat([history, current], ignore_index=True)
    return subset


def _player_defense_predictions(
    pred_games: pd.DataFrame,
    save_path: Optional[str],
    roster: pd.DataFrame,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    if not save_path:
        return None
    if pred_games is None or pred_games.empty:
        return None

    pbp_path = RAW_DIR / "nfl_pbp.parquet"
    if not pbp_path.exists():
        if debug:
            print("[predict][players] PBP file missing; cannot build defensive projections")
        return None

    pbp = pd.read_parquet(pbp_path)
    if pbp.empty:
        return None

    pbp = pbp.copy()
    pbp["season"] = pd.to_numeric(pbp.get("season"), errors="coerce")
    pbp["week"] = pd.to_numeric(pbp.get("week"), errors="coerce")
    pbp = pbp[pd.notna(pbp["season"]) & pd.notna(pbp["week"])]
    if pbp.empty:
        return None
    pbp["season"] = pbp["season"].astype("int64")
    pbp["week"] = pbp["week"].astype("int64")
    pbp = pbp[pbp.get("season_type", "REG") == "REG"]
    pbp["defteam"] = pbp.get("defteam", "").fillna("").astype(str).str.upper()
    pbp = pbp[pbp["defteam"] != ""]

    stats_columns = [
        "sack",
        "sack_player_id",
        "sack_player_name",
        "half_sack_1_player_id",
        "half_sack_1_player_name",
        "half_sack_2_player_id",
        "half_sack_2_player_name",
        "qb_hit_1_player_id",
        "qb_hit_1_player_name",
        "qb_hit_2_player_id",
        "qb_hit_2_player_name",
        "tackle_for_loss_1_player_id",
        "tackle_for_loss_1_player_name",
        "tackle_for_loss_2_player_id",
        "tackle_for_loss_2_player_name",
        "punt_blocked",
        "blocked_player_id",
        "blocked_player_name",
    ]
    keep_cols = ["season", "week", "defteam", "yards_gained"] + [c for c in stats_columns if c in pbp.columns]
    pbp = pbp[keep_cols]

    results: List[Dict[str, Any]] = []
    grouped = pred_games.groupby(["season", "week"])
    for (season, week), games in grouped:
        subset = _select_pbp_window(pbp, int(season), int(week))
        if subset.empty:
            continue

        player_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _update(
            player_id: Optional[str],
            player_name: Optional[str],
            team: str,
            stat: str,
            value: float,
            loss: float,
            game_key: Tuple[int, int],
        ) -> None:
            if not player_id and not player_name:
                return
            pid = str(player_id) if player_id not in (None, "", "nan") else None
            pname = str(player_name) if player_name not in (None, "", "nan") else None
            key = pid or pname
            if key is None:
                return
            entry_key = (team, key)
            entry = player_stats.setdefault(
                entry_key,
                {
                    "team": team,
                    "player_id": pid,
                    "player_name": pname,
                    "games": set(),
                    "sacks": 0.0,
                    "qb_hits": 0.0,
                    "tfl": 0.0,
                    "blocked_punts": 0.0,
                    "loss_yards": 0.0,
                },
            )
            if pid and not entry.get("player_id"):
                entry["player_id"] = pid
            if pname and not entry.get("player_name"):
                entry["player_name"] = pname
            entry["games"].add(game_key)
            if stat:
                entry[stat] += value
            if loss:
                entry["loss_yards"] += loss

        for row in subset.itertuples(index=False):
            team = getattr(row, "defteam", "")
            if not team:
                continue
            game_key = (getattr(row, "season"), getattr(row, "week"))
            yards_gained = getattr(row, "yards_gained", 0) or 0
            loss_yards = -yards_gained if yards_gained < 0 else 0.0

            if getattr(row, "sack", 0):
                _update(getattr(row, "sack_player_id", None), getattr(row, "sack_player_name", None), team, "sacks", 1.0, loss_yards, game_key)
            if getattr(row, "half_sack_1_player_id", None):
                _update(getattr(row, "half_sack_1_player_id", None), getattr(row, "half_sack_1_player_name", None), team, "sacks", 0.5, loss_yards / 2 if loss_yards else 0.0, game_key)
            if getattr(row, "half_sack_2_player_id", None):
                _update(getattr(row, "half_sack_2_player_id", None), getattr(row, "half_sack_2_player_name", None), team, "sacks", 0.5, loss_yards / 2 if loss_yards else 0.0, game_key)
            if getattr(row, "qb_hit_1_player_id", None):
                _update(getattr(row, "qb_hit_1_player_id", None), getattr(row, "qb_hit_1_player_name", None), team, "qb_hits", 1.0, 0.0, game_key)
            if getattr(row, "qb_hit_2_player_id", None):
                _update(getattr(row, "qb_hit_2_player_id", None), getattr(row, "qb_hit_2_player_name", None), team, "qb_hits", 1.0, 0.0, game_key)
            if getattr(row, "tackle_for_loss_1_player_id", None):
                _update(getattr(row, "tackle_for_loss_1_player_id", None), getattr(row, "tackle_for_loss_1_player_name", None), team, "tfl", 1.0, loss_yards, game_key)
            if getattr(row, "tackle_for_loss_2_player_id", None):
                _update(getattr(row, "tackle_for_loss_2_player_id", None), getattr(row, "tackle_for_loss_2_player_name", None), team, "tfl", 1.0, loss_yards, game_key)
            if getattr(row, "punt_blocked", 0) and getattr(row, "blocked_player_id", None):
                _update(getattr(row, "blocked_player_id", None), getattr(row, "blocked_player_name", None), team, "blocked_punts", 1.0, 0.0, game_key)

        if not player_stats:
            continue

        records = []
        for entry in player_stats.values():
            games_played = len(entry["games"]) or 1
            records.append(
                {
                    "player_id": entry.get("player_id"),
                    "player_name": entry.get("player_name"),
                    "team": entry.get("team"),
                    "games_played": games_played,
                    "projected_sacks": entry["sacks"] / games_played,
                    "projected_qb_hits": entry["qb_hits"] / games_played,
                    "projected_tfl": entry["tfl"] / games_played,
                    "projected_blocked_punts": entry["blocked_punts"] / games_played,
                    "projected_loss_yards": entry["loss_yards"] / games_played,
                                    }
            )

        agg = pd.DataFrame(records)
        agg["season"] = season
        agg = _attach_roster_info(agg, roster, int(season), team_col="team", fill_name=True)

        for _, game in games.iterrows():
            game_id = game.get("game_id")
            for side in ("home", "away"):
                team = str(game.get(f"{side}_team", "")).upper()
                if not team:
                    continue
                players = agg[agg["team"] == team].copy()
                if players.empty:
                    continue
                players = players.sort_values(
                    ["projected_sacks", "projected_qb_hits", "projected_tfl"],
                    ascending=[False, False, False]
                ).head(10)
                for rank, row in enumerate(players.itertuples(index=False), start=1):
                    def _round(val: float) -> Optional[float]:
                        return None if pd.isna(val) else round(float(val), 2)

                    results.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": game_id,
                            "team": team,
                            "team_side": side,
                            "player_rank": rank,
                            "player_id": row.player_id,
                            "player_name": row.player_name,
                            "jersey_number": row.jersey_number,
                            "games_sampled": row.games_played,
                                            "projected_sacks": _round(row.projected_sacks),
                            "projected_qb_hits": _round(row.projected_qb_hits),
                            "projected_tfl": _round(row.projected_tfl),
                            "projected_blocked_punts": _round(row.projected_blocked_punts),
                            "projected_loss_yards": _round(row.projected_loss_yards),
                        }
                    )
    if not results:
        return None
    df_def = pd.DataFrame(results)
    df_def.to_csv(save_path, index=False)
    if debug:
        print(f"[predict][players] saved defensive projections -> {save_path} (rows={len(df_def)})")
    return df_def

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=False, help="Season year to predict. Defaults to latest season in data.")
    ap.add_argument("--week", type=str, default="auto", help="Week number or 'auto' to select current/next regular-season week for the chosen season.")
    ap.add_argument("--save", type=str, default="predictions.csv")
    ap.add_argument("--dump-all", dest="dump_all", type=str, default=None, help="Optional path to save the full, uncurated predictions CSV")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    ap.add_argument("--save-players-qb", dest="save_players_qb", type=str, default="predictions_players_qb.csv", help="Path to save quarterback projections; set empty to skip")
    ap.add_argument("--save-players-offense", dest="save_players_offense", type=str, default="predictions_players_offense.csv", help="Path to save offensive skill-player projections (top 10 per team); set empty to skip")
    ap.add_argument("--save-players-defense", dest="save_players_defense", type=str, default="predictions_players_defense.csv", help="Path to save defensive projections; set empty to skip")
    args = ap.parse_args()

    configure_logging(args.debug)

    sched = pd.read_parquet(RAW_DIR / "espn_schedule.parquet")
    feats = pd.read_parquet(PROC_DIR / "matchup_features.parquet")
    # Choose season: prefer arg, else latest season present in schedule
    if args.season is not None:
        season = int(args.season)
    else:
        season = int(sched["season"].max())

    # Restrict to the chosen season and regular season (season_type==2) where possible
    sched_season = sched[sched["season"] == season].copy()
    if "season_type" in sched_season.columns:
        sched_regular = sched_season[sched_season["season_type"] == 2].copy()
    else:
        sched_regular = sched_season

    # Determine target week
    if args.week == "auto":
        # If dates available, try to find the smallest week that is upcoming or the max completed week
        week_candidates = sched_regular["week"].dropna().astype(int)
        if "date" in sched_regular.columns:
            # Parse dates and select the nearest upcoming or current week
            # Use timezone-aware UTC timestamp to match parsed schedule datetimes
            now = pd.Timestamp.now(tz="UTC")
            sched_regular["_dt"] = pd.to_datetime(sched_regular["date"], errors="coerce", utc=True)
            # Prefer next upcoming; if none, fall back to max week in this season
            upcoming_weeks = sched_regular.loc[sched_regular["_dt"] >= now, "week"].astype("Int64")
            if upcoming_weeks.notna().any():
                week = int(upcoming_weeks.min())
            else:
                week = int(week_candidates.max())
            sched_regular.drop(columns=["_dt"], inplace=True, errors="ignore")
        else:
            week = int(week_candidates.max())
    else:
        week = int(args.week)

    # Build the feature slice for that season/week and compute *_diff features
    this_week = feats[(feats["season"] == season) & (feats["week"] == week)].copy()
    this_week = _make_feature_diffs(this_week)
    # Deduplicate columns to avoid duplicate-name DataFrame selections downstream
    try:
        this_week_nodup = this_week.loc[:, ~pd.Index(this_week.columns).duplicated()]
    except Exception:
        this_week_nodup = this_week
    # Load models if present
    preds = this_week[["season","week","home_team","away_team"]].copy()
    # Optionally add game identifiers and date if available
    date_col = next((c for c in ["date", "start_date"] if c in this_week.columns), None)
    if date_col is not None and date_col not in preds.columns:
        preds["date"] = this_week[date_col]
    if "game_id" in this_week.columns and "game_id" not in preds.columns:
        preds["game_id"] = this_week["game_id"]
    if "game_uid" in this_week.columns and "game_uid" not in preds.columns:
        preds["game_uid"] = this_week["game_uid"]

    # Include weather fields if available (kickoff, roof, and all weather_* columns)
    try:
        wx_cols = [
            c for c in this_week.columns
            if (isinstance(c, str) and (c.startswith("weather_") or c in ("kickoff", "roof", "venue_lat", "venue_lon", "roof_is_dome")))
        ]
        if wx_cols:
            preds = pd.concat([preds, this_week[wx_cols].copy()], axis=1)
    except Exception:
        pass

    try:
        win_art = joblib.load(Path("models/winprob_gb.pkl"))
        win_model = win_art.get("model", win_art)
        win_feats = win_art.get("features", [])
        feat_ranges = win_art.get("feature_ranges", {})
        use_cols = [c for c in win_feats if c in this_week_nodup.columns]
        if use_cols:
            Xw = this_week_nodup[use_cols].copy()
            # Clip to training ranges if provided
            if isinstance(feat_ranges, dict) and feat_ranges:
                for c in use_cols:
                    if c in feat_ranges:
                        lo, hi = feat_ranges[c]
                        Xw[c] = pd.to_numeric(Xw[c], errors="coerce").clip(lower=lo, upper=hi)
            proba_vec = win_model.predict_proba(Xw.values)[:, 1]
            preds["home_win_prob"] = proba_vec

            # Per-game explanation using SHAP (best-effort)
            try:
                import shap  # type: ignore
                # Use underlying estimator if model is calibrated
                base_est = win_model
                try:
                    cc = getattr(win_model, "calibrated_classifiers_", None)
                    if isinstance(cc, list) and len(cc) > 0 and hasattr(cc[0], "estimator"):
                        base_est = cc[0].estimator
                except Exception:
                    pass
                explainer = shap.TreeExplainer(base_est)
                shap_vals = explainer.shap_values(Xw)
                # shap can return list for classifiers; pick positive class if list
                if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                    shap_mat = shap_vals[1]
                else:
                    shap_mat = shap_vals
                # Build supporting/opposing metrics strings
                supp_list = []
                opp_list = []
                for i in range(len(Xw)):
                    row = Xw.iloc[i]
                    sv = shap_mat[i]
                    pairs = list(zip(use_cols, row.values, sv))
                    # sort by absolute contribution
                    pairs.sort(key=lambda t: abs(float(t[2]) if t[2] is not None else 0.0), reverse=True)
                    support = [f"{k}={v:.3g} (Δ={float(s):+.3g})" for k, v, s in pairs if float(s) > 0][:6]
                    oppose = [f"{k}={v:.3g} (Δ={float(s):+.3g})" for k, v, s in pairs if float(s) < 0][:6]
                    supp_list.append("; ".join(support))
                    opp_list.append("; ".join(oppose))
                preds["decision_supporting_metrics"] = supp_list
                preds["decision_opposing_metrics"] = opp_list
                # Confidence explanation
                conf_expl = []
                for p in proba_vec:
                    if pd.isna(p):
                        conf_expl.append("confidence: N/A")
                    elif p >= 0.5:
                        conf_expl.append(f"confidence: {p:.1%} home")
                    else:
                        conf_expl.append(f"confidence: {(1-p):.1%} away")
                preds["decision_confidence_expl"] = conf_expl
            except Exception:
                # Fallback: simple feature-importance based explanation
                try:
                    import numpy as _np  # lightweight
                    # Use base estimator's importances when calibrated
                    base_est = win_model
                    try:
                        cc = getattr(win_model, "calibrated_classifiers_", None)
                        if isinstance(cc, list) and len(cc) > 0 and hasattr(cc[0], "estimator"):
                            base_est = cc[0].estimator
                    except Exception:
                        pass
                    imp = getattr(base_est, "feature_importances_", None)
                    if imp is not None and len(imp) == len(use_cols):
                        imp = _np.array(imp)
                        imp = imp / (imp.sum() + 1e-12)
                        supp_list = []
                        opp_list = []
                        for i in range(len(Xw)):
                            row = Xw.iloc[i].values
                            contrib = row * imp  # sign via feature value; rough heuristic
                            pairs = list(zip(use_cols, row, contrib))
                            pairs.sort(key=lambda t: abs(float(t[2])), reverse=True)
                            support = [f"{k}={v:.3g} (w={float(c):+.3g})" for k, v, c in pairs if float(c) > 0][:6]
                            oppose = [f"{k}={v:.3g} (w={float(c):+.3g})" for k, v, c in pairs if float(c) < 0][:6]
                            supp_list.append("; ".join(support))
                            opp_list.append("; ".join(oppose))
                        preds["decision_supporting_metrics"] = supp_list
                        preds["decision_opposing_metrics"] = opp_list
                        preds["decision_confidence_expl"] = [f"confidence: {p:.1%}" if pd.notna(p) else "confidence: N/A" for p in proba_vec]
                except Exception:
                    # Final fallback: use top-|diff| features to summarize
                    try:
                        supp_list = []
                        opp_list = []
                        for i in range(len(Xw)):
                            row = Xw.iloc[i]
                            pairs = list(zip(use_cols, row.values))
                            # sort by absolute value
                            pairs.sort(key=lambda t: abs(float(t[1]) if t[1] is not None else 0.0), reverse=True)
                            # Treat positive diffs as "home-leaning", negative as "away-leaning"
                            support = [f"{k}={float(v):+.3g}" for k, v in pairs if float(v) > 0][:6]
                            oppose = [f"{k}={float(v):+.3g}" for k, v in pairs if float(v) < 0][:6]
                            supp_list.append("; ".join(support))
                            opp_list.append("; ".join(oppose))
                        preds["decision_supporting_metrics"] = supp_list
                        preds["decision_opposing_metrics"] = opp_list
                        preds["decision_confidence_expl"] = [
                            (f"confidence: {p:.1%} home" if pd.notna(p) and float(p) >= 0.5 else (f"confidence: {(1-float(p)):.1%} away" if pd.notna(p) else "confidence: N/A"))
                            for p in proba_vec
                        ]
                    except Exception:
                        pass
        else:
            preds["home_win_prob"] = None
    except Exception:
        preds["home_win_prob"] = None

    try:
        spread_art = joblib.load(Path("models/spread_gb.pkl"))
        spread_model = spread_art.get("model", spread_art)
        spread_feats = spread_art.get("features", [])
        use_cols = [c for c in spread_feats if c in this_week_nodup.columns]
        if use_cols:
            preds["pred_home_margin"] = spread_model.predict(this_week_nodup[use_cols].values)
        else:
            preds["pred_home_margin"] = None
    except Exception:
        preds["pred_home_margin"] = None

    # Build human-readable explanations for the two prediction values
    def _fmt_prob(p: float, home: str) -> str:
        if p is None or pd.isna(p):
            return "Win probability unavailable"
        return f"{home} win chance: {p:.1%} (0.5≈coin flip)"

    def _fmt_margin(m: float) -> str:
        if m is None or pd.isna(m):
            return "Predicted margin unavailable"
        if abs(m) < 0.25:
            return "Pick'em (≈0 pts)"
        side = "Home" if m >= 0 else "Away"
        return f"{side} by {abs(m):.1f} pts"

    preds["home_win_prob_expl"] = [
        _fmt_prob(p, h) for p, h in zip(preds["home_win_prob"], preds["home_team"]) 
    ]
    preds["pred_home_margin_expl"] = [
        _fmt_margin(m) for m in preds["pred_home_margin"]
    ]

    # Additional helpful columns
    def _prob_to_american(p: float | None) -> str | None:
        if p is None or pd.isna(p):
            return None
        p = float(p)
        if p <= 0 or p >= 1:
            return None
        # American odds for home side
        if p >= 0.5:
            val = -int(round(100 * p / (1 - p)))
        else:
            val = int(round(100 * (1 - p) / p))
        sign = "+" if val > 0 else ""
        return f"{sign}{val}"

    def _prob_to_decimal(p: float | None) -> float | None:
        if p is None or pd.isna(p):
            return None
        p = float(p)
        if p <= 0:
            return None
        return round(1.0 / p, 3)

    # Pick the team label instead of HOME/AWAY for readability
    preds["home_pick"] = [
        h if (pd.notna(p) and float(p) >= 0.5) else (a if pd.notna(p) else None)
        for p, h, a in zip(preds["home_win_prob"], preds["home_team"], preds["away_team"]) 
    ]
    preds["home_confidence_pct"] = preds["home_win_prob"].apply(lambda p: round(float(p) * 100, 1) if pd.notna(p) else None)
    preds["home_odds_american"] = preds["home_win_prob"].apply(_prob_to_american)
    preds["home_odds_decimal"] = preds["home_win_prob"].apply(_prob_to_decimal)
    # Use the chosen team's win probability in the explanation
    def _pick_prob(p_home: float | None, pick: str | None, h: str | None, a: str | None) -> float | None:
        if p_home is None or pd.isna(p_home) or pick is None:
            return None
        p_home = float(p_home)
        return (p_home if pick == h else (1.0 - p_home))

    preds["pick_expl"] = [
        (f"Pick {pick}: {pp*100:.1f}% win prob; margin {m:+.1f} pts" if (pp is not None and pd.notna(m))
         else (f"Pick {pick}: {pp*100:.1f}% win prob" if pp is not None else None))
        for pick, pp, m in (
            (pick, _pick_prob(prob, pick, h, a), m)
            for pick, prob, h, a, m in zip(
                preds["home_pick"], preds["home_win_prob"], preds["home_team"], preds["away_team"], preds["pred_home_margin"]
            )
        )
    ]

    # Plain-text explanation block per game (keeps SHAP columns as well)
    def _explain_row(row: pd.Series) -> str:
        pick = row.get("home_pick")
        p_home = row.get("home_win_prob")
        margin = row.get("pred_home_margin")
        home = row.get("home_team")
        away = row.get("away_team")
        # picked team probability
        if pd.isna(p_home) or pick is None:
            p_pick = None
        else:
            p_pick = float(p_home) if pick == home else float(1 - p_home)
        # SHAP-based details if present
        supp = row.get("decision_supporting_metrics") or ""
        opp = row.get("decision_opposing_metrics") or ""
        # If missing, compute a fallback from top-magnitude *_diff features in this row
        if not supp and not opp:
            try:
                diff_cols = [c for c in row.index if isinstance(c, str) and c.endswith("_diff")]
                pairs = []
                for c in diff_cols:
                    try:
                        val = row[c]
                        if pd.isna(val):
                            continue
                        v = float(val)
                        pairs.append((c, v))
                    except Exception:
                        continue
                pairs.sort(key=lambda t: abs(t[1]), reverse=True)
                # Directional support based on pick
                if pick == home:
                    support_pairs = [(k, v) for k, v in pairs if v > 0][:6]
                    oppose_pairs = [(k, v) for k, v in pairs if v < 0][:6]
                elif pick == away:
                    support_pairs = [(k, v) for k, v in pairs if v < 0][:6]
                    oppose_pairs = [(k, v) for k, v in pairs if v > 0][:6]
                else:
                    support_pairs = pairs[:6]
                    oppose_pairs = pairs[6:12]
                supp = "; ".join([f"{k}={v:+.3g}" for k, v in support_pairs])
                opp = "; ".join([f"{k}={v:+.3g}" for k, v in oppose_pairs])
            except Exception:
                pass
        conf = row.get("decision_confidence_expl") or ""
        parts = []
        if pick and p_pick is not None:
            parts.append(f"Pick {pick}: {p_pick:.1%} win prob")
        elif pick:
            parts.append(f"Pick {pick}")
        if pd.notna(margin):
            parts.append(f"margin {float(margin):+.1f} pts")
        if conf:
            parts.append(str(conf))
        if supp:
            parts.append(f"support: {supp}")
        if opp:
            parts.append(f"oppose: {opp}")
        return " | ".join(parts)

    try:
        preds["decision_explanation"] = preds.apply(_explain_row, axis=1)
    except Exception:
        pass

    # Narrative explanation per game
    def _fmt_list(items):
        return ", ".join(items) if items else "none"

    def _narrative_row(row: pd.Series) -> str:
        home = str(row.get("home_team"))
        away = str(row.get("away_team"))
        pick = row.get("home_pick")
        p_home = row.get("home_win_prob")
        prob_pick = None
        if pd.notna(p_home) and pick:
            prob_pick = float(p_home) if pick == home else float(1 - p_home)
        margin = row.get("pred_home_margin")

        # Offense diffs (home - away): curated keys or any pass_/rush_/recv_*_diff
        curated_off_keys = [
            "pass_yds_diff", "pass_td_diff", "pass_att_diff", "rush_yds_diff", "rush_td_diff", "recv_yds_diff", "recv_td_diff"
        ]
        off_pairs = []
        for k in curated_off_keys:
            if k in row.index and pd.notna(row[k]):
                try:
                    off_pairs.append((k, float(row[k])))
                except Exception:
                    continue
        if not off_pairs:
            # scan any pass_/rush_/recv_*_diff columns
            for k in row.index:
                if isinstance(k, str) and k.endswith("_diff") and (k.startswith("pass_") or k.startswith("rush_") or k.startswith("recv_")):
                    try:
                        v = float(row[k])
                        if pd.notna(v) and abs(v) > 1e-9:
                            off_pairs.append((k, v))
                    except Exception:
                        continue
        off_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        if not off_pairs:
            # Final fallback: any *_diff column
            any_pairs = []
            for k in row.index:
                if isinstance(k, str) and k.endswith("_diff"):
                    try:
                        v = float(row[k])
                        if pd.notna(v) and abs(v) > 1e-9:
                            any_pairs.append((k, v))
                    except Exception:
                        continue
            any_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            off_pairs = any_pairs[:6]
        # Metrics that favor pick
        if pick == home:
            off_favor = [f"{k}={v:+.2g}" for k, v in off_pairs if v > 0][:3]
            off_against = [f"{k}={v:+.2g}" for k, v in off_pairs if v < 0][:3]
        else:
            off_favor = [f"{k}={v:+.2g}" for k, v in off_pairs if v < 0][:3]
            off_against = [f"{k}={v:+.2g}" for k, v in off_pairs if v > 0][:3]

        # Defense-allowed diffs (lower is better for home if diff < 0)
        def_curated = [
            "def_ypp_allowed_diff", "def_pass_ypp_allowed_diff", "def_rush_ypc_allowed_diff", "def_pass_rate_allowed_diff", "def_epa_per_play_allowed_diff"
        ]
        def_pairs = []
        for k in def_curated:
            if k in row.index and pd.notna(row[k]):
                try:
                    def_pairs.append((k, float(row[k])))
                except Exception:
                    continue
        if not def_pairs:
            for k in row.index:
                if isinstance(k, str) and k.startswith("def_") and k.endswith("_diff"):
                    try:
                        v = float(row[k])
                        if pd.notna(v) and abs(v) > 1e-9:
                            def_pairs.append((k, v))
                    except Exception:
                        continue
        def_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        if not def_pairs:
            # Final fallback: any def_*_diff if present, else any *_diff
            any_def = []
            for k in row.index:
                if isinstance(k, str) and k.startswith("def_") and k.endswith("_diff"):
                    try:
                        v = float(row[k])
                        if pd.notna(v) and abs(v) > 1e-9:
                            any_def.append((k, v))
                    except Exception:
                        continue
            if not any_def:
                for k in row.index:
                    if isinstance(k, str) and k.endswith("_diff"):
                        try:
                            v = float(row[k])
                            if pd.notna(v) and abs(v) > 1e-9:
                                any_def.append((k, v))
                        except Exception:
                            continue
            any_def.sort(key=lambda x: abs(x[1]), reverse=True)
            def_pairs = any_def[:6]
        # For home: negative diffs favorable; for away: positive diffs favorable (home worse)
        if pick == home:
            def_favor = [f"{k}={v:+.2g}" for k, v in def_pairs if v < 0][:3]
            def_against = [f"{k}={v:+.2g}" for k, v in def_pairs if v > 0][:3]
        else:
            def_favor = [f"{k}={v:+.2g}" for k, v in def_pairs if v > 0][:3]
            def_against = [f"{k}={v:+.2g}" for k, v in def_pairs if v < 0][:3]

        # Availability: starters, injuries, snaps
        starters = row.get("starters_diff")
        inj_out = row.get("inj_out_diff")
        snap_off = row.get("snaps_offense_snaps_diff")
        snap_def = row.get("snaps_defense_snaps_diff")
        avail_parts = []
        try:
            if pd.notna(starters):
                avail_parts.append(f"starters_diff={float(starters):+g}")
        except Exception:
            pass
        try:
            if pd.notna(inj_out):
                avail_parts.append(f"inj_out_diff={float(inj_out):+g}")
        except Exception:
            pass
        try:
            if pd.notna(snap_off):
                avail_parts.append(f"off_snaps_diff={float(snap_off):+g}")
        except Exception:
            pass
        try:
            if pd.notna(snap_def):
                avail_parts.append(f"def_snaps_diff={float(snap_def):+g}")
        except Exception:
            pass

        conf_str = None
        if prob_pick is not None:
            side = "home" if pick == home else "away"
            conf_str = f"confidence: {prob_pick:.1%} {side}"

        lines = []
        if pick and prob_pick is not None:
            lines.append(f"Pick {pick}: {prob_pick:.1%} win prob")
        elif pick:
            lines.append(f"Pick {pick}")
        if pd.notna(margin):
            lines.append(f"margin {float(margin):+.1f} pts")
        if conf_str:
            lines.append(conf_str)
        # Sections
        lines.append(f"Offense diffs: favor {_fmt_list(off_favor)}; oppose {_fmt_list(off_against)}")
        lines.append(f"Defense-allowed diffs: favor {_fmt_list(def_favor)}; oppose {_fmt_list(def_against)}")
        if avail_parts:
            lines.append("Availability: " + ", ".join(avail_parts))
        return " | ".join(lines)

    try:
        preds["decision_narrative"] = preds.apply(_narrative_row, axis=1)
    except Exception:
        pass

    # --- Baseline per-game stat projections from season totals (if present) ---
    # Helper to get per-game from *_home / *_away season totals
    def per_game(col_base: str):
        ch, ca = f"{col_base}_home", f"{col_base}_away"
        if ch in this_week.columns:
            preds[f"pred_home_{col_base}"] = pd.to_numeric(this_week[ch], errors="coerce") / 17.0
        else:
            preds[f"pred_home_{col_base}"] = None
        if ca in this_week.columns:
            preds[f"pred_away_{col_base}"] = pd.to_numeric(this_week[ca], errors="coerce") / 17.0
        else:
            preds[f"pred_away_{col_base}"] = None

    # Passing attempts, Rushing attempts, Touchdowns, Interceptions, First downs (receiving+rushing), 4th down
    per_game("passing_att")
    per_game("rushing_att")
    per_game("scoring_tottd")
    per_game("passing_int")
    # First downs: use downs_rec1st + downs_rush1st if available
    for side in ["home", "away"]:
        rush_col = f"downs_rush1st_{side}"
        rec_col = f"downs_rec1st_{side}"
        r1 = pd.to_numeric(this_week[rush_col], errors="coerce") if rush_col in this_week.columns else None
        e1 = pd.to_numeric(this_week[rec_col], errors="coerce") if rec_col in this_week.columns else None
        if r1 is not None or e1 is not None:
            total = (r1 if r1 is not None else 0) + (e1 if e1 is not None else 0)
            preds[f"pred_{side}_first_downs"] = total / 17.0
        else:
            preds[f"pred_{side}_first_downs"] = None
    # 4th down conversions: downs_4thmd per game (attempts also provided)
    per_game("downs_4thatt")
    per_game("downs_4thmd")

    # Punts and Field Goals are not available in current team stats scrape; set placeholders
    preds["pred_home_punts"] = None
    preds["pred_away_punts"] = None
    preds["pred_home_field_goals"] = None
    preds["pred_away_field_goals"] = None

    # Include key defensive allowed metrics and ESPN aggregates in a single concat
    copy_cols: list[str] = []
    allowed_cols = [
        "ypp_allowed", "pass_rate_allowed", "pass_ypp_allowed", "rush_ypc_allowed", "epa_per_play_allowed"
    ]
    # Ensure *_diff exists for defense metrics if components are available
    for base in allowed_cols:
        hcol = f"{base}_home"
        acol = f"{base}_away"
        dcol = f"{base}_diff"
        if hcol in this_week.columns:
            copy_cols.append(hcol)
        if acol in this_week.columns:
            copy_cols.append(acol)
        if dcol in this_week.columns:
            copy_cols.append(dcol)
        elif hcol in this_week.columns and acol in this_week.columns:
            # Compute diff on the fly for export only
            preds[dcol] = pd.to_numeric(this_week[hcol], errors="coerce") - pd.to_numeric(this_week[acol], errors="coerce")
            copy_cols.append(dcol)
    # Also dynamically include any *_allowed_* metrics that exist in features
    dyn_allowed_bases = set()
    for col in this_week.columns:
        if col.endswith("_home") and "allowed" in col and not col.startswith("espn_"):
            dyn_allowed_bases.add(col[:-5])  # strip _home
    for base in sorted(dyn_allowed_bases):
        for suffix in ("_home", "_away"):
            col = base + suffix
            if col in this_week.columns:
                copy_cols.append(col)
        dcol = base + "_diff"
        if dcol in this_week.columns:
            copy_cols.append(dcol)
        elif (base + "_home") in this_week.columns and (base + "_away") in this_week.columns:
            preds[dcol] = pd.to_numeric(this_week[base + "_home"], errors="coerce") - pd.to_numeric(this_week[base + "_away"], errors="coerce")

    # Curated ESPN metrics only
    curated_bases = [
        "espn_passing_att", "espn_passing_yds", "espn_passing_td", "espn_passing_int",
        "espn_rushing_att", "espn_rushing_yds", "espn_rushing_td",
        "espn_receiving_rec", "espn_receiving_tgts", "espn_receiving_yds", "espn_receiving_td",
    ]
    for base in curated_bases:
        for suffix in ("_home", "_away"):
            col = base + suffix
            if col in this_week.columns:
                copy_cols.append(col)
        dcol = base + "_diff"
        if dcol in this_week.columns:
            copy_cols.append(dcol)
    # Deduplicate while preserving order
    if copy_cols:
        seen = set()
        ordered_cols = [c for c in copy_cols if not (c in seen or seen.add(c))]
        existing = [c for c in ordered_cols if c in this_week.columns]
        if existing:
            preds = pd.concat([preds, this_week[existing].copy()], axis=1)
    # If no explicit '*_allowed_*' metrics exist, derive a few from opponent offense
    # def_ypp_allowed_* from opponent yards_per_play_*, def_pass_rate_allowed_* from opponent pass_rate_*
    if ("def_ypp_allowed_home" not in preds.columns) and ("yards_per_play_home" in this_week.columns and "yards_per_play_away" in this_week.columns):
        preds["def_ypp_allowed_home"] = this_week["yards_per_play_away"]
        preds["def_ypp_allowed_away"] = this_week["yards_per_play_home"]
        preds["def_ypp_allowed_diff"] = preds["def_ypp_allowed_home"] - preds["def_ypp_allowed_away"]
    if ("def_pass_rate_allowed_home" not in preds.columns) and ("pass_rate_home" in this_week.columns and "pass_rate_away" in this_week.columns):
        preds["def_pass_rate_allowed_home"] = this_week["pass_rate_away"]
        preds["def_pass_rate_allowed_away"] = this_week["pass_rate_home"]
        preds["def_pass_rate_allowed_diff"] = preds["def_pass_rate_allowed_home"] - preds["def_pass_rate_allowed_away"]
    # def_pass_ypp_allowed from passing yards per attempt if available
    if ("def_pass_ypp_allowed_home" not in preds.columns) and (
        "passing_yds_per_att_home" in this_week.columns and "passing_yds_per_att_away" in this_week.columns
    ):
        preds["def_pass_ypp_allowed_home"] = this_week["passing_yds_per_att_away"]
        preds["def_pass_ypp_allowed_away"] = this_week["passing_yds_per_att_home"]
        preds["def_pass_ypp_allowed_diff"] = preds["def_pass_ypp_allowed_home"] - preds["def_pass_ypp_allowed_away"]
    # def_rush_ypc_allowed from rushing_ypc if available
    if ("def_rush_ypc_allowed_home" not in preds.columns) and (
        "rushing_ypc_home" in this_week.columns and "rushing_ypc_away" in this_week.columns
    ):
        preds["def_rush_ypc_allowed_home"] = this_week["rushing_ypc_away"]
        preds["def_rush_ypc_allowed_away"] = this_week["rushing_ypc_home"]
        preds["def_rush_ypc_allowed_diff"] = preds["def_rush_ypc_allowed_home"] - preds["def_rush_ypc_allowed_away"]
    # def_epa_per_play_allowed if offense EPA per play is present
    if ("def_epa_per_play_allowed_home" not in preds.columns) and (
        "epa_per_play_home" in this_week.columns and "epa_per_play_away" in this_week.columns
    ):
        preds["def_epa_per_play_allowed_home"] = this_week["epa_per_play_away"]
        preds["def_epa_per_play_allowed_away"] = this_week["epa_per_play_home"]
        preds["def_epa_per_play_allowed_diff"] = preds["def_epa_per_play_allowed_home"] - preds["def_epa_per_play_allowed_away"]
    # Allowed totals from opponent offense if present
    def _flip_total(name: str):
        h, a, d = f"def_{name}_allowed_home", f"def_{name}_allowed_away", f"def_{name}_allowed_diff"
        oh, oa = f"{name}_home", f"{name}_away"
        if oh in this_week.columns and oa in this_week.columns:
            preds[h] = this_week[oa]
            preds[a] = this_week[oh]
            preds[d] = preds[h] - preds[a]

    # --- Build offense season totals from NFL.com proxies when PBP not available ---
    # plays ≈ downs_scrmplys; pass_plays ≈ pass_att; rush_plays ≈ rush_att
    # yards ≈ pass_yds + rush_yds; pass_yards ≈ pass_yds; rush_yards ≈ rush_yds
    def _ensure_offense_totals(prefix: str):
        # prefix is 'home' or 'away'
        scr = f"downs_scrmplys_{prefix}"
        pa = f"pass_att_{prefix}" if f"pass_att_{prefix}" in preds.columns else f"passing_att_{prefix}"
        ra = f"rush_att_{prefix}" if f"rush_att_{prefix}" in preds.columns else f"rushing_att_{prefix}"
        py = f"pass_yds_{prefix}" if f"pass_yds_{prefix}" in preds.columns else f"passing_passyds_{prefix}"
        ry = f"rush_yds_{prefix}" if f"rush_yds_{prefix}" in preds.columns else f"rushing_rushyds_{prefix}"

        # create on this_week so _flip_total can find them
        if f"plays_{prefix}" not in this_week.columns and scr in this_week.columns:
            this_week[f"plays_{prefix}"] = pd.to_numeric(this_week[scr], errors="coerce")
        if f"pass_plays_{prefix}" not in this_week.columns and pa in this_week.columns:
            this_week[f"pass_plays_{prefix}"] = pd.to_numeric(this_week[pa], errors="coerce")
        if f"rush_plays_{prefix}" not in this_week.columns and ra in this_week.columns:
            this_week[f"rush_plays_{prefix}"] = pd.to_numeric(this_week[ra], errors="coerce")
        # yards
        if py in this_week.columns and ry in this_week.columns and f"yards_{prefix}" not in this_week.columns:
            this_week[f"pass_yards_{prefix}"] = pd.to_numeric(this_week[py], errors="coerce")
            this_week[f"rush_yards_{prefix}"] = pd.to_numeric(this_week[ry], errors="coerce")
            this_week[f"yards_{prefix}"] = this_week[f"pass_yards_{prefix}"] + this_week[f"rush_yards_{prefix}"]
        else:
            # still ensure component naming for flip
            if py in this_week.columns and f"pass_yards_{prefix}" not in this_week.columns:
                this_week[f"pass_yards_{prefix}"] = pd.to_numeric(this_week[py], errors="coerce")
            if ry in this_week.columns and f"rush_yards_{prefix}" not in this_week.columns:
                this_week[f"rush_yards_{prefix}"] = pd.to_numeric(this_week[ry], errors="coerce")

    try:
        _ensure_offense_totals("home")
        _ensure_offense_totals("away")
    except Exception:
        pass

    for base in [
        "plays", "pass_plays", "rush_plays", "yards", "pass_yards", "rush_yards", "epa"
    ]:
        _flip_total(base)
    # Diagnostic: list available '*allowed*' columns from features
    try:
        allowed_in_feat = [c for c in this_week.columns if "allowed" in c]
        print(f"[predict] features allowed cols: {len(allowed_in_feat)} -> {sorted(allowed_in_feat)[:30]}")
    except Exception:
        pass

    # Friendly aliases for curated ESPN metrics
    alias_bases = {
        "espn_passing_att": "pass_att",
        "espn_passing_yds": "pass_yds",
        "espn_passing_td": "pass_td",
        "espn_passing_int": "pass_int",
        "espn_rushing_att": "rush_att",
        "espn_rushing_yds": "rush_yds",
        "espn_rushing_td": "rush_td",
        "espn_receiving_rec": "recv_rec",
        "espn_receiving_tgts": "recv_tgt",
        "espn_receiving_yds": "recv_yds",
        "espn_receiving_td": "recv_td",
    }
    rename_map = {}
    for src_base, dst_base in alias_bases.items():
        for suffix in ("_home", "_away", "_diff"):
            src = src_base + suffix
            dst = dst_base + suffix
            rename_map[src] = dst
    preds.rename(columns=rename_map, inplace=True)

    # Backfill curated offense metrics from NFL.com team stats if missing
    backfill_map = {
        "pass_att": "passing_att",
        "pass_yds": "passing_passyds",
        "pass_td": "passing_td",
        "pass_int": "passing_int",
        "rush_att": "rushing_att",
        "rush_yds": "rushing_rushyds",
        "rush_td": "rushing_td",
        # Receiving totals by team may be inconsistent; we skip backfill if not present reliably
        "recv_rec": None,
        "recv_tgt": None,
        "recv_yds": "receiving_yds",
        "recv_td": "receiving_td",
    }
    for friendly, src_base in backfill_map.items():
        if not src_base:
            continue
        for suffix in ("_home", "_away"):
            tgt = friendly + suffix
            src = src_base + suffix
            if tgt not in preds.columns and src in this_week.columns:
                preds[tgt] = pd.to_numeric(this_week[src], errors="coerce")
        # diff
        tgt_d = friendly + "_diff"
        if tgt_d not in preds.columns:
            h, a = friendly + "_home", friendly + "_away"
            if h in preds.columns and a in preds.columns:
                preds[tgt_d] = pd.to_numeric(preds[h], errors="coerce") - pd.to_numeric(preds[a], errors="coerce")

    # Friendly aliases for defense-allowed metrics
    def_alias = {
        "ypp_allowed": "def_ypp_allowed",
        "pass_rate_allowed": "def_pass_rate_allowed",
        "pass_ypp_allowed": "def_pass_ypp_allowed",
        "rush_ypc_allowed": "def_rush_ypc_allowed",
        "epa_per_play_allowed": "def_epa_per_play_allowed",
    }
    def_rename = {}
    for src_base, dst_base in def_alias.items():
        for suffix in ("_home", "_away", "_diff"):
            src = src_base + suffix
            dst = dst_base + suffix
            if src in preds.columns:
                def_rename[src] = dst
    preds.rename(columns=def_rename, inplace=True)

    # Prefix dynamically-discovered allowed metrics with def_ if not already prefixed
    for c in list(preds.columns):
        if (c.endswith(("_home", "_away", "_diff")) and "allowed" in c and not c.startswith(("def_", "espn_"))):
            preds.rename(columns={c: "def_" + c}, inplace=True)

    # Optional: quick debug summary to help verify columns (after renaming)
    try:
        def_found = [c for c in preds.columns if c.startswith("def_")]
        espn_found = [c for c in preds.columns if c.startswith(("pass_", "rush_", "recv_"))]
        print(f"[predict] exported defense cols: {len(def_found)}  espn cols: {len(espn_found)}")
    except Exception:
        pass

    # Recompute narrative now that curated offense/defense columns are appended
    try:
        preds["decision_narrative"] = preds.apply(_narrative_row, axis=1)
    except Exception:
        pass

    # Convert any UTC datetime columns to MST (America/Denver) for readability
    def _to_mst_str(s: pd.Series) -> pd.Series:
        try:
            dt = pd.to_datetime(s, errors="coerce", utc=True)
            dt_mst = dt.dt.tz_convert("America/Denver")
            return dt_mst.dt.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            return s

    for col in ["date", "start_date"]:
        if col in preds.columns:
            preds[col] = _to_mst_str(preds[col])

    # === Weekly availability metrics ===
    # Injuries: inj_out, inj_doubtful, inj_questionable (from build_features aggregation)
    for base in ["inj_out", "inj_doubtful", "inj_questionable"]:
        h, a, d = f"{base}_home", f"{base}_away", f"{base}_diff"
        if h in this_week.columns:
            preds[h] = pd.to_numeric(this_week[h], errors="coerce")
        if a in this_week.columns:
            preds[a] = pd.to_numeric(this_week[a], errors="coerce")
        if h in preds.columns and a in preds.columns:
            preds[d] = pd.to_numeric(preds[h], errors="coerce") - pd.to_numeric(preds[a], errors="coerce")

    # Depth charts: count of starters if available (depth_is_starter aggregated)
    for base in ["depth_is_starter"]:
        h, a, d = f"{base}_home", f"{base}_away", f"{base}_diff"
        if h in this_week.columns:
            preds[h] = pd.to_numeric(this_week[h], errors="coerce")
        if a in this_week.columns:
            preds[a] = pd.to_numeric(this_week[a], errors="coerce")
        if h in preds.columns and a in preds.columns:
            preds[d] = pd.to_numeric(preds[h], errors="coerce") - pd.to_numeric(preds[a], errors="coerce")
    # Friendly alias for starters
    if "depth_is_starter_home" in preds.columns:
        preds.rename(columns={
            "depth_is_starter_home": "starters_home",
            "depth_is_starter_away": "starters_away",
            "depth_is_starter_diff": "starters_diff",
        }, inplace=True)

    # Injury rates: questionable_rate = inj_questionable / (inj_out + inj_doubtful + inj_questionable)
    if "inj_questionable_home" in preds.columns:
        qh = pd.to_numeric(preds["inj_questionable_home"], errors="coerce")
        qa = pd.to_numeric(preds["inj_questionable_away"], errors="coerce")
        th = (
            pd.to_numeric(preds["inj_out_home"], errors="coerce")
            + pd.to_numeric(preds["inj_doubtful_home"], errors="coerce")
            + qh
        )
        ta = (
            pd.to_numeric(preds["inj_out_away"], errors="coerce")
            + pd.to_numeric(preds["inj_doubtful_away"], errors="coerce")
            + qa
        )
        preds["questionable_rate_home"] = (qh / th.replace(0, pd.NA)).astype(float)
        preds["questionable_rate_away"] = (qa / ta.replace(0, pd.NA)).astype(float)
        preds["questionable_rate_diff"] = preds["questionable_rate_home"] - preds["questionable_rate_away"]

    # Approximate punts and field goals if still missing using team totals (very rough placeholder)
    # punts ≈ scrimmage plays − pass_att − rush_att (clipped at 0)
    for side in ("home", "away"):
        punts_col = f"pred_{side}_punts"
        if punts_col not in preds.columns or preds[punts_col].isna().all():
            plays = pd.to_numeric(this_week[f"downs_scrmplys_{side}"], errors="coerce") if f"downs_scrmplys_{side}" in this_week.columns else None
            pa = pd.to_numeric(this_week[f"passing_att_{side}"], errors="coerce") if f"passing_att_{side}" in this_week.columns else None
            ra = pd.to_numeric(this_week[f"rushing_att_{side}"], errors="coerce") if f"rushing_att_{side}" in this_week.columns else None
            if plays is not None and pa is not None and ra is not None:
                est = (plays - pa - ra).clip(lower=0).astype(float)
                preds[punts_col] = est
    for side in ("home", "away"):
        fg_col = f"pred_{side}_field_goals"
        if fg_col not in preds.columns or preds[fg_col].isna().all():
            preds[fg_col] = 0.0  # no reliable team FG data in current sources; set to 0 baseline

    odds_columns: List[str] = []
    try:
        odds_key = _load_odds_api_key()
        if odds_key:
            odds_data = fetch_odds(
                api_key=odds_key,
                sport="americanfootball_nfl",
                regions="us",
                markets=("h2h", "spreads"),
                odds_format="american",
            )
            odds_columns = _attach_odds_spreads(preds, odds_data, debug=args.debug)
        elif args.debug:
            print("[predict][odds] skipping odds integration: ODDS_API_KEY not configured")
    except Exception as exc:
        if args.debug:
            print(f"[predict][odds] integration failed: {exc}")

    # Final cleanup: drop duplicate / empty columns so exports focus on populated predictions
    preds = _cleanup_prediction_columns(preds)

    # Optionally save full dump (consolidate decision columns: keep only pick_expl)
    if args.dump_all:
        drop_cols = [
            "decision_explanation",
            "decision_narrative",
            "decision_confidence_expl",
            "decision_supporting_metrics",
            "decision_opposing_metrics",
        ]
        dump_df = preds.drop(columns=[c for c in drop_cols if c in preds.columns], errors="ignore")
        dump_df.to_csv(args.dump_all, index=False)
        print(f"Saved full predictions to {args.dump_all} (cols={len(dump_df.columns)})")

    # Curated default output: highlight identity, weather, and actual prediction fields
    curated = [
        # identity
        "season", "week", "date", "game_id", "home_team", "away_team",
        # weather (if available)
        "kickoff", "roof", "weather_temp_f", "weather_feelslike_f", "weather_wind_mph", "weather_windgust_mph", "weather_precip_prob_pct",
        "weather_temp_kickoff_f", "weather_temp_q1_f", "weather_temp_q2_f", "weather_temp_q3_f", "weather_temp_q4_f",
        # core predictions
        "home_win_prob", "pred_home_margin", "pred_home_margin_expl", "home_pick", "home_confidence_pct", "home_odds_american", "home_odds_decimal", "pick_expl",
        # offensive production projections
        "pred_home_passing_att", "pred_away_passing_att",
        "pred_home_rushing_att", "pred_away_rushing_att",
        "pred_home_scoring_tottd", "pred_away_scoring_tottd",
        "pred_home_passing_int", "pred_away_passing_int",
        "pred_home_first_downs", "pred_away_first_downs",
        "pred_home_downs_4thatt", "pred_away_downs_4thatt",
        "pred_home_downs_4thmd", "pred_away_downs_4thmd",
        "pred_home_punts", "pred_away_punts",
        "pred_home_field_goals", "pred_away_field_goals",
        # narrative / explanation
        "decision_explanation", "decision_narrative", "decision_confidence_expl", "decision_supporting_metrics", "decision_opposing_metrics",
    ]
    if odds_columns:
        for col in odds_columns:
            if col not in curated:
                curated.append(col)
    curated_cols = [c for c in curated if c in preds.columns]
    preds[curated_cols].to_csv(args.save, index=False)
    print(f"Saved predictions to {args.save} (season={season}, week={week}, games={len(preds)}, cols={len(curated_cols)})")

    player_games = preds[["season", "week", "game_id", "home_team", "away_team"]].drop_duplicates()
    player_stats_frame = _load_player_stats_frame()
    roster_frame = _load_roster_frame()
    try:
        _player_qb_predictions(player_games, args.save_players_qb, player_stats_frame, roster_frame, debug=args.debug)
    except Exception as exc:
        if args.debug:
            print(f"[predict][players] unable to save QB projections: {exc}")
    try:
        _player_offense_predictions(player_games, args.save_players_offense, player_stats_frame, roster_frame, debug=args.debug)
    except Exception as exc:
        if args.debug:
            print(f"[predict][players] unable to save offensive projections: {exc}")
    try:
        _player_defense_predictions(player_games, args.save_players_defense, roster_frame, debug=args.debug)
    except Exception as exc:
        if args.debug:
            print(f"[predict][players] unable to save defensive projections: {exc}")

if __name__ == "__main__":
    main()
