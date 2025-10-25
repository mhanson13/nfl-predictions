from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.utils.io import RAW_DIR, PROC_DIR, write_df

logger = logging.getLogger("sportradar_transform")

RAW_SPORTRADAR_DIR = RAW_DIR / "sportradar"
PROC_SPORTRADAR_DIR = PROC_DIR / "sportradar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Sportradar raw JSON into structured tables."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_SPORTRADAR_DIR,
        help="Directory containing raw Sportradar JSON dumps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROC_SPORTRADAR_DIR,
        help="Directory to store normalized tables.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["schedule", "rosters", "game_stats", "seasonal_stats", "transactions", "change_log", "all"],
        default=["all"],
        help="Which normalized outputs to generate.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect inputs and expected outputs without writing files.",
    )
    return parser.parse_args()


def _latest_json(path: Path) -> Tuple[Optional[dict], Optional[Path]]:
    if not path.exists():
        return None, None
    files = sorted(path.glob("*.json"))
    if not files:
        return None, None
    latest = files[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8")), latest
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", latest, exc)
        return None, None


def _snapshot_year(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    ts_clean = ts.replace("Z", "") if isinstance(ts, str) else ts
    try:
        dt = datetime.fromisoformat(ts_clean)
    except ValueError:
        try:
            dt = datetime.fromisoformat(ts_clean + "+00:00")
        except ValueError:
            return None
    return dt.year

def _extract_timestamp_from_path(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    match = re.search(r"_(\d{8}T\d{6}Z)", path.name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
        return dt.replace(tzinfo=None).isoformat() + "Z"
    except ValueError:
        return None


def _write_or_log(df: pd.DataFrame, path: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] Would write %s rows to %s", len(df), path)
        return
    write_df(df, path)
    logger.info("Wrote %s rows to %s", len(df), path)


def transform_schedule(raw_root: Path, out_dir: Path, dry_run: bool) -> None:
    rows: List[Dict[str, Any]] = []
    schedule_root = raw_root / "season_schedule"
    if not schedule_root.exists():
        logger.info("No season_schedule directory at %s", schedule_root)
    for season_dir in sorted(schedule_root.glob("*")):
        if not season_dir.is_dir():
            continue
        data, source = _latest_json(season_dir)
        if not data:
            continue
        season_year = data.get("year")
        season_type = data.get("type", {}).get("code") if isinstance(data.get("type"), dict) else data.get("type")
        season_id = data.get("id")
        for week in data.get("weeks", []):
            week_seq = week.get("sequence")
            week_title = week.get("title")
            for game in week.get("games", []):
                summary = game.get("summary")
                venue = None
                if isinstance(summary, dict) and summary.get("venue"):
                    venue = summary.get("venue")
                if not isinstance(venue, dict):
                    venue = game.get("venue", {})
                broadcast = game.get("broadcast", {})
                weather = game.get("weather", {})
                home = None
                away = None
                if isinstance(summary, dict):
                    home = summary.get("home")
                    away = summary.get("away")
                if not isinstance(home, dict) or not home:
                    home = game.get("home", {})
                if not isinstance(away, dict) or not away:
                    away = game.get("away", {})
                rows.append(
                    {
                        "season_id": season_id,
                        "season_year": season_year,
                        "season_type": season_type,
                        "week": week_seq,
                        "week_title": week_title,
                        "game_id": game.get("id"),
                        "game_status": game.get("status"),
                        "scheduled": game.get("scheduled"),
                        "attendance": game.get("attendance"),
                        "entry_mode": game.get("entry_mode"),
                        "game_type": game.get("game_type"),
                        "conference_game": game.get("conference_game"),
                        "duration": game.get("duration"),
                        "venue_id": venue.get("id") if isinstance(venue, dict) else None,
                        "venue_name": venue.get("name") if isinstance(venue, dict) else None,
                        "venue_capacity": venue.get("capacity") if isinstance(venue, dict) else None,
                        "venue_city": venue.get("city") if isinstance(venue, dict) else None,
                        "venue_state": venue.get("state") if isinstance(venue, dict) else None,
                        "home_id": home.get("id") if isinstance(home, dict) else None,
                        "home_name": home.get("name") if isinstance(home, dict) else None,
                        "home_market": home.get("market") if isinstance(home, dict) else None,
                        "home_alias": home.get("alias") if isinstance(home, dict) else None,
                        "home_points": home.get("points") if isinstance(home, dict) else None,
                        "away_id": away.get("id") if isinstance(away, dict) else None,
                        "away_name": away.get("name") if isinstance(away, dict) else None,
                        "away_market": away.get("market") if isinstance(away, dict) else None,
                        "away_alias": away.get("alias") if isinstance(away, dict) else None,
                        "away_points": away.get("points") if isinstance(away, dict) else None,
                        "weather_temp_f": weather.get("temperature") if isinstance(weather, dict) else None,
                        "weather_conditions": weather.get("condition") if isinstance(weather, dict) else None,
                        "broadcast_network": broadcast.get("network") if isinstance(broadcast, dict) else None,
                        "source_path": str(source) if source else None,
                    }
                )
    df = pd.DataFrame(rows)
    output = out_dir / "schedule.parquet"
    _write_or_log(df, output, dry_run)


def transform_rosters(raw_root: Path, out_dir: Path, dry_run: bool) -> None:
    roster_root = raw_root / "team_roster"
    rows: List[Dict[str, Any]] = []
    if not roster_root.exists():
        logger.info("No team_roster directory at %s", roster_root)
    for team_dir in sorted(roster_root.glob("*")):
        if not team_dir.is_dir():
            continue
        data, source = _latest_json(team_dir)
        if not data:
            continue
        team_id = data.get("id")
        team_name = data.get("name")
        team_market = data.get("market")
        team_alias = data.get("alias")
        snapshot_ts = _extract_timestamp_from_path(source)
        for player in data.get("players", []):
            rows.append(
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "team_market": team_market,
                    "team_alias": team_alias,
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "first_name": player.get("first_name"),
                    "last_name": player.get("last_name"),
                    "jersey": player.get("jersey"),
                    "position": player.get("position"),
                    "status": player.get("status"),
                    "birth_date": player.get("birth_date"),
                    "height": player.get("height"),
                    "weight": player.get("weight"),
                    "college": player.get("college"),
                    "rookie_year": player.get("rookie_year"),
                    "experience": player.get("experience"),
                    "sr_id": player.get("sr_id"),
                    "references": player.get("references"),
                    "snapshot_ts": snapshot_ts,
                    "source_path": str(source) if source else None,
                }
            )
    df = pd.DataFrame(rows)
    output = out_dir / "roster_players.parquet"
    _write_or_log(df, output, dry_run)

    summary_output = out_dir / "roster_status_summary.parquet"
    if df.empty:
        _write_or_log(pd.DataFrame(), summary_output, dry_run)
        return

    summary = (
        df.groupby("team_id", as_index=False)
        .agg(
            team_name=("team_name", "first"),
            team_alias=("team_alias", "first"),
            roster_snapshot_ts=("snapshot_ts", "max"),
            roster_total_players=("player_id", "count"),
        )
    )
    summary["season"] = summary["roster_snapshot_ts"].apply(_snapshot_year)

    status_counts = (
        df.pivot_table(index="team_id", columns="status", values="player_id", aggfunc="count", fill_value=0)
        .rename(columns=lambda c: f"roster_status_{str(c).lower()}_count")
        .reset_index()
    )
    summary = summary.merge(status_counts, on="team_id", how="left")

    injury_cols = [c for c in summary.columns if c in {"roster_status_ir_count", "roster_status_pup_count", "roster_status_non_count"}]
    summary["roster_active_count"] = summary.get("roster_status_act_count", 0)
    summary["roster_practice_count"] = summary.get("roster_status_pra_count", 0)
    summary["roster_injured_count"] = summary[injury_cols].sum(axis=1, min_count=1) if injury_cols else 0
    summary["roster_injured_pct"] = summary["roster_injured_count"] / summary["roster_total_players"]
    summary["roster_active_pct"] = summary["roster_active_count"] / summary["roster_total_players"]

    skill_positions = {"QB", "RB", "WR", "TE"}
    active_skill = (
        df[(df["status"] == "ACT") & df["position"].isin(skill_positions)]
        .groupby("team_id")["player_id"]
        .count()
        .rename("roster_active_skill_count")
        .reset_index()
    )
    summary = summary.merge(active_skill, on="team_id", how="left")

    summary["season"] = pd.to_numeric(summary["season"], errors="coerce")
    _write_or_log(summary, summary_output, dry_run)


def _team_stat_rows(game_id: str, role: str, team_data: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for category, payload in team_data.items():
        if not isinstance(payload, dict):
            continue
        totals = payload.get("totals")
        if totals is not None:
            rows.append(
                {
                    "scope": "team",
                    "game_id": game_id,
                    "team_id": team_data.get("id"),
                    "team_role": role,
                    "stat_category": category,
                    "stats": totals,
                }
            )
    return rows


def _player_stat_rows(game_id: str, role: str, team_data: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for category, payload in team_data.items():
        if not isinstance(payload, dict):
            continue
        for player in payload.get("players", []) or []:
            rows.append(
                {
                    "scope": "player",
                    "game_id": game_id,
                    "team_id": team_data.get("id"),
                    "team_role": role,
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "player_position": player.get("position"),
                    "stat_category": category,
                    "stats": {k: v for k, v in player.items() if k not in {"id", "name", "position"}},
                }
            )
    return rows


def transform_game_stats(raw_root: Path, out_dir: Path, dry_run: bool) -> None:
    stats_root = raw_root / "game_statistics"
    team_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []
    if not stats_root.exists():
        logger.info("No game_statistics directory at %s", stats_root)
    for game_dir in sorted(stats_root.glob("*")):
        if not game_dir.is_dir():
            continue
        data, source = _latest_json(game_dir)
        if not data:
            continue
        statistics = data.get("statistics", {})
        game_id = data.get("id")
        for role in ("home", "away"):
            team_data = statistics.get(role)
            if not isinstance(team_data, dict):
                continue
            team_rows.extend(_team_stat_rows(game_id, role, team_data))
            player_rows.extend(_player_stat_rows(game_id, role, team_data))
    team_df = pd.DataFrame(team_rows)
    player_df = pd.DataFrame(player_rows)
    _write_or_log(team_df, out_dir / "game_team_stats.parquet", dry_run)
    _write_or_log(player_df, out_dir / "game_player_stats.parquet", dry_run)


def transform_seasonal_stats(raw_root: Path, out_dir: Path, dry_run: bool) -> None:
    stats_root = raw_root / "seasonal_statistics"
    rows: List[Dict[str, Any]] = []
    if not stats_root.exists():
        logger.info("No seasonal_statistics directory at %s", stats_root)
    for season_dir in sorted(stats_root.glob("*")):
        if not season_dir.is_dir():
            continue
        data, source = _latest_json(season_dir)
        if not data:
            continue
        season_meta = data.get("season", {})
        season_year = None
        if isinstance(season_meta, dict):
            season_year = season_meta.get("year")
        if season_year is None:
            season_year = data.get("season_year")
        teams = data.get("teams", [])
        for team in teams:
            if not isinstance(team, dict):
                continue
            base: Dict[str, Any] = {
                "season": season_year,
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "team_alias": team.get("alias"),
                "team_market": team.get("market"),
                "source_path": str(source) if source else None,
            }
            stats_block = team.get("statistics", {})
            for category, payload in stats_block.items():
                if not isinstance(payload, dict):
                    continue
                for key, value in payload.items():
                    if isinstance(value, (int, float)) and pd.notna(value):
                        base[f"sr_season_{category}_{key}"] = float(value)
                    elif isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, (int, float)) and pd.notna(sub_val):
                                base[f"sr_season_{category}_{key}_{sub_key}"] = float(sub_val)
            rows.append(base)
    df = pd.DataFrame(rows)
    output = out_dir / "seasonal_team_stats.parquet"
    _write_or_log(df, output, dry_run)


def transform_transactions(raw_root: Path, out_dir: Path, dry_run: bool) -> None:
    tx_root = raw_root / "daily_transactions"
    data, source = _latest_json(tx_root)
    rows: List[Dict[str, Any]] = []
    if data:
        for tx in data.get("transactions", []):
            team = tx.get("team", {})
            player = tx.get("player", {})
            rows.append(
                {
                    "transaction_id": tx.get("id"),
                    "transaction_type": tx.get("transaction_type") or tx.get("type"),
                    "status": tx.get("status"),
                    "effective_date": tx.get("effective_date") or tx.get("effective_on"),
                    "description": tx.get("description"),
                    "team_id": team.get("id") if isinstance(team, dict) else None,
                    "team_name": team.get("name") if isinstance(team, dict) else None,
                    "player_id": player.get("id") if isinstance(player, dict) else None,
                    "player_name": player.get("name") if isinstance(player, dict) else None,
                    "player_position": player.get("position") if isinstance(player, dict) else None,
                    "source_path": str(source) if source else None,
                }
            )
    df = pd.DataFrame(rows)
    _write_or_log(df, out_dir / "transactions.parquet", dry_run)


def transform_change_log(raw_root: Path, out_dir: Path, dry_run: bool) -> None:
    change_root = raw_root / "daily_change_log"
    data, source = _latest_json(change_root)
    rows: List[Dict[str, Any]] = []
    if data:
        for change in data.get("changes", []):
            team = change.get("team", {})
            player = change.get("player", {})
            game = change.get("game", {})
            rows.append(
                {
                    "change_id": change.get("id"),
                    "type": change.get("type"),
                    "description": change.get("description"),
                    "effective_date": change.get("effective"),
                    "team_id": team.get("id") if isinstance(team, dict) else None,
                    "team_name": team.get("name") if isinstance(team, dict) else None,
                    "player_id": player.get("id") if isinstance(player, dict) else None,
                    "player_name": player.get("name") if isinstance(player, dict) else None,
                    "game_id": game.get("id") if isinstance(game, dict) else None,
                    "source_path": str(source) if source else None,
                }
            )
    df = pd.DataFrame(rows)
    _write_or_log(df, out_dir / "change_log.parquet", dry_run)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[sportradar_transform] %(message)s",
    )
    targets = args.targets
    if "all" in targets:
        targets = ["schedule", "rosters", "game_stats", "seasonal_stats", "transactions", "change_log"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "schedule" in targets:
        transform_schedule(args.raw_dir, args.output_dir, args.dry_run)
    if "rosters" in targets:
        transform_rosters(args.raw_dir, args.output_dir, args.dry_run)
    if "game_stats" in targets:
        transform_game_stats(args.raw_dir, args.output_dir, args.dry_run)
    if "seasonal_stats" in targets:
        transform_seasonal_stats(args.raw_dir, args.output_dir, args.dry_run)
    if "transactions" in targets:
        transform_transactions(args.raw_dir, args.output_dir, args.dry_run)
    if "change_log" in targets:
        transform_change_log(args.raw_dir, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
