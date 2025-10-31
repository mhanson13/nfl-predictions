from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

import httpx

from src.utils.http import SportradarClient
from src.utils.secrets import get_secret
from src.utils.io import RAW_DIR
from src.utils.logging import configure as configure_logging
from src.utils.checkpoints import get_last_timestamp, update_timestamp


STATIC_FEEDS = {
    "league_hierarchy": "league/hierarchy.json",
    "teams": "league/teams.json",
    "seasons": "league/seasons.json",
    "current_season_schedule": "games/current_season/schedule.json",
}

SCHEDULE_FEEDS = {"season_schedule", "weekly_schedule"}
DEPTH_CHART_FEEDS = {"weekly_depth_charts"}
ROSTER_FEEDS = {"team_roster"}
GAME_FEEDS = {"game_boxscore", "game_statistics", "game_play_by_play"}
PROFILE_FEEDS = {"team_profile", "player_profile"}
CHANGE_FEEDS = {"daily_change_log", "daily_transactions"}
SEASONAL_FEEDS = {"seasonal_statistics"}

ALL_FEEDS = (
    set(STATIC_FEEDS.keys())
    | SCHEDULE_FEEDS
    | DEPTH_CHART_FEEDS
    | ROSTER_FEEDS
    | GAME_FEEDS
    | PROFILE_FEEDS
    | CHANGE_FEEDS
    | SEASONAL_FEEDS
)

DEFAULT_FEEDS = ["league_hierarchy", "teams", "seasons"]

logger = logging.getLogger("sportradar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch supplemental NFL data from Sportradar (Phase B)."
    )
    parser.add_argument(
        "--feeds",
        nargs="+",
        default=DEFAULT_FEEDS,
        choices=sorted(ALL_FEEDS),
        help="Feeds to fetch.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=RAW_DIR / "sportradar",
        help="Directory to store raw JSON responses.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Season years to target (required for schedule/depth chart feeds).",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        help="Week numbers for weekly feeds (e.g., 1 2 3).",
    )
    parser.add_argument(
        "--team-ids",
        nargs="+",
        help="Team IDs for roster/profile feeds.",
    )
    parser.add_argument(
        "--team-ids-file",
        type=Path,
        help="File containing team IDs (one per line).",
    )
    parser.add_argument(
        "--game-ids",
        nargs="+",
        help="Game IDs for game-level feeds.",
    )
    parser.add_argument(
        "--game-ids-file",
        type=Path,
        help="File containing game IDs (one per line).",
    )
    parser.add_argument(
        "--schedule-json",
        type=Path,
        help="Path to a Sportradar season schedule JSON used to derive game IDs.",
    )
    parser.add_argument(
        "--player-ids",
        nargs="+",
        help="Player IDs for player profile feed.",
    )
    parser.add_argument(
        "--player-ids-file",
        type=Path,
        help="File containing player IDs (one per line).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log target URLs without fetching.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--since",
        type=str,
        help="ISO timestamp to override checkpoint for change feeds.",
    )
    return parser.parse_args()


def ensure_api_key() -> str:
    key = get_secret("SPORTSRADAR_NFL_API_KEY")
    if not key:
        raise RuntimeError("Missing SPORTSRADAR_NFL_API_KEY in environment or secrets.env.")
    return key


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", value)


def load_ids_from_file(path: Optional[Path]) -> list[str]:
    results: list[str] = []
    if not path:
        return results
    if not path.exists():
        raise FileNotFoundError(f"IDs file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token:
            results.append(token)
    return results


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for val in values:
        if val not in seen:
            ordered.append(val)
            seen.add(val)
    return ordered


def load_latest_team_ids(save_dir: Path) -> list[str]:
    team_dir = save_dir / "teams"
    files = sorted(team_dir.glob("*.json"))
    if not files:
        raise RuntimeError(
            "No saved team data found. Fetch 'teams' feed before requesting team-dependent feeds."
        )
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    teams = data.get("teams", [])
    return [team["id"] for team in teams if "id" in team]


def extract_game_ids_from_schedule(schedule_path: Path, weeks: Optional[set[int]] = None) -> list[str]:
    weeks = weeks or set()
    data = json.loads(schedule_path.read_text(encoding="utf-8"))
    collected: list[str] = []
    schedule_weeks = data.get("weeks") or []
    for wk in schedule_weeks:
        seq = wk.get("sequence")
        if weeks and seq not in weeks:
            continue
        for game in wk.get("games", []):
            gid = game.get("id")
            if gid:
                collected.append(gid)
    if not collected and "games" in data:
        for game in data.get("games", []):
            seq = game.get("week")
            if weeks and seq not in weeks:
                continue
            gid = game.get("id")
            if gid:
                collected.append(gid)
    return dedupe_preserve_order(collected)


@dataclass(frozen=True)
class FeedRequest:
    feed: str
    path: str
    qualifiers: Mapping[str, str]


def build_output_path(base_dir: Path, feed: str, qualifiers: Mapping[str, str]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dir_parts = [base_dir, feed]
    for key in ("season", "week", "team_id", "game_id", "player_id"):
        value = qualifiers.get(key)
        if value:
            dir_parts.append(sanitize(value))
    target_dir = Path(*dir_parts)
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = "__".join(
        f"{key}-{sanitize(value)}"
        for key, value in sorted(qualifiers.items())
        if value
    )
    filename = f"{feed}_{suffix}_{timestamp}.json" if suffix else f"{feed}_{timestamp}.json"
    return target_dir / filename


def fetch_to_path(client: SportradarClient, path: str, target: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] Would fetch %s -> %s", path, target)
        return

    logger.info("Fetching %s", path)
    try:
        response = client.request(path)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response else "unknown"
        if status in (401, 403):
            logger.warning("Unauthorized (%s) for %s; skipping Sportradar fetch.", status, path)
            return
        if status == 404:
            logger.warning("Feed not available (404) for %s; skipping.", path)
            return
        if status == 429:
            logger.warning("Rate limited (429) for %s; skipping after retries.", path)
            return
        raise
    except Exception as exc:
        logger.warning("Failed to fetch %s (%s); skipping.", path, exc)
        return
    data = response.json()
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %s (%d bytes)", target, target.stat().st_size)


def fetch_change_feed(
    client: SportradarClient,
    feed: str,
    params: Mapping[str, str],
    save_dir: Path,
    dry_run: bool,
) -> Path | None:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "__".join(f"{k}-{sanitize(v)}" for k, v in sorted(params.items()))
    name = f"{feed}_{suffix}_{timestamp}.json" if suffix else f"{feed}_{timestamp}.json"
    target = save_dir / name
    if dry_run:
        logger.info("[dry-run] Would fetch %s with %s -> %s", feed, params, target)
        return None
    try:
        response = client.request(f"league/{feed}.json", params=params)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response else "unknown"
        if status in (401, 403):
            logger.warning("Unauthorized (%s) for change feed %s; skipping.", status, feed)
            return None
        if status == 404:
            logger.warning("Change feed %s not available (404); skipping.", feed)
            return None
        if status == 429:
            logger.warning("Rate limited (429) for change feed %s; skipping after retries.", feed)
            return None
        raise
    except Exception as exc:
        logger.warning("Failed to fetch change feed %s (%s); skipping.", feed, exc)
        return None
    data = response.json()
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %s (%d bytes)", target, target.stat().st_size)
    return target


def build_requests(
    feeds: Iterable[str],
    seasons: list[int],
    weeks: list[int],
    team_ids: list[str],
    game_ids: list[str],
    player_ids: list[str],
) -> list[FeedRequest]:
    requests: list[FeedRequest] = []
    season_set = seasons or []
    week_set = weeks or []

    for feed in feeds:
        if feed in STATIC_FEEDS:
            requests.append(FeedRequest(feed, STATIC_FEEDS[feed], {}))
        elif feed == "season_schedule":
            if not season_set:
                raise ValueError("season_schedule feed requires --seasons.")
            for season in season_set:
                path = f"games/{season}/REG/schedule.json"
                requests.append(
                    FeedRequest(feed, path, {"season": str(season)})
                )
        elif feed == "weekly_schedule":
            if not season_set or not week_set:
                raise ValueError("weekly_schedule feed requires --seasons and --weeks.")
            for season in season_set:
                for week in week_set:
                    path = f"games/{season}/REG/{week}/schedule.json"
                    qualifiers = {"season": str(season), "week": f"{week:02d}"}
                    requests.append(FeedRequest(feed, path, qualifiers))
        elif feed == "weekly_depth_charts":
            if not season_set or not week_set:
                raise ValueError("weekly_depth_charts feed requires --seasons and --weeks.")
            for season in season_set:
                for week in week_set:
                    path = f"depthcharts/{season}/REG/{week}.json"
                    qualifiers = {"season": str(season), "week": f"{week:02d}"}
                    requests.append(FeedRequest(feed, path, qualifiers))
        elif feed == "team_roster":
            if not team_ids:
                raise ValueError("team_roster feed requires --team-ids or preloaded team data.")
            for team_id in team_ids:
                path = f"teams/{team_id}/full_roster.json"
                qualifiers = {"team_id": team_id}
                requests.append(FeedRequest(feed, path, qualifiers))
        elif feed in {"team_profile"}:
            if not team_ids:
                raise ValueError(f"{feed} feed requires --team-ids or preloaded team data.")
            for team_id in team_ids:
                path = f"teams/{team_id}/profile.json"
                qualifiers = {"team_id": team_id}
                requests.append(FeedRequest(feed, path, qualifiers))
        elif feed in {"player_profile"}:
            if not player_ids:
                raise ValueError("player_profile feed requires --player-ids.")
            for player_id in player_ids:
                path = f"players/{player_id}/profile.json"
                qualifiers = {"player_id": player_id}
                requests.append(FeedRequest(feed, path, qualifiers))
        elif feed in {"game_boxscore"}:
            if not game_ids:
                raise ValueError("game_boxscore feed requires --game-ids or --schedule-json.")
            for game_id in game_ids:
                path = f"games/{game_id}/boxscore.json"
                qualifiers = {"game_id": game_id}
                requests.append(FeedRequest(feed, path, qualifiers))
        elif feed in {"game_statistics"}:
            if not game_ids:
                raise ValueError("game_statistics feed requires --game-ids or --schedule-json.")
            for game_id in game_ids:
                path = f"games/{game_id}/statistics.json"
                qualifiers = {"game_id": game_id}
                requests.append(FeedRequest(feed, path, qualifiers))
        elif feed in {"game_play_by_play"}:
            if not game_ids:
                raise ValueError("game_play_by_play feed requires --game-ids or --schedule-json.")
            for game_id in game_ids:
                path = f"games/{game_id}/pbp.json"
                qualifiers = {"game_id": game_id}
                requests.append(FeedRequest(feed, path, qualifiers))
        elif feed == "seasonal_statistics":
            if not season_set:
                raise ValueError("seasonal_statistics feed requires --seasons.")
            for season in season_set:
                path = f"seasons/{season}/REG/teams/statistics.json"
                qualifiers = {"season": str(season)}
                requests.append(FeedRequest(feed, path, qualifiers))
        elif feed == "daily_change_log":
            requests.append(
                FeedRequest(
                    feed,
                    "league/daily_change_log.json",
                    {},
                )
            )
        elif feed == "daily_transactions":
            requests.append(
                FeedRequest(
                    feed,
                    "league/daily_transactions.json",
                    {},
                )
            )
        else:
            raise ValueError(f"Unhandled feed '{feed}'")

    # Deduplicate identical requests
    unique: list[FeedRequest] = []
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    for req in requests:
        key = (req.feed, req.path, tuple(sorted(req.qualifiers.items())))
        if key not in seen:
            seen.add(key)
            unique.append(req)
    return unique


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    key = ensure_api_key()
    if args.dry_run:
        logger.info("Dry-run mode: no requests will be sent.")

    seasons = args.seasons[:] if args.seasons else []
    weeks = args.weeks[:] if args.weeks else []

    team_ids = dedupe_preserve_order(
        list(args.team_ids or []) + load_ids_from_file(args.team_ids_file)
    )
    player_ids = dedupe_preserve_order(
        list(args.player_ids or []) + load_ids_from_file(args.player_ids_file)
    )

    game_ids = dedupe_preserve_order(
        list(args.game_ids or []) + load_ids_from_file(args.game_ids_file)
    )
    if args.schedule_json:
        week_filter = set(weeks) if weeks else None
        ids_from_schedule = extract_game_ids_from_schedule(args.schedule_json, week_filter)
        game_ids = dedupe_preserve_order(game_ids + ids_from_schedule)

    feeds = list(args.feeds)
    feeds_needing_teams = set(feeds) & (ROSTER_FEEDS | {"team_profile"})
    if feeds_needing_teams and not team_ids:
        try:
            team_ids = load_latest_team_ids(args.save_dir)
            logger.info("Loaded %d team IDs from latest teams feed.", len(team_ids))
        except Exception as exc:
            logger.warning(
                "Unable to derive team IDs for roster/profile requests (%s); skipping feeds %s.",
                exc,
                sorted(feeds_needing_teams),
            )
            feeds = [f for f in feeds if f not in feeds_needing_teams]
            feeds_needing_teams = set()

    manual_change_since = None
    if args.since:
        try:
            manual_change_since = datetime.fromisoformat(args.since)
        except ValueError as exc:
            raise ValueError("--since must be ISO8601 timestamp.") from exc

    feed_requests = build_requests(
        feeds, seasons, weeks, team_ids, game_ids, player_ids
    )
    logger.info("Prepared %d feed requests.", len(feed_requests))

    change_updates: dict[str, datetime] = {}
    change_targets: dict[str, Path] = {}
    client = SportradarClient(api_key=key)
    try:
        for req in feed_requests:
            if req.feed in CHANGE_FEEDS:
                last_ts = manual_change_since or get_last_timestamp(req.feed)
                params = {}
                if last_ts:
                    params["since"] = last_ts.isoformat()
                target = fetch_change_feed(
                    client,
                    req.feed,
                    params,
                    args.save_dir / req.feed,
                    args.dry_run,
                )
                if target:
                    change_targets[req.feed] = target
                    change_updates[req.feed] = datetime.utcnow()
                continue

            target = build_output_path(args.save_dir, req.feed, req.qualifiers)
            fetch_to_path(client, req.path, target, args.dry_run)
    finally:
        client.close()

    if change_updates and not args.dry_run:
        for feed, ts in change_updates.items():
            update_timestamp(feed, ts)

    # Cascade change log follow-ups
    if change_targets and not args.dry_run:
        team_ids_from_changes: list[str] = []
        player_ids_from_changes: list[str] = []
        game_ids_from_changes: list[str] = []

        for feed, path in change_targets.items():
            data = json.loads(path.read_text(encoding="utf-8"))
            items = []
            if feed == "daily_change_log":
                items = data.get("changes", [])
            elif feed == "daily_transactions":
                items = data.get("transactions", [])

            for item in items:
                if "team" in item and isinstance(item["team"], dict):
                    tid = item["team"].get("id")
                    if tid:
                        team_ids_from_changes.append(tid)
                if "player" in item and isinstance(item["player"], dict):
                    pid = item["player"].get("id")
                    if pid:
                        player_ids_from_changes.append(pid)
                if "game" in item and isinstance(item["game"], dict):
                    gid = item["game"].get("id")
                    if gid:
                        game_ids_from_changes.append(gid)

        team_ids_from_changes = dedupe_preserve_order(team_ids_from_changes)
        player_ids_from_changes = dedupe_preserve_order(player_ids_from_changes)
        game_ids_from_changes = dedupe_preserve_order(game_ids_from_changes)

        if team_ids_from_changes or player_ids_from_changes or game_ids_from_changes:
            logger.info(
                "Triggered follow-up fetches from change feeds: %d teams, %d players, %d games.",
                len(team_ids_from_changes),
                len(player_ids_from_changes),
                len(game_ids_from_changes),
            )
            follow_up = []
            if team_ids_from_changes:
                follow_up.append("team_roster")
                follow_up.append("team_profile")
            if player_ids_from_changes:
                follow_up.append("player_profile")
            if game_ids_from_changes:
                follow_up.extend(["game_boxscore", "game_statistics"])

            if follow_up:
                follow_reqs = build_requests(
                    follow_up,
                    seasons,
                    weeks,
                    team_ids_from_changes or [],
                    game_ids_from_changes or [],
                    player_ids_from_changes or [],
                )
                logger.info("Executing %d follow-up requests.", len(follow_reqs))
                client = SportradarClient(api_key=key)
                try:
                    for req in follow_reqs:
                        target = build_output_path(args.save_dir, req.feed, req.qualifiers)
                        fetch_to_path(client, req.path, target, args.dry_run)
                finally:
                    client.close()


if __name__ == "__main__":
    main()
