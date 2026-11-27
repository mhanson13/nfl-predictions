"""ESPN NFL rosters, depth charts and injuries.

Injuries come from the depth-chart athletes' ``injuries`` arrays (status, type, date, comments).
The ``teams/{id}/injuries`` endpoint returns ``{}`` (HTTP 200) and never produced a row, so it is
no longer called. Empty payloads and empty outputs print ``[espn_rosters] Warning:`` lines.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

from src.utils.io import RAW_DIR, write_df
from src.utils.logging_config import setup_logging
from src.utils.teams import normalize_team_abbr


BASE_URLS = (
    "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl",
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl",
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 20.0


def _request_json(session: requests.Session, url: str, *, timeout: float = DEFAULT_TIMEOUT) -> Any:
    response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _request_endpoint(session: requests.Session, endpoint: str, *, timeout: float = DEFAULT_TIMEOUT) -> Any:
    errors: list[requests.HTTPError] = []
    for base_url in BASE_URLS:
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            return _request_json(session, url, timeout=timeout)
        except requests.HTTPError as exc:
            errors.append(exc)
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in {403, 404}:
                raise
    if errors:
        raise errors[-1]
    raise RuntimeError(f"No ESPN base URLs configured for endpoint {endpoint!r}")


def _warn(message: str) -> None:
    logging.warning(message)
    print(f"[espn_rosters] Warning: {message}", flush=True)


def _is_empty_payload(payload: Any) -> bool:
    return payload is None or (isinstance(payload, (dict, list)) and len(payload) == 0)


def _get_nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _status_fields(status: Any) -> tuple[Any, Any, Any]:
    if isinstance(status, dict):
        return status.get("name"), status.get("type"), status.get("abbreviation")
    return status, None, None


def _team_abbr(team: dict[str, Any]) -> str | None:
    raw = team.get("abbreviation")
    if not raw:
        return None
    return normalize_team_abbr(str(raw), team.get("displayName"))


def normalize_teams(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sports = payload.get("sports") or []
    for sport in sports:
        for league in sport.get("leagues") or []:
            for wrapper in league.get("teams") or []:
                team = wrapper.get("team") if isinstance(wrapper, dict) else None
                if not isinstance(team, dict):
                    continue
                rows.append(
                    {
                        "espn_team_id": str(team.get("id")) if team.get("id") is not None else None,
                        "team": _team_abbr(team),
                        "abbreviation": team.get("abbreviation"),
                        "display_name": team.get("displayName"),
                        "short_display_name": team.get("shortDisplayName"),
                        "location": team.get("location"),
                        "name": team.get("name"),
                    }
                )
    return pd.DataFrame(rows).dropna(subset=["espn_team_id", "team"]).drop_duplicates("espn_team_id")


def normalize_roster(payload: dict[str, Any], *, season: int, week: int) -> pd.DataFrame:
    team = payload.get("team") if isinstance(payload.get("team"), dict) else {}
    team_abbr = _team_abbr(team)
    season_payload = payload.get("season") if isinstance(payload.get("season"), dict) else {}
    rows: list[dict[str, Any]] = []
    for group in payload.get("athletes") or []:
        group_position = group.get("position") if isinstance(group, dict) else None
        if isinstance(group_position, dict):
            roster_section = group_position.get("abbreviation") or group_position.get("name")
        else:
            roster_section = group_position
        for item in group.get("items") or []:
            if not isinstance(item, dict):
                continue
            status_name, status_type, status_abbr = _status_fields(item.get("status"))
            position = item.get("position") if isinstance(item.get("position"), dict) else {}
            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "espn_season_year": season_payload.get("year"),
                    "espn_season_type": season_payload.get("type"),
                    "team": team_abbr,
                    "espn_team_id": str(team.get("id")) if team.get("id") is not None else None,
                    "espn_player_id": str(item.get("id")) if item.get("id") is not None else None,
                    "uid": item.get("uid"),
                    "guid": item.get("guid"),
                    "display_name": item.get("displayName"),
                    "full_name": item.get("fullName") or item.get("displayName"),
                    "short_name": item.get("shortName"),
                    "first_name": item.get("firstName"),
                    "last_name": item.get("lastName"),
                    "jersey_number": item.get("jersey"),
                    "position": position.get("abbreviation") or position.get("name"),
                    "position_name": position.get("displayName") or position.get("name"),
                    "roster_section": roster_section,
                    "status_name": status_name,
                    "status_type": status_type,
                    "status_abbreviation": status_abbr,
                    "experience_years": _get_nested(item, "experience", "years"),
                    "height": item.get("height"),
                    "weight": item.get("weight"),
                }
            )
    return pd.DataFrame(rows)


def normalize_depthchart(payload: dict[str, Any], *, season: int, week: int) -> pd.DataFrame:
    team = payload.get("team") if isinstance(payload.get("team"), dict) else {}
    team_abbr = _team_abbr(team)
    rows: list[dict[str, Any]] = []
    for section in payload.get("depthchart") or []:
        if not isinstance(section, dict):
            continue
        positions = section.get("positions") or {}
        iterable = positions.values() if isinstance(positions, dict) else positions
        for pos_key, position_info in _iter_positions(positions, iterable):
            position = position_info.get("position") if isinstance(position_info.get("position"), dict) else {}
            athletes = position_info.get("athletes") or []
            for rank, athlete in enumerate(athletes, start=1):
                if not isinstance(athlete, dict):
                    continue
                rows.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "team": team_abbr,
                        "espn_team_id": str(team.get("id")) if team.get("id") is not None else None,
                        "depth_chart_section_id": section.get("id"),
                        "depth_chart_section": section.get("name"),
                        "depth_chart_position_key": pos_key,
                        "depth_chart_position": position.get("abbreviation") or position.get("name") or pos_key,
                        "depth_chart_position_name": position.get("name"),
                        "depth_rank": rank,
                        "espn_player_id": str(athlete.get("id")) if athlete.get("id") is not None else None,
                        "display_name": athlete.get("displayName"),
                        "short_name": athlete.get("shortName"),
                        "injury_count": len(athlete.get("injuries") or []),
                    }
                )
    return pd.DataFrame(rows)


def _iter_positions(positions: Any, iterable: Iterable[Any]) -> Iterable[tuple[Any, dict[str, Any]]]:
    if isinstance(positions, dict):
        for key, value in positions.items():
            if isinstance(value, dict):
                yield key, value
        return
    for value in iterable:
        if isinstance(value, dict):
            yield value.get("abbreviation") or value.get("name"), value


def normalize_injuries(payload: Any, *, season: int, week: int, team: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records: list[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("injuries") or payload.get("athletes") or payload.get("items") or []
        if isinstance(records, dict):
            records = list(records.values())
    else:
        records = []

    for record in records:
        if not isinstance(record, dict):
            continue
        athlete = record.get("athlete") if isinstance(record.get("athlete"), dict) else record.get("player")
        if not isinstance(athlete, dict):
            athlete = record
        record_team = record.get("team") if isinstance(record.get("team"), dict) else {}
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": team or _team_abbr(record_team),
                "espn_player_id": str(athlete.get("id")) if athlete.get("id") is not None else None,
                "display_name": athlete.get("displayName") or athlete.get("fullName"),
                "short_name": athlete.get("shortName"),
                "injury_status": record.get("status") or record.get("statusName") or record.get("type"),
                "injury_type": record.get("type"),
                "injury_detail": record.get("details") or record.get("detail"),
                "injury_date": record.get("date"),
            }
        )
    return pd.DataFrame(rows)


def _injury_type(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("abbreviation") or value.get("name") or value.get("description")
    return value


INJURY_COLUMNS = [
    "season", "week", "team", "espn_player_id", "display_name", "short_name",
    "injury_status", "injury_type", "injury_detail", "injury_date",
]


def injuries_from_depthchart(
    payload: dict[str, Any], *, season: int, week: int, team: str | None = None
) -> pd.DataFrame:
    """One row per injured athlete from a ``teams/{id}/depthcharts`` payload.

    An athlete listed at several depth slots appears once; with several injury entries the
    latest ``date`` wins. Columns match :func:`normalize_injuries`.
    """
    team_obj = payload.get("team") if isinstance(payload, dict) and isinstance(payload.get("team"), dict) else {}
    team_abbr = team or _team_abbr(team_obj)
    rows: list[dict[str, Any]] = []
    for section in (payload.get("depthchart") if isinstance(payload, dict) else None) or []:
        if not isinstance(section, dict):
            continue
        positions = section.get("positions") or {}
        iterable = positions.values() if isinstance(positions, dict) else positions
        for _, position_info in _iter_positions(positions, iterable):
            for athlete in position_info.get("athletes") or []:
                if not isinstance(athlete, dict) or athlete.get("id") is None:
                    continue
                for injury in athlete.get("injuries") or []:
                    if not isinstance(injury, dict):
                        continue
                    rows.append(
                        {
                            "season": int(season),
                            "week": int(week),
                            "team": team_abbr,
                            "espn_player_id": str(athlete.get("id")),
                            "display_name": athlete.get("displayName") or athlete.get("fullName"),
                            "short_name": athlete.get("shortName"),
                            "injury_status": injury.get("status") or _get_nested(injury, "type", "description"),
                            "injury_type": _injury_type(injury.get("type")),
                            "injury_detail": injury.get("shortComment") or injury.get("details"),
                            "injury_date": injury.get("date"),
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=INJURY_COLUMNS)
    out = pd.DataFrame(rows, columns=INJURY_COLUMNS)
    out = out.sort_values(["espn_player_id", "injury_date"], kind="stable", na_position="first")
    return out.drop_duplicates("espn_player_id", keep="last").reset_index(drop=True)


def fetch_espn_roster_snapshots(
    *,
    season: int,
    week: int,
    team_ids: Sequence[str] | None = None,
    sleep: float = 0.1,
    timeout: float = DEFAULT_TIMEOUT,
    include_depthcharts: bool = True,
    include_injuries: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session = requests.Session()
    teams_payload = _request_endpoint(session, "teams", timeout=timeout)
    teams = normalize_teams(teams_payload)
    selected_ids = [str(value) for value in team_ids] if team_ids else teams["espn_team_id"].dropna().astype(str).tolist()

    roster_frames: list[pd.DataFrame] = []
    depth_frames: list[pd.DataFrame] = []
    injury_frames: list[pd.DataFrame] = []
    team_lookup = teams.set_index("espn_team_id")["team"].to_dict() if not teams.empty else {}

    if _is_empty_payload(teams_payload) or teams.empty:
        _warn("ESPN teams endpoint returned no teams")
    empty_endpoints: list[str] = []

    for team_id in selected_ids:
        logging.info("Fetching ESPN roster for team_id=%s", team_id)
        roster_payload = _request_endpoint(session, f"teams/{team_id}/roster", timeout=timeout)
        if _is_empty_payload(roster_payload):
            empty_endpoints.append(f"teams/{team_id}/roster")
        roster_frames.append(normalize_roster(roster_payload, season=season, week=week))
        if sleep > 0:
            time.sleep(sleep)

        # One depth-chart request serves both the depth chart and the injuries (its athletes
        # carry ``injuries`` arrays). teams/{id}/injuries returns {} and is not called.
        if include_depthcharts or include_injuries:
            logging.info("Fetching ESPN depthchart for team_id=%s", team_id)
            depth_payload = _request_endpoint(session, f"teams/{team_id}/depthcharts", timeout=timeout)
            if _is_empty_payload(depth_payload):
                empty_endpoints.append(f"teams/{team_id}/depthcharts")
            if include_depthcharts:
                depth_frames.append(normalize_depthchart(depth_payload, season=season, week=week))
            if include_injuries:
                injury_frames.append(
                    injuries_from_depthchart(
                        depth_payload, season=season, week=week, team=team_lookup.get(str(team_id))
                    )
                )
            if sleep > 0:
                time.sleep(sleep)

    if empty_endpoints:
        _warn(f"{len(empty_endpoints)} ESPN endpoint(s) returned an empty payload: {', '.join(empty_endpoints[:8])}")

    rosters = pd.concat(roster_frames, ignore_index=True, sort=False) if roster_frames else pd.DataFrame()
    depthcharts = pd.concat(depth_frames, ignore_index=True, sort=False) if depth_frames else pd.DataFrame()
    injury_frames = [f for f in injury_frames if not f.empty]
    injuries = (
        pd.concat(injury_frames, ignore_index=True, sort=False) if injury_frames else pd.DataFrame(columns=INJURY_COLUMNS)
    )
    return teams, rosters, depthcharts, injuries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch current ESPN NFL team rosters, depth charts, and injuries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--team-ids", nargs="+", default=None, help="Optional ESPN team IDs to fetch.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between ESPN requests.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--skip-depthcharts", action="store_true")
    parser.add_argument("--skip-injuries", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("nfl_predictions", level="DEBUG" if args.debug else "INFO")
    teams, rosters, depthcharts, injuries = fetch_espn_roster_snapshots(
        season=args.season,
        week=args.week,
        team_ids=args.team_ids,
        sleep=args.sleep,
        timeout=args.timeout,
        include_depthcharts=not args.skip_depthcharts,
        include_injuries=not args.skip_injuries,
    )

    teams_path = write_df(teams, RAW_DIR / "espn_teams.parquet")
    rosters_path = write_df(rosters, RAW_DIR / f"espn_rosters_{int(args.season)}_wk{int(args.week):02d}.parquet")
    depth_path = write_df(depthcharts, RAW_DIR / f"espn_depthcharts_{int(args.season)}_wk{int(args.week):02d}.parquet")
    injuries_path = write_df(injuries, RAW_DIR / f"espn_injuries_{int(args.season)}_wk{int(args.week):02d}.parquet")
    print(f"[espn_rosters] wrote teams rows={len(teams)} -> {teams_path.resolve()}")
    print(f"[espn_rosters] wrote roster rows={len(rosters)} -> {rosters_path.resolve()}")
    print(f"[espn_rosters] wrote depthchart rows={len(depthcharts)} -> {depth_path.resolve()}")
    print(f"[espn_rosters] wrote injury rows={len(injuries)} -> {injuries_path.resolve()}")
    expected = {"teams": teams, "roster": rosters}
    if not args.skip_depthcharts:
        expected["depthchart"] = depthcharts
    if not args.skip_injuries:
        # In season every depth chart lists some injured athletes; 0 means the source changed.
        expected["injury"] = injuries
    for name, frame in expected.items():
        if frame.empty:
            _warn(f"wrote 0 {name} rows for {int(args.season)} week {int(args.week)}; check the ESPN response")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
