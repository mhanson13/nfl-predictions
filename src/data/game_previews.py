"""Early-week game previews for MTB: schedule and market data only, never model output.

One row per game of the target week, written to::

    predictions/previews/previews.csv                       (MTB: current/previews.csv)
    predictions/previews/previews_<season>_week_<nn>.csv    (MTB: previews/<season>/week_<nn>.csv)

Built after a week is graded (``tools.dispatcher``) for the next week, and refreshed on
Wednesday. Sources:

* schedule: a fresh download of nflverse's ``games`` release (falls back to the local
  ``data/raw/nfl_schedules.parquet``): kickoff, roof, venue;
* odds: one The Odds API call on the free key (moneyline, spread, total; region ``us``),
  reduced to the median across books -- the same median the pick page shows
  (``predict_upcoming``'s ``market_*`` columns). Games without a line there fall back to
  the nflverse schedule's line; games with neither keep blank odds columns.

The builder never reads predictions or any model artifact. ``PREVIEW_COLUMNS`` is the
whole output; :func:`validate_preview_frame` refuses anything else before writing.

Usage::

    python -m src.data.game_previews --season 2026 --after-week 12   # previews for the next week
    python -m src.data.game_previews --season 2026 --week 13 --max-age-hours 6
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PREVIEW_DIR = Path("predictions/previews")
CURRENT_NAME = "previews.csv"
LOCAL_SCHEDULE_PATH = Path("data/raw/nfl_schedules.parquet")
ODDS_MARKETS = ("h2h", "spreads", "totals")
ODDS_SOURCE_API = "the-odds-api:us-median"
ODDS_SOURCE_NFLVERSE = "nflverse-schedule"
# An Odds API event matches a schedule game when the teams match and kickoffs are this close.
EVENT_MATCH_WINDOW = timedelta(hours=36)

PREVIEW_COLUMNS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "pick_expected_at",
    "roof",
    "venue_name",
    "venue_city",
    "neutral_site",
    "spread_home",
    "total",
    "home_ml",
    "away_ml",
    "odds_source",
    "odds_captured_at",
    "generated_at",
]
ODDS_COLUMNS = ["spread_home", "total", "home_ml", "away_ml", "odds_source", "odds_captured_at"]
# Belt and braces on top of the exact column list: nothing that reads like model output.
MODEL_COLUMN_RE = re.compile(r"prob|pick(?!_expected_at$)|margin|edge|confidence|model|pred|tier|recommend", re.I)
# Columns that change on every build; ignored when deciding whether a refresh changed anything.
VOLATILE_COLUMNS = ("odds_captured_at", "generated_at")


class PreviewError(RuntimeError):
    pass


def _iso_utc(ts) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value) -> str:
    """A label or ''; never NaN/None in the file."""
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


def week_paths(season: int, week: int, out_dir: Path = PREVIEW_DIR) -> tuple[Path, Path]:
    return out_dir / CURRENT_NAME, out_dir / f"previews_{season}_week_{week:02d}.csv"


def next_week(schedule: pd.DataFrame, season: int, after_week: int) -> int | None:
    """The next week number on the schedule after ``after_week`` (playoff weeks included)."""
    weeks = pd.to_numeric(schedule.loc[schedule["season"] == season, "week"], errors="coerce").dropna().astype(int)
    later = sorted(w for w in weeks.unique() if w > after_week)
    return later[0] if later else None


def load_schedule(season: int, *, download: bool = True) -> tuple[pd.DataFrame, str]:
    """Fresh nflverse schedule for ``season`` (its lines are current), else the local copy."""
    if download:
        try:
            from src.data.nflverse import download_schedules

            fresh = download_schedules([season])
            if fresh is not None and not fresh.empty:
                return fresh, "nflverse-release"
        except Exception as exc:  # network or parse failure: fall back to the local copy
            print(f"[previews] Warning: schedule download failed ({type(exc).__name__}: {exc}); using local copy")
    if not LOCAL_SCHEDULE_PATH.exists():
        raise PreviewError(f"no schedule: download failed and {LOCAL_SCHEDULE_PATH} is missing")
    local = pd.read_parquet(LOCAL_SCHEDULE_PATH)
    return local[local["season"] == season].copy(), "local-parquet"


def pick_expected_times(schedule: pd.DataFrame) -> pd.Series:
    """game_id -> when the dispatcher's pick run for that game's slot can start (UTC)."""
    from tools.dispatcher import START_LEAD, build_slots

    out: dict[str, pd.Timestamp] = {}
    for slot in build_slots(schedule):
        opens = pd.Timestamp(slot.first_kickoff - START_LEAD)
        for gid in slot.game_ids:
            out[gid] = opens
    return pd.Series(out, dtype="object")


def fetch_api_odds() -> tuple[list[dict] | None, str]:
    """One live Odds API call on the free key; ``(None, reason)`` when unavailable."""
    from src.utils.odds import fetch_odds, get_odds_api_free_key

    key = get_odds_api_free_key()
    if not key:
        return None, "free Odds API key not configured"
    data = fetch_odds(api_key=key, regions="us", markets=ODDS_MARKETS, odds_format="american")
    if data is None:
        return None, "Odds API request failed"
    return data, ""


def _median_moneyline(prices: list[float]) -> float:
    """Median of American prices taken in decimal odds (a plain median of e.g. -105 and +100 gives -2.5)."""
    from src.utils.odds import american_to_decimal, decimal_to_american

    decimals = [american_to_decimal(p) for p in prices if p not in (0, None) and np.isfinite(p)]
    if not decimals:
        return np.nan
    return float(decimal_to_american(float(np.median(decimals))))


def summarize_api_odds(events: list[dict] | None) -> pd.DataFrame:
    """Median across books per event: home/away moneyline, home spread (negative = home favoured), total."""
    from src.data.odds_api_pick_time_lines import _abbr

    rows = []
    for event in events or []:
        home, away = event.get("home_team"), event.get("away_team")
        prices: dict[str, list[float]] = {"home_ml": [], "away_ml": [], "spread_home": [], "total": []}
        for book in event.get("bookmakers") or []:
            for market in book.get("markets") or []:
                key = market.get("key")
                for outcome in market.get("outcomes") or []:
                    name, price, point = outcome.get("name"), outcome.get("price"), outcome.get("point")
                    if key == "h2h" and price is not None:
                        if name == home:
                            prices["home_ml"].append(float(price))
                        elif name == away:
                            prices["away_ml"].append(float(price))
                    elif key == "spreads" and name == home and point is not None:
                        prices["spread_home"].append(float(point))
                    elif key == "totals" and str(name).lower() == "over" and point is not None:
                        prices["total"].append(float(point))
        rows.append({
            "home_abbr": _abbr(home),
            "away_abbr": _abbr(away),
            "commence": pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce"),
            "home_ml": _median_moneyline(prices["home_ml"]),
            "away_ml": _median_moneyline(prices["away_ml"]),
            "spread_home": float(np.median(prices["spread_home"])) if prices["spread_home"] else np.nan,
            "total": float(np.median(prices["total"])) if prices["total"] else np.nan,
        })
    return pd.DataFrame(rows, columns=["home_abbr", "away_abbr", "commence", "home_ml", "away_ml", "spread_home", "total"])


def _round_line(x: float) -> float:
    return np.nan if pd.isna(x) else round(float(x) * 2) / 2  # spreads/totals on the half point


def _round_ml(x: float) -> float:
    return np.nan if pd.isna(x) else float(round(x))


def build_previews(
    schedule: pd.DataFrame,
    season: int,
    week: int,
    *,
    api_odds: pd.DataFrame | None,
    odds_captured_at: datetime | None,
    generated_at: datetime,
    pick_times: pd.Series | None = None,
) -> pd.DataFrame:
    """The preview rows for one week. Pure function of the schedule and the odds summary."""
    from src.utils.nfl_calendar import _kickoffs
    from src.utils.venues import game_venue_table

    games = schedule[(schedule["season"] == season) & (pd.to_numeric(schedule["week"], errors="coerce") == week)].copy()
    if games.empty:
        raise PreviewError(f"no games on the schedule for {season} week {week}")
    games = games.drop_duplicates("game_id", keep="last")
    games["kickoff_ts"] = _kickoffs(games).dt.tz_convert("UTC")
    venues = game_venue_table(games).set_index("game_id")
    pick_times = pick_times if pick_times is not None else pick_expected_times(schedule[schedule["season"] == season])
    captured = _iso_utc(pd.Timestamp(odds_captured_at)) if odds_captured_at else ""

    rows = []
    for g in games.sort_values(["kickoff_ts", "game_id"]).itertuples(index=False):
        row = {
            "season": int(season),
            "week": int(week),
            "game_id": str(g.game_id),
            "home_team": g.home_team,
            "away_team": g.away_team,
            "kickoff": _iso_utc(g.kickoff_ts),
            "pick_expected_at": _iso_utc(pick_times.get(str(g.game_id))),
            "roof": _text(getattr(g, "roof", None)),
            "venue_name": _text(venues["venue_name"].get(g.game_id)),
            "venue_city": _text(venues["venue_city"].get(g.game_id)),
            "neutral_site": bool(venues["neutral_site"].get(g.game_id, False)),
            "spread_home": np.nan, "total": np.nan, "home_ml": np.nan, "away_ml": np.nan,
            "odds_source": "", "odds_captured_at": "",
            "generated_at": _iso_utc(pd.Timestamp(generated_at)),
        }
        match = pd.DataFrame()
        if api_odds is not None and not api_odds.empty and pd.notna(g.kickoff_ts):
            match = api_odds[(api_odds["home_abbr"] == g.home_team) & (api_odds["away_abbr"] == g.away_team)
                             & ((api_odds["commence"] - g.kickoff_ts).abs() <= EVENT_MATCH_WINDOW)]
        if not match.empty and match[["home_ml", "away_ml", "spread_home", "total"]].notna().any(axis=None):
            m = match.iloc[0]
            row.update(spread_home=_round_line(m.spread_home), total=_round_line(m.total),
                       home_ml=_round_ml(m.home_ml), away_ml=_round_ml(m.away_ml),
                       odds_source=ODDS_SOURCE_API, odds_captured_at=captured)
        else:
            spread_line = pd.to_numeric(getattr(g, "spread_line", np.nan), errors="coerce")
            fallback = {
                # nflverse spread_line is the expected home margin (positive = home favoured).
                "spread_home": -spread_line if pd.notna(spread_line) else np.nan,
                "total": pd.to_numeric(getattr(g, "total_line", np.nan), errors="coerce"),
                "home_ml": pd.to_numeric(getattr(g, "home_moneyline", np.nan), errors="coerce"),
                "away_ml": pd.to_numeric(getattr(g, "away_moneyline", np.nan), errors="coerce"),
            }
            if any(pd.notna(v) for v in fallback.values()):
                row.update(spread_home=_round_line(fallback["spread_home"]), total=_round_line(fallback["total"]),
                           home_ml=_round_ml(fallback["home_ml"]), away_ml=_round_ml(fallback["away_ml"]),
                           odds_source=ODDS_SOURCE_NFLVERSE, odds_captured_at=_iso_utc(pd.Timestamp(generated_at)))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=PREVIEW_COLUMNS)
    for col in ("home_ml", "away_ml"):
        frame[col] = frame[col].astype("Int64")
    validate_preview_frame(frame)
    return frame


def validate_preview_frame(frame: pd.DataFrame) -> None:
    """Refuse anything but the agreed columns, in order, and anything that reads like model output."""
    if list(frame.columns) != PREVIEW_COLUMNS:
        raise PreviewError(f"preview columns {list(frame.columns)} != agreed {PREVIEW_COLUMNS}")
    leaked = [c for c in frame.columns if MODEL_COLUMN_RE.search(c)]
    if leaked:
        raise PreviewError(f"model-like columns in preview: {leaked}")
    if frame["game_id"].duplicated().any():
        raise PreviewError("duplicate game_id in preview")


def content_key(frame: pd.DataFrame) -> str:
    """Stable fingerprint of a preview file without its timestamps (for 'publish only on change')."""
    body = frame.drop(columns=[c for c in VOLATILE_COLUMNS if c in frame.columns])
    return str(pd.util.hash_pandas_object(body.astype(str), index=False).sum())


def recent_build(path: Path, season: int, week: int, max_age: timedelta, now: datetime) -> pd.DataFrame | None:
    """The existing week file if it was generated within ``max_age`` (reused instead of a new Odds API call)."""
    if max_age <= timedelta(0) or not path.exists():
        return None
    try:
        frame = pd.read_csv(path, dtype={"game_id": str})
        validate_preview_frame(frame)
        for col in ("home_ml", "away_ml"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").round().astype("Int64")
        for col in ("odds_source", "odds_captured_at", "roof", "venue_name", "venue_city"):
            frame[col] = frame[col].fillna("")
        generated = pd.to_datetime(frame["generated_at"], utc=True, errors="coerce").max()
    except Exception:
        return None
    if frame.empty or int(frame["season"].iloc[0]) != season or int(frame["week"].iloc[0]) != week:
        return None
    if pd.isna(generated) or now - generated.to_pydatetime() > max_age:
        return None
    return frame


def write_previews(frame: pd.DataFrame, season: int, week: int, out_dir: Path = PREVIEW_DIR) -> tuple[Path, Path]:
    validate_preview_frame(frame)
    current, archive = week_paths(season, week, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["neutral_site"] = out["neutral_site"].map(lambda v: "true" if str(v).lower() == "true" else "false")
    for path in (archive, current):
        tmp = path.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(path)
    return current, archive


def run(
    season: int,
    week: int,
    *,
    max_age: timedelta = timedelta(0),
    use_odds_api: bool = True,
    download_schedule: bool = True,
    out_dir: Path = PREVIEW_DIR,
    now: datetime | None = None,
    schedule: pd.DataFrame | None = None,
    odds_fetcher: Callable[[], tuple[list[dict] | None, str]] = fetch_api_odds,
) -> tuple[pd.DataFrame, str]:
    """Build (or reuse) and write the week's previews. Returns ``(frame, note)``."""
    now = now or datetime.now(timezone.utc)
    current, archive = week_paths(season, week, out_dir)
    reused = recent_build(archive, season, week, max_age, now)
    if reused is not None:
        write_previews(reused, season, week, out_dir)  # current/ may point at another week; re-point it
        return reused, f"reused {archive} (generated {reused['generated_at'].max()}, within {max_age})"
    if schedule is None:
        schedule, schedule_source = load_schedule(season, download=download_schedule)
    else:
        schedule_source = "caller"
    api_odds, captured, odds_note = None, None, "Odds API not used"
    if use_odds_api:
        events, reason = odds_fetcher()
        captured = datetime.now(timezone.utc)
        if events is None:
            odds_note = f"Warning: {reason}; using nflverse lines where present"
        else:
            api_odds = summarize_api_odds(events)
            odds_note = f"Odds API: {len(api_odds)} events"
    frame = build_previews(schedule, season, week, api_odds=api_odds, odds_captured_at=captured, generated_at=now)
    write_previews(frame, season, week, out_dir)
    by_source = frame["odds_source"].replace("", "none").value_counts().to_dict()
    return frame, f"schedule {schedule_source}; {odds_note}; odds by source {by_source}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--week", type=int, help="Week to preview.")
    target.add_argument("--after-week", type=int, help="Preview the next week on the schedule after this one.")
    ap.add_argument("--max-age-hours", type=float, default=0.0,
                    help="Reuse the week's file if it was generated within this many hours (no new Odds API call).")
    ap.add_argument("--no-odds-api", action="store_true", help="Skip the Odds API; nflverse lines only.")
    ap.add_argument("--no-download", action="store_true", help="Use the local schedule copy only.")
    ap.add_argument("--out-dir", type=Path, default=PREVIEW_DIR)
    args = ap.parse_args(argv)

    schedule, source = load_schedule(args.season, download=not args.no_download)
    print(f"[previews] schedule: {source}")
    week = args.week
    if week is None:
        week = next_week(schedule, args.season, args.after_week)
        if week is None:
            print(f"[previews] no week after {args.season} week {args.after_week} on the schedule; nothing to build")
            return 0
    try:
        frame, note = run(args.season, week, max_age=timedelta(hours=args.max_age_hours),
                          use_odds_api=not args.no_odds_api, download_schedule=not args.no_download,
                          out_dir=args.out_dir, schedule=schedule)
    except PreviewError as exc:
        print(f"[previews] failed: {exc}")
        return 1
    current, archive = week_paths(args.season, week, args.out_dir)
    print(f"[previews] {args.season} week {week}: {len(frame)} games -> {archive} (and {current}); {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
