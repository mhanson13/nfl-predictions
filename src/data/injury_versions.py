"""Injury rows as they were at each game's pick time.

nflverse revises a week's injury rows as practice and game-status reports come in,
and 2026 rows carry no modification time. ``src.data.nflverse`` keeps every distinct
download with its UTC fetch time (``data/raw/history/nfl_injuries``). For each game,
features should use the latest version downloaded before that game's pick, not the
week's final report, which can include status changes made after the pick.

Pick time per game:
  * the pick of record's ``recorded_at`` (``predictions/picks_of_record``), if any;
  * otherwise kickoff minus the 2-hour lock lead (the latest a pick can be made).
The cutoff is never later than ``now``, so a live run uses every version downloaded
so far, the same rows as the cache.

Team-weeks with no logged version before the cutoff (everything before logging
began on 2026-11-19) keep their cached rows; ``summary`` counts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.utils.io import RAW_DIR

HISTORY_DIR = RAW_DIR / "history" / "nfl_injuries"
RECORDS_DIR = Path("predictions") / "picks_of_record"
LOCK_LEAD = timedelta(hours=2)
KEY = ["season", "week", "team"]


@dataclass
class AsOfSummary:
    team_weeks: int = 0
    from_log: int = 0
    from_cache: int = 0
    versions_used: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"[injuries] as-of-pick: {self.team_weeks} team-weeks; {self.from_log} from logged downloads "
            f"({len(self.versions_used)} versions), {self.from_cache} from cache (no logged version before the pick)"
        )


def kickoff_utc(schedule: pd.DataFrame) -> pd.Series:
    """Kickoff in UTC from nflverse ``gameday`` + ``gametime`` (US Eastern)."""
    local = pd.to_datetime(
        schedule["gameday"].astype(str) + " " + schedule["gametime"].fillna("13:00").astype(str),
        errors="coerce",
    )
    return local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def pick_times(schedule: pd.DataFrame, records_dir: Path = RECORDS_DIR) -> pd.DataFrame:
    """One row per (season, week, team): the pick-time cutoff for that team's game."""
    games = schedule[["game_id", "season", "week", "home_team", "away_team"]].copy()
    games["kickoff"] = kickoff_utc(schedule)
    games["pick_time"] = games["kickoff"] - LOCK_LEAD
    for season in sorted(pd.to_numeric(games["season"], errors="coerce").dropna().astype(int).unique()):
        path = records_dir / f"picks_of_record_{season}.csv"
        if not path.exists():
            continue
        rec = pd.read_csv(path, usecols=lambda c: c in {"game_id", "recorded_at"})
        if rec.empty or "recorded_at" not in rec:
            continue
        recorded = pd.to_datetime(rec["recorded_at"], utc=True, errors="coerce")
        recorded.index = rec["game_id"].astype(str).to_numpy()
        mapped = games["game_id"].map(recorded)
        games["pick_time"] = mapped.where(mapped.notna(), games["pick_time"])
    long = pd.concat(
        [
            games.rename(columns={"home_team": "team"})[["season", "week", "team", "pick_time"]],
            games.rename(columns={"away_team": "team"})[["season", "week", "team", "pick_time"]],
        ],
        ignore_index=True,
    )
    long["season"] = pd.to_numeric(long["season"], errors="coerce")
    long["week"] = pd.to_numeric(long["week"], errors="coerce")
    return long.dropna(subset=["season", "week", "team"]).drop_duplicates(KEY, keep="last")


def _manifest(history_dir: Path) -> pd.DataFrame:
    path = history_dir / "manifest.csv"
    if not path.exists():
        return pd.DataFrame(columns=["fetched_at_utc", "season", "file"])
    man = pd.read_csv(path)
    man["fetched_at"] = pd.to_datetime(man["fetched_at_utc"], utc=True, errors="coerce")
    return man.dropna(subset=["fetched_at"]).sort_values("fetched_at")


def injuries_as_of_pick(
    cache: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    history_dir: Path = HISTORY_DIR,
    records_dir: Path = RECORDS_DIR,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, AsOfSummary]:
    """``cache`` with each logged team-week replaced by the version current at its pick."""
    summary = AsOfSummary()
    man = _manifest(history_dir)
    if cache.empty or man.empty or not set(KEY).issubset(cache.columns):
        return cache, summary
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    cutoffs = pick_times(schedule, records_dir)
    cutoffs["cutoff"] = cutoffs["pick_time"].where(cutoffs["pick_time"] < now_ts, now_ts)

    base = cache.copy()
    base["season"] = pd.to_numeric(base["season"], errors="coerce")
    base["week"] = pd.to_numeric(base["week"], errors="coerce")
    logged_seasons = set(pd.to_numeric(man["season"], errors="coerce").dropna().astype(int))
    # Every scheduled team-week in a logged season, plus any cached team-week the
    # schedule lacks (those keep cached rows: no cutoff).
    scheduled = cutoffs[cutoffs["season"].isin(logged_seasons)]
    cached = base[base["season"].isin(logged_seasons)][KEY].drop_duplicates()
    team_weeks = scheduled.merge(cached, on=KEY, how="outer")
    summary.team_weeks = len(team_weeks)

    versions: dict[str, pd.DataFrame] = {}
    replaced_keys: list[tuple] = []
    replacement: list[pd.DataFrame] = []
    for row in team_weeks.itertuples(index=False):
        season_versions = man[pd.to_numeric(man["season"], errors="coerce") == row.season]
        if pd.notna(row.cutoff):
            season_versions = season_versions[season_versions["fetched_at"] <= row.cutoff]
        if season_versions.empty or pd.isna(row.cutoff):
            summary.from_cache += 1
            continue
        name = str(season_versions.iloc[-1]["file"])
        if name not in versions:
            v = pd.read_parquet(history_dir / name)
            v["season"] = pd.to_numeric(v["season"], errors="coerce")
            v["week"] = pd.to_numeric(v["week"], errors="coerce")
            versions[name] = v
        v = versions[name]
        rows = v[(v["season"] == row.season) & (v["week"] == row.week) & (v["team"] == row.team)]
        replaced_keys.append((row.season, row.week, row.team))
        replacement.append(rows)
        summary.from_log += 1
        summary.versions_used[name] = summary.versions_used.get(name, 0) + 1

    if not replaced_keys:
        return cache, summary
    keys = pd.MultiIndex.from_tuples(replaced_keys, names=KEY)
    keep = ~pd.MultiIndex.from_frame(base[KEY]).isin(keys)
    out = pd.concat([base.loc[keep], *replacement], ignore_index=True, sort=False)
    return out.reindex(columns=cache.columns), summary
