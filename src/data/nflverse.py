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

"""Wrappers around nflverse / SportsDataverse style APIs used for data ingestion.

Each fetch_* function mirrors a single nfl_data_py dataset and writes the results
to the canonical parquet files consumed downstream.  The CLI in this module is
invoked from the main pipeline during the data-ingestion stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from src.utils.io import RAW_DIR, read_df, write_df
from src.utils.logging import configure as configure_logging
from src.utils.pydantic_schemas import validate_dataframe

SEASON_HISTORY_PATH = Path("data/reference/nfl_seasons_history.csv")
MIN_INJURY_SEASON = 2009
MIN_PFR_SEASON = 2018


def _load_finalized_seasons() -> set[int]:
    try:
        df = pd.read_csv(SEASON_HISTORY_PATH)
    except FileNotFoundError:
        return set()
    current_year = datetime.utcnow().year
    finalized = (
        df[df["champion"].notna() & (df["season"].astype(int) < current_year)]["season"]
        .astype(int)
        .tolist()
    )
    return set(finalized)


def _load_existing(path: Path) -> pd.DataFrame:
    """Return cached dataset at ``path`` or an empty frame when absent/invalid."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = read_df(path)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as exc:
        print(f"[nflverse] warning: unable to read {path}: {exc}")
        return pd.DataFrame()


def _determine_seasons_to_fetch(
    existing: pd.DataFrame,
    requested: List[int],
    finalized: set[int],
    *,
    force: bool,
) -> List[int]:
    """Decide which seasons need downloading given cached data and finalized season list."""
    if force or existing.empty or "season" not in existing.columns:
        return requested
    existing_seasons = set(
        pd.to_numeric(existing["season"], errors="coerce").dropna().astype(int).tolist()
    )
    to_fetch: List[int] = []
    for season in requested:
        if force:
            to_fetch.append(season)
        elif season not in existing_seasons:
            to_fetch.append(season)
        elif season not in finalized:
            to_fetch.append(season)
    return to_fetch


def _merge_and_write(path: Path, existing: pd.DataFrame, new_df: pd.DataFrame) -> None:
    """Append new rows onto cached dataset and write back to disk."""
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(ignore_index=True)
    write_df(combined, path)


# Map from dataset label → schema key (for datasets where validation is meaningful)
_LABEL_TO_SCHEMA = {
    "schedules":    "nflverse_schedule",
    "rosters":      "nflverse_roster",
    "injuries":     "nflverse_injury",
    "player stats": "nflverse_player_stats",
    "PBP":          "nflverse_pbp",
}


def _validate_and_report(label: str, df: pd.DataFrame) -> None:
    """Run schema validation for *df* if a schema is registered for *label*."""
    schema_key = _LABEL_TO_SCHEMA.get(label)
    if schema_key is None:
        return
    try:
        report = validate_dataframe(df, schema_key, log_errors=False)
        print(f"[nflverse] schema validation ({label}): {report}")
    except Exception as exc:  # pragma: no cover
        print(f"[nflverse] schema validation skipped for {label}: {exc}")


def _print_skip(dataset: str, seasons: List[int]) -> None:
    print(f"[nflverse] {dataset}: all requested seasons {seasons} already cached (finalized). Skipping fetch.")


def _try_import_nfl():
    """Import nfl_data_py with a consistent error message when the dependency is missing."""
    try:
        import nfl_data_py as nfl  # type: ignore
        return nfl
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'nfl-data-py'. Install with: pip install nfl-data-py"
        ) from e


def _filter_supported_seasons(
    seasons: List[int],
    *,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    label: str,
) -> List[int]:
    """Return seasons within [min_year, max_year] (inclusive) and log skips for unsupported years."""
    supported = []
    for season in seasons:
        if min_year is not None and season < min_year:
            continue
        if max_year is not None and season > max_year:
            continue
        supported.append(season)
    skipped = sorted(set(seasons) - set(supported))
    if skipped:
        details = []
        if min_year is not None:
            details.append(f"minimum {min_year}")
        if max_year is not None:
            details.append(f"maximum {max_year}")
        window = ", ".join(details) if details else "supported window"
        print(f"[nflverse] {label} skipping unsupported seasons {skipped} ({window}).")
    return supported


def fetch_pbp(seasons: List[int]) -> pd.DataFrame:
    """Fetch play-by-play data for the requested seasons, skipping future years."""
    nfl = _try_import_nfl()
    frames: List[pd.DataFrame] = []
    current_year = datetime.utcnow().year
    for yr in seasons:
        if yr > current_year:
            print(f"[nflverse] skipping PBP fetch for future season {yr}")
            continue
        try:
            df = nfl.import_pbp_data([yr], downcast=True)
            if df is None or df.empty:
                print(f"[nflverse] no PBP for {yr}")
                continue
            if "season" not in df.columns:
                df["season"] = yr
            if "game_id" not in df.columns and "gameId" in df.columns:
                df = df.rename(columns={"gameId": "game_id"})
            frames.append(df)
            print(f"{yr} done.")
        except Exception as e:
            print(f"[nflverse] PBP fetch failed for {yr}: {e}")
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_schedules(seasons: List[int]) -> pd.DataFrame:
    """Download schedule rows from nfl_data_py without mutating columns."""
    nfl = _try_import_nfl()
    return nfl.import_schedules(seasons)


# ---- Additional nflverse datasets ----

def _report_missing_function(func_name: str, seasons: List[int]) -> None:
    """Emit a consistent log message when a requested nfl_data_py function does not exist."""
    print(f"[nflverse] function missing: {func_name} (seasons={seasons})")


def _fetch_generic(nfl, func_name: str, seasons: List[int]) -> Optional[pd.DataFrame]:
    """Invoke an nfl_data_py import_* function if present and return the resulting DataFrame."""
    fn: Optional[Callable] = getattr(nfl, func_name, None)
    if not callable(fn):
        _report_missing_function(func_name, seasons)
        return None
    try:
        df = fn(seasons)  # type: ignore
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            print(f"[nflverse] {func_name} returned empty")
            return None
        return df
    except Exception as e:
        print(f"[nflverse] {func_name} failed: {e}")
        return None


def fetch_rosters(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch roster snapshots via nfl_data_py (falls back to releases in run_dataset)."""
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_rosters", seasons)


def fetch_injuries(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch injury reports (available 2009+) with nfl_data_py."""
    nfl = _try_import_nfl()
    supported = _filter_supported_seasons(seasons, min_year=MIN_INJURY_SEASON, label="injuries")
    if not supported:
        return None
    return _fetch_generic(nfl, "import_injuries", supported)


def fetch_snap_counts(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch team snap counts (available from 2012 onward) and skip unsupported seasons."""
    nfl = _try_import_nfl()
    min_year = 2012
    current_year = datetime.utcnow().year
    supported = [season for season in seasons if min_year <= season <= current_year]
    skipped = sorted(set(seasons) - set(supported))
    if skipped:
        print(
            f"[nflverse] snap counts not requested for unsupported seasons {skipped} "
            f"(provider minimum {min_year})."
        )
    if not supported:
        return None
    return _fetch_generic(nfl, "import_snap_counts", supported)


def fetch_participation(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch weekly participation reports (players on field per play)."""
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_participation", seasons)


def fetch_depth_charts(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch depth chart snapshots (starters/backups) from nfl_data_py."""
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_depth_charts", seasons)


def fetch_player_stats(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch player-level seasonal stats, preferring modern APIs with legacy fallbacks."""
    nfl = _try_import_nfl()
    # Preferred modern endpoint (available in recent nfl_data_py builds)
    fn = getattr(nfl, "import_seasonal_data", None)
    if callable(fn):
        try:
            df = fn(seasons, s_type="REG")  # type: ignore[arg-type]
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as exc:  # pragma: no cover - upstream errors
            print(f"[nflverse] import_seasonal_data failed: {exc}")
    # Legacy fallbacks
    for name in ("import_seasonal_player_stats", "import_player_stats"):
        fn_legacy = getattr(nfl, name, None)
        if not callable(fn_legacy):
            continue
        try:
            df = fn_legacy(seasons)  # type: ignore[misc]
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as exc:  # pragma: no cover
            print(f"[nflverse] {name} failed: {exc}")
    return None


def fetch_pfr_passing(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch PFR passing stats via modern or legacy nflverse helpers."""
    nfl = _try_import_nfl()
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_passing")
    if not supported:
        return None
    fn = getattr(nfl, "import_seasonal_pfr", None)
    if callable(fn):
        try:
            df = fn("pass", years=supported)  # type: ignore[misc]
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as exc:
            print(f"[nflverse] import_seasonal_pfr(pass) failed: {exc}")
    return _fetch_generic(nfl, "import_pfr_passing", supported)


def fetch_pfr_rushing(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch PFR rushing stats via modern or legacy nflverse helpers."""
    nfl = _try_import_nfl()
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_rushing")
    if not supported:
        return None
    fn = getattr(nfl, "import_seasonal_pfr", None)
    if callable(fn):
        try:
            df = fn("rush", years=supported)  # type: ignore[misc]
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as exc:
            print(f"[nflverse] import_seasonal_pfr(rush) failed: {exc}")
    return _fetch_generic(nfl, "import_pfr_rushing", supported)


def fetch_pfr_receiving(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch PFR receiving stats via modern or legacy nflverse helpers."""
    nfl = _try_import_nfl()
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_receiving")
    if not supported:
        return None
    fn = getattr(nfl, "import_seasonal_pfr", None)
    if callable(fn):
        try:
            df = fn("rec", years=supported)  # type: ignore[misc]
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as exc:
            print(f"[nflverse] import_seasonal_pfr(rec) failed: {exc}")
    return _fetch_generic(nfl, "import_pfr_receiving", supported)


# ---- Direct download fallbacks from nflverse-data releases ----

BASE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{fname}"


def _try_read_url(url: str) -> Optional[pd.DataFrame]:
    try:
        if url.endswith(".parquet"):
            return pd.read_parquet(url)
        return pd.read_csv(url, low_memory=False)
    except Exception:
        return None


def _download_release_dataset(tag: str, patterns: List[str], seasons: List[int]) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for yr in seasons:
        success = False
        for pat in patterns:
            fname_csv = pat.format(season=yr)
            # try csv
            url_csv = BASE_RELEASE.format(tag=tag, fname=fname_csv)
            df = _try_read_url(url_csv)
            if df is not None and not df.empty:
                df["season"] = df.get("season", yr)
                frames.append(df)
                success = True
                break
            # try parquet with same base name
            fname_pq = fname_csv.rsplit(".", 1)[0] + ".parquet"
            url_pq = BASE_RELEASE.format(tag=tag, fname=fname_pq)
            df = _try_read_url(url_pq)
            if df is not None and not df.empty:
                df["season"] = df.get("season", yr)
                frames.append(df)
                success = True
                break
        if not success:
            print(f"[nflverse] release not found for {tag} {yr}")
    return pd.concat(frames, ignore_index=True) if frames else None


def download_rosters(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("rosters", ["roster_{season}.csv", "rosters_{season}.csv"], seasons)


def download_injuries(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_INJURY_SEASON, label="injuries release")
    if not supported:
        return None
    return _download_release_dataset("injuries", ["injuries_{season}.csv"], supported)


def download_participation(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("participation", ["participation_{season}.csv"], seasons)


def download_player_stats(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("player_stats", ["player_stats_{season}.csv"], seasons)


def download_pfr_passing(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_passing release")
    if not supported:
        return None
    return _download_release_dataset("pfr", ["pfr_passing_{season}.csv"], supported)


def download_pfr_rushing(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_rushing release")
    if not supported:
        return None
    return _download_release_dataset("pfr", ["pfr_rushing_{season}.csv"], supported)


def download_pfr_receiving(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_receiving release")
    if not supported:
        return None
    return _download_release_dataset("pfr", ["pfr_receiving_{season}.csv"], supported)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True)
    ap.add_argument("--pbp", action="store_true")
    ap.add_argument("--schedules", action="store_true")
    ap.add_argument("--rosters", action="store_true")
    ap.add_argument("--injuries", action="store_true")
    ap.add_argument("--snaps", action="store_true")
    ap.add_argument("--participation", action="store_true")
    ap.add_argument("--depthcharts", action="store_true")
    ap.add_argument("--players", action="store_true")
    ap.add_argument("--pfr", nargs="*", choices=["passing", "rushing", "receiving"], help="PFR tables to fetch")
    ap.add_argument("--force-refresh", action="store_true", help="Ignore caches and re-download data")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    args = ap.parse_args()

    configure_logging(args.debug)

    seasons = [int(s) for s in args.season]
    finalized_seasons = _load_finalized_seasons()

    def run_dataset(
        label: str,
        path: Path,
        fetch_fn: Callable[[List[int]], Optional[pd.DataFrame]],
        fallback_fn: Optional[Callable[[List[int]], Optional[pd.DataFrame]]] = None,
    ) -> None:
        """
        Fetch ``label`` for the requested seasons and persist to ``path``.

        Args:
            label: Human-readable dataset name used in log messages.
            path: Destination parquet path under ``data/raw``.
            fetch_fn: Primary callable that accepts a list of seasons and returns a DataFrame or None.
            fallback_fn: Optional secondary callable used when the primary fetcher returns empty/None.
        """
        existing = _load_existing(path)
        seasons_to_fetch = _determine_seasons_to_fetch(existing, seasons, finalized_seasons, force=args.force_refresh)
        if not seasons_to_fetch:
            _print_skip(label, seasons)
            return

        df = fetch_fn(seasons_to_fetch)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            _validate_and_report(label, df)
            _merge_and_write(path, existing, df)
            print(f"[nflverse] saved {label} rows={len(df)} -> {path}")
            return

        if fallback_fn is not None:
            df_fallback = fallback_fn(seasons_to_fetch)
            if df_fallback is not None and isinstance(df_fallback, pd.DataFrame) and not df_fallback.empty:
                _validate_and_report(label, df_fallback)
                _merge_and_write(path, existing, df_fallback)
                print(f"[nflverse] downloaded {label} rows={len(df_fallback)} -> {path}")
                return

        print(f"[nflverse] no {label} returned for seasons {seasons_to_fetch}")

    if args.pbp:
        run_dataset("PBP", RAW_DIR / "nfl_pbp.parquet", fetch_pbp)

    if args.schedules:
        run_dataset("schedules", RAW_DIR / "nfl_schedules.parquet", fetch_schedules)

    if args.rosters:
        run_dataset("rosters", RAW_DIR / "nfl_rosters.parquet", fetch_rosters, download_rosters)

    if args.injuries:
        run_dataset("injuries", RAW_DIR / "nfl_injuries.parquet", fetch_injuries, download_injuries)

    if args.snaps:
        run_dataset("snap counts", RAW_DIR / "nfl_snap_counts.parquet", fetch_snap_counts)

    if args.participation:
        run_dataset("participation", RAW_DIR / "nfl_participation.parquet", fetch_participation, download_participation)

    if args.depthcharts:
        run_dataset("depth charts", RAW_DIR / "nfl_depth_charts.parquet", fetch_depth_charts)

    if args.players:
        run_dataset("player stats", RAW_DIR / "nfl_player_stats.parquet", fetch_player_stats, download_player_stats)

    if args.pfr:
        if "passing" in args.pfr:
            run_dataset("pfr_passing", RAW_DIR / "pfr_passing.parquet", fetch_pfr_passing, download_pfr_passing)
        if "rushing" in args.pfr:
            run_dataset("pfr_rushing", RAW_DIR / "pfr_rushing.parquet", fetch_pfr_rushing, download_pfr_rushing)
        if "receiving" in args.pfr:
            run_dataset("pfr_receiving", RAW_DIR / "pfr_receiving.parquet", fetch_pfr_receiving, download_pfr_receiving)


if __name__ == "__main__":
    main()
