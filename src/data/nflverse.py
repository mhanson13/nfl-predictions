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

"""Wrappers around nflverse release assets and SportsDataverse-style APIs.

Each fetch_* function writes one canonical parquet file consumed downstream.
Maintained nflverse-data GitHub release assets are the primary source when
available; nfl_data_py remains a compatibility fallback for missing release
assets or provider-specific helper behavior.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from src.utils.io import RAW_DIR, read_df, write_df
from src.utils.logging_config import setup_logging
from src.utils.pydantic_schemas import validate_dataframe

SEASON_HISTORY_PATH = Path("data/reference/nfl_seasons_history.csv")
MIN_INJURY_SEASON = 2009
MIN_PFR_SEASON = 2018
MIN_SNAP_COUNT_SEASON = 2012
SCHEDULE_TEXT_ID_COLUMNS = (
    "old_game_id",
    "gsis",
    "nfl_detail_id",
    "pfr",
    "pff",
    "espn",
    "ftn",
)


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


def _dedupe_schedule_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "game_id" in df.columns:
        key = ["game_id"]
    elif {"season", "week", "home_team", "away_team"}.issubset(df.columns):
        key = ["season", "week", "home_team", "away_team"]
    else:
        return df.drop_duplicates(ignore_index=True)

    dup_mask = df.duplicated(subset=key, keep=False)
    if dup_mask.any():
        examples = df.loc[dup_mask, key].drop_duplicates().head(5).to_dict("records")
        print(f"[nflverse] schedules: dropping duplicate rows on {key}: {examples}")
    return df.drop_duplicates(subset=key, keep="last", ignore_index=True)


def _as_optional_text_id(value: object) -> object:
    """Normalize provider ID values to stable text while preserving missing values."""
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_schedule_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schedule columns that frequently mix numeric and string IDs."""
    if df.empty:
        return df
    out = df.copy()
    for col in SCHEDULE_TEXT_ID_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(_as_optional_text_id).astype("string")
    return out


def _merge_and_write(path: Path, existing: pd.DataFrame, new_df: pd.DataFrame) -> Path:
    """Append new rows onto cached dataset and write back to disk."""
    if path.name == "nfl_player_stats.parquet":
        if not existing.empty:
            existing = _normalize_player_stats(existing)
        new_df = _normalize_player_stats(new_df)
    elif path.name == "nfl_schedules.parquet":
        existing = _normalize_schedule_columns(existing)
        new_df = _normalize_schedule_columns(new_df)

    if existing.empty:
        combined = new_df
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
                category=FutureWarning,
            )
            combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    if path.name == "nfl_schedules.parquet":
        combined = _normalize_schedule_columns(_dedupe_schedule_rows(combined))
    else:
        combined = combined.drop_duplicates(ignore_index=True)
    return write_df(combined, path)


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
    """Fetch play-by-play data, preferring maintained nflverse release files."""
    release_df = download_pbp(seasons)
    if release_df is not None and not release_df.empty:
        return release_df

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
            msg = str(e)
            if isinstance(e, NameError) and "name 'Error' is not defined" in msg:
                print(
                    f"[nflverse] PBP unavailable for {yr}: nfl_data_py raised "
                    "NameError('Error') for the current-season feed; continuing with cached data."
                )
            else:
                print(f"[nflverse] PBP fetch failed for {yr}: {e}")
            continue

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()





def fetch_schedules(seasons: List[int]) -> pd.DataFrame:
    """Fetch schedule rows, preferring maintained nflverse release files."""
    release_df = download_schedules(seasons)
    if release_df is not None and not release_df.empty:
        return release_df

    nfl = _try_import_nfl()
    return nfl.import_schedules(seasons)




# ---- Additional nflverse datasets ----



def _report_missing_function(func_name: str, seasons: List[int]) -> None:
    """Emit a consistent log message when a requested nfl_data_py function does not exist."""
    if func_name.startswith("import_pfr_"):
        print(f"[nflverse] optional PFR helper unavailable: {func_name} (seasons={seasons})")
    else:
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
    """Fetch roster snapshots, preferring maintained nflverse release files."""
    release_df = download_rosters(seasons)
    if release_df is not None and not release_df.empty:
        return release_df

    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_rosters", seasons)




def fetch_injuries(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch injury reports, preferring maintained nflverse release files."""
    release_df = download_injuries(seasons)
    if release_df is not None and not release_df.empty:
        return release_df

    nfl = _try_import_nfl()
    supported = _filter_supported_seasons(seasons, min_year=MIN_INJURY_SEASON, label="injuries")
    if not supported:
        return None
    return _fetch_generic(nfl, "import_injuries", supported)




def fetch_snap_counts(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch team snap counts, preferring maintained nflverse release files."""
    release_df = download_snap_counts(seasons)
    if release_df is not None and not release_df.empty:
        return release_df

    nfl = _try_import_nfl()
    current_year = datetime.utcnow().year
    supported = [season for season in seasons if MIN_SNAP_COUNT_SEASON <= season <= current_year]
    skipped = sorted(set(seasons) - set(supported))
    if skipped:
        print(
            f"[nflverse] snap counts not requested for unsupported seasons {skipped} "
            f"(provider minimum {MIN_SNAP_COUNT_SEASON})."
        )
    if not supported:
        return None
    return _fetch_generic(nfl, "import_snap_counts", supported)




def fetch_participation(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch weekly participation reports (players on field per play)."""
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_participation", seasons)




def fetch_depth_charts(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch depth chart snapshots, preferring maintained nflverse release files."""
    release_df = download_depth_charts(seasons)
    if release_df is not None and not release_df.empty:
        return release_df

    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_depth_charts", seasons)




def fetch_player_stats(seasons: List[int]) -> Optional[pd.DataFrame]:
    """Fetch player-level weekly stats, preferring current nflverse release files."""
    release_df = download_player_stats(seasons)
    if release_df is not None and isinstance(release_df, pd.DataFrame) and not release_df.empty:
        return _normalize_player_stats(release_df)

    nfl = _try_import_nfl()
    # Downstream player projection exports expect weekly rows. Fetch seasons
    # individually so an unavailable current-season file does not discard every
    # season in the request.
    fn_weekly = getattr(nfl, "import_weekly_data", None)
    if callable(fn_weekly):
        frames: List[pd.DataFrame] = []
        for season in seasons:
            try:
                df = fn_weekly([season], downcast=True, thread_requests=False)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - upstream/network errors vary
                print(f"[nflverse] import_weekly_data failed for {season}: {exc}")
                continue
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        if frames:
            return _normalize_player_stats(pd.concat(frames, ignore_index=True))

    # Fallback endpoints return seasonal rows in some nfl_data_py versions.
    fn = getattr(nfl, "import_seasonal_data", None)
    if callable(fn):
        try:
            df = fn(seasons, s_type="REG")  # type: ignore[arg-type]
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize_player_stats(df)
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
                return _normalize_player_stats(df)
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





def _download_release_dataset(
    tag: str,
    patterns: List[str],
    seasons: List[int],
    *,
    log_missing: bool = True,
) -> Optional[pd.DataFrame]:
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

        if not success and log_missing:
            print(f"[nflverse] release not found for {tag} {yr}")

    return pd.concat(frames, ignore_index=True) if frames else None



def _download_all_seasons_release_dataset(
    tag: str,
    filenames: List[str],
    seasons: List[int],
    *,
    season_col: str = "season",
) -> Optional[pd.DataFrame]:
    """Download one release asset containing multiple seasons and filter it."""
    wanted = {int(season) for season in seasons}

    for fname in filenames:
        url = BASE_RELEASE.format(tag=tag, fname=fname)
        df = _try_read_url(url)
        if df is None or df.empty:
            continue
        if season_col not in df.columns:
            print(f"[nflverse] release {tag}/{fname} missing '{season_col}' column")
            return None
        season_values = pd.to_numeric(df[season_col], errors="coerce")
        out = df.loc[season_values.isin(wanted)].copy()
        return out if not out.empty else None

    print(f"[nflverse] release not found for {tag}")
    return None


def download_pbp(seasons: List[int]) -> Optional[pd.DataFrame]:
    current_year = datetime.utcnow().year
    supported = _filter_supported_seasons(seasons, max_year=current_year, label="PBP release")
    if not supported:
        return None
    return _download_release_dataset(
        "pbp",
        [
            "play_by_play_{season}.parquet",
            "play_by_play_{season}.csv.gz",
            "play_by_play_{season}.csv",
        ],
        supported,
    )


def download_schedules(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_all_seasons_release_dataset(
        "schedules",
        ["games.parquet", "games.csv.gz", "games.csv"],
        seasons,
    )




def download_rosters(seasons: List[int]) -> Optional[pd.DataFrame]:

    return _download_release_dataset("rosters", ["roster_{season}.csv", "rosters_{season}.csv"], seasons)





def download_injuries(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_INJURY_SEASON, label="injuries release")
    if not supported:
        return None
    return _download_release_dataset("injuries", ["injuries_{season}.csv"], supported)




def download_snap_counts(seasons: List[int]) -> Optional[pd.DataFrame]:
    current_year = datetime.utcnow().year
    supported = _filter_supported_seasons(
        seasons,
        min_year=MIN_SNAP_COUNT_SEASON,
        max_year=current_year,
        label="snap counts release",
    )
    if not supported:
        return None
    return _download_release_dataset(
        "snap_counts",
        [
            "snap_counts_{season}.parquet",
            "snap_counts_{season}.csv.gz",
            "snap_counts_{season}.csv",
        ],
        supported,
    )


def download_depth_charts(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset(
        "depth_charts",
        [
            "depth_charts_{season}.parquet",
            "depth_charts_{season}.csv",
        ],
        seasons,
    )


def download_participation(seasons: List[int]) -> Optional[pd.DataFrame]:

    return _download_release_dataset("participation", ["participation_{season}.csv"], seasons)





def download_player_stats(seasons: List[int]) -> Optional[pd.DataFrame]:

    modern = _download_release_dataset(
        "stats_player",
        [
            "stats_player_week_{season}.parquet",
            "stats_player_week_{season}.csv.gz",
            "stats_player_week_{season}.csv",
        ],
        seasons,
        log_missing=False,
    )
    if modern is not None and not modern.empty:
        return _normalize_player_stats(modern)

    legacy = _download_release_dataset(
        "player_stats",
        [
            "player_stats_{season}.parquet",
            "player_stats_{season}.csv.gz",
            "player_stats_{season}.csv",
        ],
        seasons,
    )
    return _normalize_player_stats(legacy) if legacy is not None and not legacy.empty else None

def _normalize_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "recent_team" not in out.columns and "team" in out.columns:
        out["recent_team"] = out["team"]
    if "team" not in out.columns and "recent_team" in out.columns:
        out["team"] = out["recent_team"]

    if "season_type" in out.columns:
        season_type = out["season_type"]
        season_type_text = season_type.astype("string").str.strip().str.upper()
        mapped = season_type_text.map(
            {
                "REG": 2,
                "REGULAR": 2,
                "REGULAR_SEASON": 2,
                "REGULAR SEASON": 2,
                "POST": 3,
                "POSTSEASON": 3,
                "POST_SEASON": 3,
            }
        )
        numeric = pd.to_numeric(season_type, errors="coerce")
        out["season_type"] = numeric.where(numeric.notna(), mapped)

    return out


def download_pfr_passing(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_passing release")
    if not supported:
        return None
    return _download_release_dataset("pfr", ["pfr_passing_{season}.csv"], supported, log_missing=False)


def download_pfr_rushing(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_rushing release")
    if not supported:
        return None
    return _download_release_dataset("pfr", ["pfr_rushing_{season}.csv"], supported, log_missing=False)


def download_pfr_receiving(seasons: List[int]) -> Optional[pd.DataFrame]:
    supported = _filter_supported_seasons(seasons, min_year=MIN_PFR_SEASON, label="pfr_receiving release")
    if not supported:
        return None
    return _download_release_dataset("pfr", ["pfr_receiving_{season}.csv"], supported, log_missing=False)




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

    setup_logging("nfl_predictions", level="DEBUG" if args.debug else "INFO")

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
            output_path = _merge_and_write(path, existing, df)
            print(f"[nflverse] saved {label} rows={len(df)} -> {output_path}")
            return

        if fallback_fn is not None:
            df_fallback = fallback_fn(seasons_to_fetch)
            if df_fallback is not None and isinstance(df_fallback, pd.DataFrame) and not df_fallback.empty:
                _validate_and_report(label, df_fallback)
                output_path = _merge_and_write(path, existing, df_fallback)
                print(f"[nflverse] downloaded {label} rows={len(df_fallback)} -> {output_path}")
                return

        if label.startswith("pfr_"):
            print(f"[nflverse] optional {label} unavailable for seasons {seasons_to_fetch}; continuing without it.")
        else:
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
        run_dataset("player stats", RAW_DIR / "nfl_player_stats.parquet", fetch_player_stats)

    if args.pfr:
        if "passing" in args.pfr:
            run_dataset("pfr_passing", RAW_DIR / "pfr_passing.parquet", fetch_pfr_passing, download_pfr_passing)
        if "rushing" in args.pfr:
            run_dataset("pfr_rushing", RAW_DIR / "pfr_rushing.parquet", fetch_pfr_rushing, download_pfr_rushing)
        if "receiving" in args.pfr:
            run_dataset("pfr_receiving", RAW_DIR / "pfr_receiving.parquet", fetch_pfr_receiving, download_pfr_receiving)


if __name__ == "__main__":
    main()
