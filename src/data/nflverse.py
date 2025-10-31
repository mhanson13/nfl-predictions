from __future__ import annotations
import argparse
from typing import List, Optional, Callable
from pathlib import Path

import pandas as pd

from datetime import datetime

from src.utils.io import RAW_DIR, write_df, read_df
from src.utils.logging import configure as configure_logging

SEASON_HISTORY_PATH = Path("data/reference/nfl_seasons_history.csv")


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
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(ignore_index=True)
    write_df(combined, path)


def _print_skip(dataset: str, seasons: List[int]) -> None:
    print(f"[nflverse] {dataset}: all requested seasons {seasons} already cached (finalized). Skipping fetch.")


def _try_import_nfl():
    try:
        import nfl_data_py as nfl  # type: ignore
        return nfl
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'nfl-data-py'. Install with: pip install nfl-data-py"
        ) from e


def fetch_pbp(seasons: List[int]) -> pd.DataFrame:
    nfl = _try_import_nfl()
    frames: List[pd.DataFrame] = []
    for yr in seasons:
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
    nfl = _try_import_nfl()
    df = nfl.import_schedules(seasons)
    # Normalize a few common columns
    if "game_id" not in df.columns and "game_id" in df.columns:
        pass
    if "season" not in df.columns and "season" in df.columns:
        pass
    return df


# ---- Additional nflverse datasets ----

def _fetch_generic(nfl, func_name: str, seasons: List[int]) -> Optional[pd.DataFrame]:
    fn: Optional[Callable] = getattr(nfl, func_name, None)
    if not callable(fn):
        print(f"[nflverse] function missing: {func_name}")
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
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_rosters", seasons)


def fetch_injuries(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_injuries", seasons)


def fetch_snap_counts(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_snap_counts", seasons)


def fetch_participation(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_participation", seasons)


def fetch_depth_charts(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_depth_charts", seasons)


def fetch_player_stats(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    # Try modern seasonal player stats; fall back to player stats
    for name in ("import_seasonal_player_stats", "import_player_stats"):
        df = _fetch_generic(nfl, name, seasons)
        if df is not None:
            return df
    return None


def fetch_pfr_passing(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_pfr_passing", seasons)


def fetch_pfr_rushing(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_pfr_rushing", seasons)


def fetch_pfr_receiving(seasons: List[int]) -> Optional[pd.DataFrame]:
    nfl = _try_import_nfl()
    return _fetch_generic(nfl, "import_pfr_receiving", seasons)


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
    return _download_release_dataset("injuries", ["injuries_{season}.csv"], seasons)


def download_participation(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("participation", ["participation_{season}.csv"], seasons)


def download_player_stats(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("player_stats", ["player_stats_{season}.csv"], seasons)


def download_pfr_passing(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("pfr", ["pfr_passing_{season}.csv"], seasons)


def download_pfr_rushing(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("pfr", ["pfr_rushing_{season}.csv"], seasons)


def download_pfr_receiving(seasons: List[int]) -> Optional[pd.DataFrame]:
    return _download_release_dataset("pfr", ["pfr_receiving_{season}.csv"], seasons)


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
        existing = _load_existing(path)
        seasons_to_fetch = _determine_seasons_to_fetch(existing, seasons, finalized_seasons, force=args.force_refresh)
        if not seasons_to_fetch:
            _print_skip(label, seasons)
            return

        df = fetch_fn(seasons_to_fetch)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            _merge_and_write(path, existing, df)
            print(f"[nflverse] saved {label} rows={len(df)} -> {path}")
            return

        if fallback_fn is not None:
            df_fallback = fallback_fn(seasons_to_fetch)
            if df_fallback is not None and isinstance(df_fallback, pd.DataFrame) and not df_fallback.empty:
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
