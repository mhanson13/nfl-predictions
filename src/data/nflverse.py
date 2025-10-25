from __future__ import annotations
import argparse
from typing import List, Optional, Callable
from pathlib import Path

import pandas as pd

from src.utils.io import RAW_DIR, write_df
from src.utils.logging import configure as configure_logging


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
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    args = ap.parse_args()

    configure_logging(args.debug)

    seasons = [int(s) for s in args.season]

    if args.pbp:
        pbp = fetch_pbp(seasons)
        if pbp is not None and not pbp.empty:
            write_df(pbp, RAW_DIR / "nfl_pbp.parquet")
            print(f"[nflverse] saved PBP rows={len(pbp)} -> {RAW_DIR / 'nfl_pbp.parquet'}")
        else:
            print("[nflverse] no PBP returned")

    if args.schedules:
        sched = fetch_schedules(seasons)
        if sched is not None and not sched.empty:
            write_df(sched, RAW_DIR / "nfl_schedules.parquet")
            print(f"[nflverse] saved schedules rows={len(sched)} -> {RAW_DIR / 'nfl_schedules.parquet'}")
        else:
            print("[nflverse] no schedules returned")

    if args.rosters:
        df = fetch_rosters(seasons)
        if df is not None and not df.empty:
            write_df(df, RAW_DIR / "nfl_rosters.parquet")
            print(f"[nflverse] saved rosters rows={len(df)}")
        else:
            dfd = download_rosters(seasons)
            if dfd is not None and not dfd.empty:
                write_df(dfd, RAW_DIR / "nfl_rosters.parquet")
                print(f"[nflverse] downloaded rosters rows={len(dfd)}")

    if args.injuries:
        df = fetch_injuries(seasons)
        if df is not None and not df.empty:
            write_df(df, RAW_DIR / "nfl_injuries.parquet")
            print(f"[nflverse] saved injuries rows={len(df)}")
        else:
            dfd = download_injuries(seasons)
            if dfd is not None and not dfd.empty:
                write_df(dfd, RAW_DIR / "nfl_injuries.parquet")
                print(f"[nflverse] downloaded injuries rows={len(dfd)}")

    if args.snaps:
        df = fetch_snap_counts(seasons)
        if df is not None and not df.empty:
            write_df(df, RAW_DIR / "nfl_snap_counts.parquet")
            print(f"[nflverse] saved snap counts rows={len(df)}")

    if args.participation:
        df = fetch_participation(seasons)
        if df is not None and not df.empty:
            write_df(df, RAW_DIR / "nfl_participation.parquet")
            print(f"[nflverse] saved participation rows={len(df)}")
        else:
            dfd = download_participation(seasons)
            if dfd is not None and not dfd.empty:
                write_df(dfd, RAW_DIR / "nfl_participation.parquet")
                print(f"[nflverse] downloaded participation rows={len(dfd)}")

    if args.depthcharts:
        df = fetch_depth_charts(seasons)
        if df is not None and not df.empty:
            write_df(df, RAW_DIR / "nfl_depth_charts.parquet")
            print(f"[nflverse] saved depth charts rows={len(df)}")

    if args.players:
        df = fetch_player_stats(seasons)
        if df is not None and not df.empty:
            write_df(df, RAW_DIR / "nfl_player_stats.parquet")
            print(f"[nflverse] saved player stats rows={len(df)}")
        else:
            dfd = download_player_stats(seasons)
            if dfd is not None and not dfd.empty:
                write_df(dfd, RAW_DIR / "nfl_player_stats.parquet")
                print(f"[nflverse] downloaded player stats rows={len(dfd)}")

    if args.pfr:
        if "passing" in args.pfr:
            df = fetch_pfr_passing(seasons)
            if df is not None and not df.empty:
                write_df(df, RAW_DIR / "pfr_passing.parquet")
                print(f"[nflverse] saved PFR passing rows={len(df)}")
            else:
                dfd = download_pfr_passing(seasons)
                if dfd is not None and not dfd.empty:
                    write_df(dfd, RAW_DIR / "pfr_passing.parquet")
                    print(f"[nflverse] downloaded PFR passing rows={len(dfd)}")
        if "rushing" in args.pfr:
            df = fetch_pfr_rushing(seasons)
            if df is not None and not df.empty:
                write_df(df, RAW_DIR / "pfr_rushing.parquet")
                print(f"[nflverse] saved PFR rushing rows={len(df)}")
            else:
                dfd = download_pfr_rushing(seasons)
                if dfd is not None and not dfd.empty:
                    write_df(dfd, RAW_DIR / "pfr_rushing.parquet")
                    print(f"[nflverse] downloaded PFR rushing rows={len(dfd)}")
        if "receiving" in args.pfr:
            df = fetch_pfr_receiving(seasons)
            if df is not None and not df.empty:
                write_df(df, RAW_DIR / "pfr_receiving.parquet")
                print(f"[nflverse] saved PFR receiving rows={len(df)}")
            else:
                dfd = download_pfr_receiving(seasons)
                if dfd is not None and not dfd.empty:
                    write_df(dfd, RAW_DIR / "pfr_receiving.parquet")
                    print(f"[nflverse] downloaded PFR receiving rows={len(dfd)}")


if __name__ == "__main__":
    main()
