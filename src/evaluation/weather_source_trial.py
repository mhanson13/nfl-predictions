"""Xweather trial: compare weather sources on the same games.

Two parts, both written under ``analysis/weather_trial/``:

1. **History** (``--history N``, run once): sample N outdoor, non-neutral games
   from 2017-2024 that Visual Crossing (VX) covers with observations, fetch the
   same kickoff +/-3h window from Xweather's archive, and report agreement
   (MAE, bias, correlation, precipitation yes/no agreement) ->
   ``history_detail.csv`` / ``history_summary.csv``.
2. **Live forecasts** (``--live``, every run): snapshot the current slate's
   forecasts from Xweather, VX and Tomorrow.io into ``forecast_snapshots.csv``.
   Once those games are played and VX holds observations, each snapshot is
   scored against them -> ``forecast_skill.csv``. Over a few weeks this measures
   which forecast is closest to what actually happened at kickoff.

Spend is capped by ``--max-requests`` (default 250); raw Xweather responses are
cached, so re-running never pays twice for the same game window.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data import visualcrossing as vx
from src.data.xweather import (
    BudgetExhausted,
    XweatherClient,
    fetch_game_window,
    get_xweather_credentials,
)
from src.utils.io import PROC_DIR, RAW_DIR, REF_DIR, read_df

OUT_DIR = Path("analysis/weather_trial")
SNAPSHOTS = OUT_DIR / "forecast_snapshots.csv"
# (label, xweather column, VX column, Tomorrow column)
METRICS = [
    ("temp_kickoff_f", "xw_temp_kickoff_f", "weather_temp_kickoff_f", "weather_temp_kickoff_f"),
    ("temp_window_f", "xw_temp_f", "weather_temp_f", "weather_temp_f"),
    ("wind_mph", "xw_wind_mph", "weather_wind_mph", "weather_wind_mph"),
    ("gust_mph", "xw_windgust_mph", "weather_windgust_mph", "weather_windgust_mph"),
    ("precip_in_total", "xw_precip_in_total", "weather_precip_in_total", "weather_precip_in_total"),
]
OUTDOOR_ROOFS = {"outdoors", "open"}


def _games(seasons: list[int]) -> pd.DataFrame:
    sched = read_df(RAW_DIR / "nfl_schedules.parquet")
    sched = sched[pd.to_numeric(sched["season"], errors="coerce").isin(seasons)].copy()
    sched = vx._resolve_game_times(sched, pd.read_csv(REF_DIR / "team_locations.csv"))
    sched["game_id"] = sched["game_id"].astype(str)
    outdoor = sched["roof"].astype(str).str.lower().isin(OUTDOOR_ROOFS)
    neutral = sched.get("location", pd.Series("", index=sched.index)).astype(str).str.lower().eq("neutral")
    return sched[outdoor & ~neutral].dropna(subset=["home_lat", "home_lon", "start_utc"])


def _compare(frame: pd.DataFrame, a_prefix_cols: list[str], b_cols: list[str], labels: list[str], source: str) -> list[dict]:
    rows = []
    for label, a, b in zip(labels, a_prefix_cols, b_cols):
        if a not in frame or b not in frame:
            continue
        x = pd.to_numeric(frame[a], errors="coerce")
        y = pd.to_numeric(frame[b], errors="coerce")
        ok = x.notna() & y.notna()
        if not ok.any():
            continue
        diff = x[ok] - y[ok]
        rows.append(
            {
                "source": source,
                "metric": label,
                "n": int(ok.sum()),
                "mae": float(diff.abs().mean()),
                "bias": float(diff.mean()),
                "corr": float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan,
            }
        )
    if "precip_in_total" in labels:
        i = labels.index("precip_in_total")
        a, b = a_prefix_cols[i], b_cols[i]
        if a in frame and b in frame:
            x = pd.to_numeric(frame[a], errors="coerce")
            y = pd.to_numeric(frame[b], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.any():
                rows.append(
                    {
                        "source": source,
                        "metric": "precip_yes_no_agreement",
                        "n": int(ok.sum()),
                        "mae": np.nan,
                        "bias": float((x[ok] > 0.01).mean() - (y[ok] > 0.01).mean()),
                        "corr": float(((x[ok] > 0.01) == (y[ok] > 0.01)).mean()),
                    }
                )
    return rows


def run_history(client: XweatherClient, n: int, seed: int = 42) -> pd.DataFrame:
    games = _games(list(range(2017, 2025)))
    cache = read_df(vx.CACHE_PATH)
    cache = cache[cache["vx_data_source"].astype(str).str.lower().eq("obs")]
    games = games[games["game_id"].isin(set(cache["game_id"].astype(str)))]
    # spread the sample over seasons and months so cold/windy games are represented
    games["month"] = pd.to_datetime(games["start_utc"], utc=True).dt.month
    per_group = max(1, -(-n // max(1, games.groupby(["season", "month"]).ngroups)))
    sample = (
        games.sample(frac=1.0, random_state=seed)
        .groupby(["season", "month"], sort=False)
        .head(per_group)
        .head(n)
    )
    print(f"[weather_trial] history: {len(sample)} outdoor games sampled from {len(games)} with VX observations")
    rows = []
    # Like-for-like source comparison: request Xweather at exactly the place and time
    # Visual Crossing was fetched for (its stored coordinates and window centre).
    vx_at = cache.drop_duplicates("game_id").set_index("game_id")
    for _, g in sample.iterrows():
        ref = vx_at.loc[g["game_id"]]
        lat = float(ref["vx_latitude"]) if pd.notna(ref.get("vx_latitude")) else float(g["home_lat"])
        lon = float(ref["vx_longitude"]) if pd.notna(ref.get("vx_longitude")) else float(g["home_lon"])
        centre = pd.Timestamp(ref["start_utc"]) if pd.notna(ref.get("start_utc")) else pd.Timestamp(g["start_utc"])
        if centre.tzinfo is None:
            centre = centre.tz_localize("UTC")
        try:
            agg = fetch_game_window(client, g["game_id"], lat, lon, centre.to_pydatetime())
        except BudgetExhausted as exc:
            print(f"[weather_trial] {exc}; stopping history fetch")
            break
        if agg:
            rows.append(agg)
    xw = pd.DataFrame(rows)
    if xw.empty:
        return xw
    detail = xw.merge(cache.drop_duplicates("game_id"), on="game_id", how="left")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT_DIR / "history_detail.csv", index=False)
    labels = [m[0] for m in METRICS]
    summary = pd.DataFrame(_compare(detail, [m[1] for m in METRICS], [m[2] for m in METRICS], labels, "xweather_vs_vx_obs"))
    summary.to_csv(OUT_DIR / "history_summary.csv", index=False)
    return summary


def snapshot_live(client: Optional[XweatherClient], season: int, week: int) -> pd.DataFrame:
    """Record each source's current forecast for the not-yet-started games of the slate."""
    now = datetime.now(timezone.utc)
    games = _games([season])
    games = games[(pd.to_numeric(games["week"]) == week) & (pd.to_datetime(games["start_utc"], utc=True) > now)]
    rows: list[dict] = []
    stamp = now.isoformat(timespec="minutes")
    tag = now.strftime("%Y%m%dT%H")
    vx_cache = read_df(vx.CACHE_PATH).drop_duplicates("game_id").set_index("game_id")
    tm_path = PROC_DIR / "weather_games.parquet"
    tm = read_df(tm_path).drop_duplicates("game_id").set_index("game_id") if tm_path.exists() else pd.DataFrame()
    for _, g in games.iterrows():
        gid = g["game_id"]
        base = {"game_id": gid, "season": season, "week": week, "snapshot_utc": stamp,
                "kickoff_utc": pd.Timestamp(g["start_utc"]).isoformat(),
                "lead_hours": round((pd.Timestamp(g["start_utc"]) - pd.Timestamp(now)).total_seconds() / 3600, 1)}
        if client is not None:
            try:
                agg = fetch_game_window(client, gid, float(g["home_lat"]), float(g["home_lon"]),
                                        g["start_utc"].to_pydatetime(), cache_tag=tag)
            except BudgetExhausted as exc:
                print(f"[weather_trial] {exc}; skipping remaining Xweather snapshots")
                client = None
                agg = {}
            if agg:
                rows.append({**base, "source": "xweather", **{m[0]: agg.get(m[1]) for m in METRICS}})
        for source, table, col_idx in (("visualcrossing", vx_cache, 2), ("tomorrow", tm, 3)):
            if gid in getattr(table, "index", []):
                rec = table.loc[gid]
                rows.append({**base, "source": source,
                             **{m[0]: pd.to_numeric(rec.get(m[col_idx]), errors="coerce") for m in METRICS}})
    snap = pd.DataFrame(rows)
    if not snap.empty:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        snap.to_csv(SNAPSHOTS, mode="a", header=not SNAPSHOTS.exists(), index=False)
    print(f"[weather_trial] live: snapshot {len(snap)} rows for {games['game_id'].nunique()} upcoming outdoor games")
    return snap


def score_snapshots() -> pd.DataFrame:
    """Score stored forecast snapshots against VX observations for games now played."""
    if not SNAPSHOTS.exists():
        return pd.DataFrame()
    snaps = pd.read_csv(SNAPSHOTS)
    obs = read_df(vx.CACHE_PATH)
    obs = obs[obs["vx_data_source"].astype(str).str.lower().eq("obs")].drop_duplicates("game_id")
    obs = obs.rename(columns={m[2]: f"obs_{m[0]}" for m in METRICS})
    obs = obs[["game_id", *[c for c in (f"obs_{m[0]}" for m in METRICS) if c in obs.columns]]]
    merged = snaps.merge(obs, on="game_id", how="inner")
    if merged.empty:
        print("[weather_trial] skill: no snapshot games have observations yet")
        return pd.DataFrame()
    merged["lead_bucket"] = pd.cut(merged["lead_hours"], [0, 24, 72, 168, 400], labels=["<1d", "1-3d", "3-7d", "7d+"])
    rows = []
    labels = [m[0] for m in METRICS]
    for (source, bucket), frame in merged.groupby(["source", "lead_bucket"], observed=True):
        for row in _compare(frame, labels, [f"obs_{l}" for l in labels], labels, source):
            row["lead_bucket"] = bucket
            rows.append(row)
    skill = pd.DataFrame(rows)
    skill.to_csv(OUT_DIR / "forecast_skill.csv", index=False)
    return skill


def _print(title: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    print(f"[weather_trial] {title}")
    print(frame.round(3).to_string(index=False))


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return _main(argv)
    except Exception as exc:  # the trial must never fail the pipeline
        print(f"[weather_trial] Warning: trial failed ({type(exc).__name__}: {exc}); continuing")
        return 0


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=int, default=0, help="Sample size for the historical comparison (0 = skip).")
    parser.add_argument("--live", action="store_true", help="Snapshot current-slate forecasts from all sources.")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--max-requests", type=int, default=250)
    args = parser.parse_args(argv)

    client_id, client_secret = get_xweather_credentials()
    client = XweatherClient(client_id, client_secret, max_requests=args.max_requests) if client_id and client_secret else None
    if client is None:
        print("[weather_trial] Xweather credentials not found (XWEATHER_CLIENT_API_KEY / XWEATHER_CLIENT_API_SECRET); "
              "Xweather rows skipped")

    if args.history and client is not None:
        _print("history: Xweather archive vs Visual Crossing observations", run_history(client, args.history))
    if args.live:
        season, week = args.season, args.week
        if season is None or week is None:
            from src.utils.nfl_calendar import load_week_context

            ctx = load_week_context()
            season, week = season or ctx.season, week or ctx.week
        if season is not None and week is not None:
            snapshot_live(client, int(season), int(week))
    _print("forecast skill vs observed (by lead time)", score_snapshots())
    if client is not None:
        remaining = {k: v for k, v in client.last_rate_headers.items() if "period" in k.lower()}
        print(f"[weather_trial] Xweather requests this run: {client.requests_made}; quota: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
