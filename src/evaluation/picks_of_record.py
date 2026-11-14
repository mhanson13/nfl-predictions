"""Per-game picks of record for live slates, built so they cannot leak.

Every live run appends one row per not-yet-started game to an append-only ledger
(``predictions/picks_of_record/picks_ledger_<season>.csv``). A game's pick of
record is the latest ledger row recorded at least ``lead`` before its kickoff
(default 2h, leaving time to publish); if no row made the deadline, the first row
recorded after it (``late``). This mirrors MTB's week archive
(mtb/lib/picks-of-record.ts), so the graded pick is the pick the site showed. A
row's ``recorded_at`` is the predictions file's write time -- the same timestamp
MTB uses when it imports the run. Rows are never rewritten, rows recorded
at/after kickoff are never written, and resolution only looks at rows recorded
before the lock deadline, so a locked pick cannot change after the fact.

Each row also stores the market line at pick time and a fingerprint of the
game's slow-moving features (everything except odds, injuries, news, weather).
Between a game's first and last pick only those pick-time inputs may change;
``static_features_changed`` flags anything else, which would indicate leakage
from games played earlier in the same week.

Usage:
    python -m src.evaluation.picks_of_record --record --season 2026 --week 10 \
        --run-started-at 2026-11-15T10:00:00+00:00
    python -m src.evaluation.picks_of_record --resolve --season 2026
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PREDICTIONS_PATH = Path("predictions/predictions.csv")
FEATURES_PATH = Path("data/processed/matchup_features.parquet")
SCHEDULE_PATH = Path("data/raw/nfl_schedules.parquet")
OUT_DIR = Path("predictions/picks_of_record")
DEFAULT_LEAD = timedelta(hours=2)
SLOT_ENV = "NFL_RUN_SLOT"

PICK_COLUMNS = [
    "home_win_prob",
    "home_pick",
    "home_confidence_pct",
    "pred_home_margin",
    "market_home_implied_prob_novig",
    "market_home_spread",
    "market_home_moneyline",
    "market_away_moneyline",
    "market_public_recommendation",
    # Grading context (src.evaluation.live_results) for the pick as published.
    "market_away_spread",
    "market_away_implied_prob_novig",
    "market_spread_edge_points",
    "market_spread_selected_side",
    "volatility_label",
]
# Inputs that legitimately change between a game's first and last pick.
PICK_TIME_MARKERS = (
    "spread", "moneyline", "implied", "total_line", "odds",
    "inj_", "qb_", "news_", "team_news_", "weather_", "wx_", "kickoff",
)
LEDGER_KEY = ["game_id", "run_id"]


def ledger_path(season: int, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"picks_ledger_{season}.csv"


def records_path(season: int, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"picks_of_record_{season}.csv"


def _utc(value) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def static_feature_hashes(features: pd.DataFrame, game_ids) -> pd.Series:
    """Fingerprint each game's slow-moving features (excludes pick-time inputs)."""
    rows = features[features["game_id"].astype(str).isin(set(map(str, game_ids)))]
    rows = rows.drop_duplicates("game_id", keep="last").set_index("game_id")
    cols = sorted(
        c for c in rows.columns
        if isinstance(c, str)
        and not any(m in c.lower() for m in PICK_TIME_MARKERS)
        and pd.api.types.is_numeric_dtype(rows[c])
    )
    values = rows[cols].astype(float).round(6)

    def _hash(row: pd.Series) -> str:
        payload = ",".join("nan" if pd.isna(v) else repr(float(v)) for v in row.to_numpy())
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    return values.apply(_hash, axis=1) if len(values) else pd.Series(dtype=str)


def build_ledger_rows(
    predictions: pd.DataFrame,
    *,
    recorded_at: datetime,
    run_id: str,
    slot: str,
    git_sha: str,
    feature_hashes: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Ledger rows for games that have not kicked off at ``recorded_at``."""
    preds = predictions.copy()
    preds["kickoff_utc"] = _utc(preds["kickoff"])
    upcoming = preds[preds["kickoff_utc"] > pd.Timestamp(recorded_at)].copy()
    if upcoming.empty:
        return pd.DataFrame()
    odds_cols = [c for c in upcoming.columns if c.startswith("odds_") and c.endswith("_last_update")]
    out = upcoming[["season", "week", "game_id", "home_team", "away_team", "kickoff_utc"]].copy()
    for col in PICK_COLUMNS:
        out[col] = upcoming[col] if col in upcoming.columns else np.nan
    if odds_cols:
        last = upcoming[odds_cols].apply(lambda s: _utc(s)).max(axis=1)
        out["odds_last_update"] = last.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        out["odds_last_update"] = None
    out["static_feature_hash"] = (
        out["game_id"].map(feature_hashes) if feature_hashes is not None else None
    )
    out["kickoff_utc"] = out["kickoff_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["recorded_at"] = pd.Timestamp(recorded_at).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["run_id"] = run_id
    out["slot"] = slot
    out["git_sha"] = git_sha
    return out.reset_index(drop=True)


def append_ledger(rows: pd.DataFrame, path: Path) -> int:
    """Append rows (append-only; a (game_id, run_id) pair is written at most once)."""
    if rows.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, usecols=LEDGER_KEY, dtype=str)
        seen = set(map(tuple, existing[LEDGER_KEY].to_numpy()))
        rows = rows[[tuple(k) not in seen for k in rows[LEDGER_KEY].astype(str).to_numpy()]]
        if rows.empty:
            return 0
        header = list(pd.read_csv(path, nrows=0).columns)
        rows = rows.reindex(columns=header + [c for c in rows.columns if c not in header])
        if len(rows.columns) != len(header):
            full = pd.concat([pd.read_csv(path), rows], ignore_index=True)
            full.to_csv(path, index=False, lineterminator="\n")
            return len(rows)
        rows.to_csv(path, mode="a", header=False, index=False, lineterminator="\n")
    else:
        rows.to_csv(path, index=False, lineterminator="\n")
    return len(rows)


def resolve_records(
    ledger: pd.DataFrame,
    *,
    now: datetime,
    lead: timedelta = DEFAULT_LEAD,
    schedule: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per game: the pick of record and its lock status.

    status: ``locked`` (lock deadline passed; latest row recorded by the deadline),
    ``late`` (no row by the deadline; the first row after it, as MTB shows), ``provisional``
    (deadline not reached yet), ``missed`` is implied by absence.
    """
    if ledger.empty:
        return pd.DataFrame()
    led = ledger.copy()
    led["kickoff_ts"] = _utc(led["kickoff_utc"])
    led["recorded_ts"] = _utc(led["recorded_at"])
    led = led[led["recorded_ts"] < led["kickoff_ts"]]  # defensive: never use post-kickoff rows
    now_ts = pd.Timestamp(now)
    out = []
    for game_id, rows in led.sort_values("recorded_ts").groupby("game_id", sort=False):
        kickoff = rows["kickoff_ts"].iloc[-1]
        deadline = kickoff - lead
        on_time = rows[rows["recorded_ts"] <= deadline]
        if now_ts < deadline:
            pick, status = rows.iloc[-1], "provisional"
        elif not on_time.empty:
            pick, status = on_time.iloc[-1], "locked"
        else:
            pick, status = rows.iloc[0], "late"
        rec = pick.drop(labels=["kickoff_ts", "recorded_ts"]).to_dict()
        rec["status"] = status
        rec["lock_deadline_utc"] = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
        rec["n_versions"] = int(len(rows))
        rec["first_recorded_at"] = rows["recorded_at"].iloc[0]
        first = rows.iloc[0]
        rec["prob_change_since_first"] = float(pick["home_win_prob"]) - float(first["home_win_prob"])
        hashes = rows["static_feature_hash"].dropna().astype(str) if "static_feature_hash" in rows else pd.Series([], dtype=str)
        hashes = hashes[hashes.str.len() > 0]
        rec["static_features_changed"] = bool(hashes.nunique() > 1)
        out.append(rec)
    records = pd.DataFrame(out)
    if schedule is not None and not records.empty:
        records = attach_results(records, schedule)
    return records.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)


def attach_results(records: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Add final scores and per-game Brier for the pick and the pick-time market."""
    cols = [c for c in ("game_id", "home_score", "away_score") if c in schedule.columns]
    if len(cols) < 3:
        return records
    out = records.drop(columns=[c for c in ("home_score", "away_score") if c in records.columns])
    out = out.merge(schedule[cols].drop_duplicates("game_id"), on="game_id", how="left")
    margin = out["home_score"] - out["away_score"]
    y = np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5))
    y = np.where(margin.isna(), np.nan, y)
    p = pd.to_numeric(out["home_win_prob"], errors="coerce")
    m = pd.to_numeric(out["market_home_implied_prob_novig"], errors="coerce")
    out["home_won"] = y
    out["pick_correct"] = np.where(np.isnan(y) | (y == 0.5), np.nan, ((p > 0.5) == (y == 1.0)).astype(float))
    out["brier"] = (p - y) ** 2
    out["market_brier"] = (m - y) ** 2
    return out


def summarize(records: pd.DataFrame) -> str:
    if records.empty:
        return "[picks_of_record] no records"
    counts = records["status"].value_counts().to_dict()
    msg = f"[picks_of_record] games={len(records)} status={counts}"
    scored = records.dropna(subset=["brier", "market_brier"]) if "brier" in records else pd.DataFrame()
    scored = scored[scored["status"].isin(["locked", "late"])] if not scored.empty else scored
    if len(scored):
        msg += (
            f" | scored={len(scored)} acc={scored['pick_correct'].mean():.3f}"
            f" brier={scored['brier'].mean():.4f} market_brier={scored['market_brier'].mean():.4f}"
        )
    changed = int(records.get("static_features_changed", pd.Series(dtype=bool)).sum())
    if changed:
        ids = records.loc[records["static_features_changed"], "game_id"].tolist()
        msg += f"\n[picks_of_record] Warning: non-pick-time features changed between runs for {changed} games: {ids}"
    return msg


def _read_schedule(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def record(season: int, week: Optional[int], run_started_at: datetime, *, now: Optional[datetime] = None,
           predictions_path: Path = PREDICTIONS_PATH, features_path: Path = FEATURES_PATH,
           out_dir: Path = OUT_DIR, slot: Optional[str] = None) -> int:
    mtime = datetime.fromtimestamp(predictions_path.stat().st_mtime, tz=timezone.utc)
    if mtime < run_started_at:
        raise RuntimeError(
            f"{predictions_path} was written at {mtime:%Y-%m-%dT%H:%M:%SZ}, before this run started "
            f"({run_started_at:%Y-%m-%dT%H:%M:%SZ}); refusing to record stale predictions."
        )
    # Stamp rows with when the predictions were produced (file write time), the same
    # timestamp MTB's import uses, so both sides lock the same run.
    recorded_at = now or mtime
    preds = pd.read_csv(predictions_path)
    preds = preds[preds["season"] == season]
    if week is not None:
        preds = preds[preds["week"] == week]
    hashes = None
    try:
        hashes = static_feature_hashes(pd.read_parquet(features_path), preds["game_id"])
    except Exception as exc:  # fingerprint is a guard, not a requirement
        print(f"[picks_of_record] feature fingerprint unavailable: {exc}")
    rows = build_ledger_rows(
        preds,
        recorded_at=recorded_at,
        run_id=run_started_at.strftime("%Y%m%dT%H%M%SZ"),
        slot=slot or os.environ.get(SLOT_ENV, "manual"),
        git_sha=_git_sha(),
        feature_hashes=hashes,
    )
    skipped = len(preds) - len(rows)
    n = append_ledger(rows, ledger_path(season, out_dir))
    print(f"[picks_of_record] recorded {n} games ({skipped} already kicked off) -> {ledger_path(season, out_dir)}")
    return n


def resolve(season: int, *, now: Optional[datetime] = None, out_dir: Path = OUT_DIR,
            lead: timedelta = DEFAULT_LEAD, schedule_path: Path = SCHEDULE_PATH) -> pd.DataFrame:
    path = ledger_path(season, out_dir)
    if not path.exists():
        print(f"[picks_of_record] no ledger at {path}")
        return pd.DataFrame()
    records = resolve_records(
        pd.read_csv(path), now=now or datetime.now(timezone.utc), lead=lead, schedule=_read_schedule(schedule_path)
    )
    dest = records_path(season, out_dir)
    records.to_csv(dest, index=False, lineterminator="\n")
    print(summarize(records))
    print(f"[picks_of_record] wrote {len(records)} records -> {dest}")
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int)
    parser.add_argument("--record", action="store_true", help="Append this run's upcoming picks to the ledger.")
    parser.add_argument("--resolve", action="store_true", help="Rebuild picks_of_record_<season>.csv.")
    parser.add_argument("--run-started-at", help="ISO time the pipeline run started (stale-file guard).")
    parser.add_argument("--lead-hours", type=float, default=DEFAULT_LEAD.total_seconds() / 3600)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.record:
        started = _utc(args.run_started_at) if args.run_started_at else None
        if started is None or pd.isna(started):
            parser.error("--record requires --run-started-at")
        record(args.season, args.week, started.to_pydatetime(), out_dir=args.out_dir)
    if args.record or args.resolve:
        resolve(args.season, out_dir=args.out_dir, lead=timedelta(hours=args.lead_hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
