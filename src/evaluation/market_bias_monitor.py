"""Watch the signed gap between the published win probability and the no-vig market.

The anchored model learns a residual on top of the market. A *systematic*
residual (e.g. home teams marked down across a whole slate) is either real
signal or the model chasing a short-run pattern. This monitor makes the gap
visible every run and records whether it has been paying off:

* ``current_slate``  - mean(model - market) over the games being predicted;
* ``season`` / ``last_4_weeks`` / ``last_8_weeks`` - same on walk-forward history,
  with the actual home win rate vs the market and the model-minus-market Brier
  (negative = the residual helped).

Status: ``watch`` when the current slate's mean gap is at least
``SLATE_WATCH`` or a recent window's gap is significant (|t| >= 2);
``alert`` when a recent gap is significant *and* removing it would have
improved Brier by more than ``BRIER_TOLERANCE`` (``brier_cost_of_shift``).
``all_history`` and per-season rows are context only. Output: ``predictions/evaluation/market_bias_monitor.csv``.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HISTORY_GLOB = "predictions/history/*_predictions_history_*.csv"
FEATURES_PATH = Path("data/processed/matchup_features.parquet")
CURRENT_PATH = Path("predictions/predictions_full.csv")
OUTPUT_PATH = Path("predictions/evaluation/market_bias_monitor.csv")
SLATE_WATCH = 0.015
T_WATCH = 2.0
BRIER_TOLERANCE = 0.0005  # Brier the gap may cost vs the market before it is an alert


def _market(features: pd.DataFrame) -> pd.DataFrame:
    frame = features[["game_id", "sched_implied_prob_home", "sched_implied_prob_away"]].copy()
    home = pd.to_numeric(frame["sched_implied_prob_home"], errors="coerce")
    away = pd.to_numeric(frame["sched_implied_prob_away"], errors="coerce")
    frame["market_prob"] = (home / (home + away)).where((home > 0) & (away > 0))
    return frame[["game_id", "market_prob"]].drop_duplicates("game_id")


def _summarize(scope: str, frame: pd.DataFrame, graded: bool) -> dict:
    gap = frame["model_prob"] - frame["market_prob"]
    n = int(gap.notna().sum())
    row = {
        "scope": scope,
        "n_games": n,
        "mean_gap": float(gap.mean()) if n else np.nan,
        "mean_abs_gap": float(gap.abs().mean()) if n else np.nan,
        "share_home_below_market": float((gap < 0).mean()) if n else np.nan,
        "t_stat": float(gap.mean() / (gap.std(ddof=1) / np.sqrt(n))) if n > 2 and gap.std(ddof=1) > 0 else np.nan,
        "model_mean": float(frame["model_prob"].mean()) if n else np.nan,
        "market_mean": float(frame["market_prob"].mean()) if n else np.nan,
        "home_win_rate": np.nan,
        "brier_model_minus_market": np.nan,
        "brier_cost_of_shift": np.nan,
    }
    if graded and n:
        y = frame["home_win"].astype(float)
        row["home_win_rate"] = float(y.mean())
        row["brier_model_minus_market"] = float(
            ((frame["model_prob"] - y) ** 2).mean() - ((frame["market_prob"] - y) ** 2).mean()
        )
        # What the systematic shift itself cost: Brier of the model minus Brier of the
        # model with its mean gap removed (positive = the shift hurt).
        unshifted = (frame["model_prob"] - row["mean_gap"]).clip(0.01, 0.99)
        row["brier_cost_of_shift"] = float(((frame["model_prob"] - y) ** 2).mean() - ((unshifted - y) ** 2).mean())
    return row


def _status(row: dict, *, slate: bool) -> str:
    if not row["n_games"]:
        return "no_data"
    if slate:
        return "watch" if abs(row["mean_gap"]) >= SLATE_WATCH else "ok"
    significant = abs(row["t_stat"]) >= T_WATCH if np.isfinite(row["t_stat"]) else False
    if significant and row["brier_cost_of_shift"] > BRIER_TOLERANCE:
        return "alert"
    return "watch" if significant else "ok"


def build_monitor(
    history: pd.DataFrame,
    features: pd.DataFrame,
    current: Optional[pd.DataFrame],
    *,
    season: Optional[int] = None,
) -> pd.DataFrame:
    market = _market(features)
    hist = history.merge(market, on="game_id", how="inner")
    hist = hist.rename(columns={"home_win_prob": "model_prob"})
    hist = hist.dropna(subset=["model_prob", "market_prob", "home_score", "away_score"])
    hist["home_win"] = (hist["home_score"] > hist["away_score"]).astype(float)
    hist = hist[hist["home_score"] != hist["away_score"]]
    rows: list[dict] = []

    if current is not None and not current.empty:
        cur = current.merge(market, on="game_id", how="left").rename(columns={"home_win_prob": "model_prob"})
        row = _summarize("current_slate", cur, graded=False)
        row["season"], row["week"] = int(cur["season"].max()), int(cur["week"].max())
        row["status"] = _status(row, slate=True)
        rows.append(row)

    if not hist.empty:
        season = int(season or hist["season"].max())
        this = hist[hist["season"] == season]
        weeks = sorted(this["week"].unique())
        scopes = [("season", this)]
        for k in (4, 8):
            scopes.append((f"last_{k}_weeks", this[this["week"].isin(weeks[-k:])]))
        scopes.append(("all_history", hist))
        for name, frame in scopes:
            row = _summarize(name, frame, graded=True)
            row["season"] = season if name != "all_history" else np.nan
            row["week"] = int(frame["week"].max()) if len(frame) and name != "all_history" else np.nan
            row["status"] = "info" if name == "all_history" else _status(row, slate=False)
            rows.append(row)
        for past in sorted(hist["season"].unique()):
            row = _summarize(f"season_{int(past)}", hist[hist["season"] == past], graded=True)
            row["season"], row["week"] = int(past), np.nan
            row["status"] = "history"
            rows.append(row)

    cols = ["scope", "season", "week", "status", "n_games", "mean_gap", "mean_abs_gap", "share_home_below_market",
            "t_stat", "model_mean", "market_mean", "home_win_rate", "brier_model_minus_market", "brier_cost_of_shift"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _print(report: pd.DataFrame) -> None:
    print("[market_bias] model-minus-market home win probability (negative = home marked below the market)")
    for _, r in report[~report["scope"].str.startswith("season_")].iterrows():
        extra = ""
        if pd.notna(r["brier_model_minus_market"]):
            extra = (f" home_win={r['home_win_rate']:.3f} market={r['market_mean']:.3f}"
                     f" brier_vs_mkt={r['brier_model_minus_market']:+.4f} shift_cost={r['brier_cost_of_shift']:+.4f}")
        print(f"[market_bias] {r['scope']:<14} n={int(r['n_games']):>4} mean_gap={r['mean_gap']:+.4f}"
              f" below_mkt={r['share_home_below_market']:.0%} t={r['t_stat']:+.2f}{extra} status={r['status'].upper()}")
    flagged = report[report["status"].isin(["watch", "alert"])]
    if not flagged.empty:
        print(f"[market_bias] Warning: {', '.join(flagged['scope'])} flagged; see docs/market_bias_watch.md")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-glob", default=HISTORY_GLOB)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--current", type=Path, default=CURRENT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv or [])
    paths = sorted(glob.glob(args.history_glob))
    if not paths or not args.features.exists():
        print("[market_bias] history or features missing; skipping")
        return 0
    history = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    history = history.drop_duplicates("game_id", keep="last")
    features = pd.read_parquet(args.features, columns=["game_id", "sched_implied_prob_home", "sched_implied_prob_away"])
    current = pd.read_csv(args.current) if args.current.exists() else None
    report = build_monitor(history, features, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    _print(report)
    print(f"[market_bias] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
