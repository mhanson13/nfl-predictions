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

"""Compare walk-forward win probabilities to the no-vig market on identical games."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

DEFAULT_MARKET_REPORT_PATH = Path("predictions/evaluation/walk_forward_vs_market.csv")

PROBABILITY_SOURCES: dict[str, str] = {
    "model_raw": "home_win_prob_raw",
    "model_final": "home_win_prob",
    "market_novig": "market_home_prob_novig",
}

WEEK_BUCKETS = [(1, 4, "wk01-04"), (5, 8, "wk05-08"), (9, 13, "wk09-13"), (14, 18, "wk14-18"), (19, 99, "postseason")]


def no_vig_home_probability(features: pd.DataFrame) -> pd.Series:
    """Return the vig-free home win probability from per-side implied probabilities."""
    home = pd.to_numeric(features.get("sched_implied_prob_home"), errors="coerce")
    away = pd.to_numeric(features.get("sched_implied_prob_away"), errors="coerce")
    total = home + away
    prob = (home / total).where(total > 0)
    return prob.clip(1e-4, 1 - 1e-4)


def attach_market_probability(history: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach ``market_home_prob_novig`` to history rows by ``game_id``."""
    if "game_id" not in features.columns:
        out = history.copy()
        out["market_home_prob_novig"] = np.nan
        return out
    market = pd.DataFrame(
        {"game_id": features["game_id"], "market_home_prob_novig": no_vig_home_probability(features)}
    ).drop_duplicates("game_id", keep="last")
    return history.merge(market, on="game_id", how="left")


def _week_bucket(week: float) -> str:
    if pd.isna(week):
        return "unknown"
    for lo, hi, label in WEEK_BUCKETS:
        if lo <= int(week) <= hi:
            return label
    return "unknown"


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    out = {
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(np.mean((p >= 0.5).astype(int) == y)),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


def compare_to_market(history: pd.DataFrame) -> pd.DataFrame:
    """Score model and market probabilities on the identical decided games with a market line.

    Returns one row per (scope, segment, source) plus ``*_minus_market`` deltas for
    the model sources so lower Brier/LogLoss deltas and higher AUC deltas are wins.
    """
    frame = history.copy()
    margin = pd.to_numeric(frame.get("home_margin"), errors="coerce")
    frame["actual_home_win"] = np.where(margin > 0, 1, np.where(margin < 0, 0, np.nan))
    required = ["actual_home_win"] + [c for c in PROBABILITY_SOURCES.values() if c in frame.columns]
    if "market_home_prob_novig" not in frame.columns:
        return pd.DataFrame()
    frame = frame.dropna(subset=required)
    if frame.empty:
        return pd.DataFrame()
    frame["week_bucket"] = pd.to_numeric(frame.get("week"), errors="coerce").map(_week_bucket)

    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    for season, group in frame.groupby("season", sort=True):
        scopes.append(("season", str(int(season)), group))
    for bucket, group in frame.groupby("week_bucket", sort=True):
        scopes.append(("week_bucket", str(bucket), group))

    rows: list[dict[str, object]] = []
    for scope, segment, group in scopes:
        y = group["actual_home_win"].astype(int).to_numpy()
        market = _metrics(y, group["market_home_prob_novig"].to_numpy())
        for source, column in PROBABILITY_SOURCES.items():
            if column not in group.columns:
                continue
            values = _metrics(y, group[column].to_numpy())
            row: dict[str, object] = {"scope": scope, "segment": segment, "source": source, "n_games": int(len(group))}
            row.update(values)
            if source != "market_novig":
                row["auc_minus_market"] = values["auc"] - market["auc"]
                row["brier_minus_market"] = values["brier"] - market["brier"]
                row["logloss_minus_market"] = values["logloss"] - market["logloss"]
            rows.append(row)
    return pd.DataFrame(rows)


def write_market_report(report: pd.DataFrame, path: Path = DEFAULT_MARKET_REPORT_PATH) -> Path | None:
    if report.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return path


def format_overall_summary(report: pd.DataFrame) -> str:
    """One-line-per-source summary of the overall scope for logs."""
    overall = report[report["scope"].eq("overall")]
    lines = []
    for _, row in overall.iterrows():
        lines.append(
            f"{row['source']:<13} n={int(row['n_games'])} AUC={row['auc']:.4f} "
            f"Brier={row['brier']:.4f} LogLoss={row['logloss']:.4f}"
        )
    return "\n".join(lines)
