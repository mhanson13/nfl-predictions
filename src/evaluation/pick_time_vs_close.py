"""Pick-time vs closing-line report for the picks of record (market-gap plan, step 4).

Each pick of record carries the market at pick time (``market_*`` columns, The Odds API,
~2.5-3h before kickoff). The walk-forward benchmark scores the model against the nflverse
schedule lines instead (``home_moneyline`` / ``away_moneyline`` / ``spread_line``), which
nflverse keeps updating until kickoff, i.e. closing lines. This report puts the two side
by side for graded picks of record:

* how far the market moved between pick time and close (no-vig probability, spread);
* whether it moved toward the model's view (closing-line value of the model's lean);
* Brier / log loss / accuracy of the model, the pick-time market and the closing market,
  i.e. how much harder the walk-forward (close) benchmark is than the live one.

Writes ``predictions/evaluation/pick_time_vs_close_detail.csv`` and ``..._summary.csv``.
Read-only on everything else; not part of the weekly pipeline.

    python -m src.evaluation.pick_time_vs_close --season 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RECORDS_DIR = Path("predictions/picks_of_record")
SCHEDULE_PATH = Path("data/raw/nfl_schedules.parquet")
OUT_DIR = Path("predictions/evaluation")
EPS = 1e-6


def american_to_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.where(odds > 0, 100.0 / (odds + 100.0), -odds / (-odds + 100.0)), index=odds.index).where(
        odds.notna() & (odds != 0)
    )


def novig_home_prob(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    home, away = american_to_prob(home_ml), american_to_prob(away_ml)
    return home / (home + away)


def _log_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    p = p.clip(EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def build_detail(records: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per graded pick of record with pick-time and closing market side by side."""
    recs = records[records["status"].eq("locked") & pd.to_numeric(records["home_won"], errors="coerce").notna()]
    close = schedule[["game_id", "home_moneyline", "away_moneyline", "spread_line"]].drop_duplicates("game_id",
                                                                                                    keep="last")
    df = recs.merge(close, on="game_id", how="left")
    out = pd.DataFrame({
        "season": df["season"], "week": df["week"], "game_id": df["game_id"],
        "home_team": df["home_team"], "away_team": df["away_team"],
        "kickoff_utc": df["kickoff_utc"], "recorded_at": df["recorded_at"],
    })
    out["hours_before_kickoff"] = (
        (pd.to_datetime(df["kickoff_utc"], utc=True) - pd.to_datetime(df["recorded_at"], utc=True))
        .dt.total_seconds() / 3600.0
    ).round(2)
    out["home_won"] = pd.to_numeric(df["home_won"], errors="coerce")
    out["model_home_prob"] = pd.to_numeric(df["home_win_prob"], errors="coerce")
    out["pick_market_home_prob"] = pd.to_numeric(df["market_home_implied_prob_novig"], errors="coerce")
    out["close_market_home_prob"] = novig_home_prob(df["home_moneyline"], df["away_moneyline"])
    out["pick_home_spread"] = pd.to_numeric(df["market_home_spread"], errors="coerce")
    out["close_home_spread"] = -pd.to_numeric(df["spread_line"], errors="coerce")  # nflverse: home margin
    out["market_prob_move"] = out["close_market_home_prob"] - out["pick_market_home_prob"]
    out["spread_move"] = out["close_home_spread"] - out["pick_home_spread"]
    # The model's lean vs the pick-time market, and whether the close moved that way.
    lean = np.sign(out["model_home_prob"] - out["pick_market_home_prob"])
    out["model_lean"] = lean.map({1.0: "home", -1.0: "away", 0.0: "none"})
    out["clv_prob"] = out["market_prob_move"] * lean  # > 0: the market moved toward the model
    moved = out["market_prob_move"].abs() > EPS
    out["moved_toward_model"] = pd.Series(np.where(moved & (lean != 0), out["clv_prob"] > 0, np.nan),
                                          index=out.index)
    y = out["home_won"]
    for name in ("model", "pick_market", "close_market"):
        p = out[f"{name}_home_prob"]
        out[f"{name}_brier"] = (p - y) ** 2
        out[f"{name}_log_loss"] = _log_loss(p, y)
        out[f"{name}_correct"] = ((p > 0.5).astype(float) == y).where(p.notna() & (p != 0.5))
    out["close_available"] = out["close_market_home_prob"].notna()
    return out.sort_values(["week", "kickoff_utc", "game_id"]).reset_index(drop=True)


def _summary_row(scope: str, d: pd.DataFrame) -> dict:
    d = d[d["close_available"]]
    row: dict = {"scope": scope, "n_games": len(d)}
    if d.empty:
        return row
    row.update({
        "avg_hours_before_kickoff": d["hours_before_kickoff"].mean(),
        "avg_abs_market_prob_move": d["market_prob_move"].abs().mean(),
        "max_abs_market_prob_move": d["market_prob_move"].abs().max(),
        "avg_abs_spread_move": d["spread_move"].abs().mean(),
        "share_spread_moved": (d["spread_move"].abs() > 0).mean(),
        "n_moved_with_lean": int(d["moved_toward_model"].notna().sum()),
        "share_moved_toward_model": d["moved_toward_model"].dropna().astype(float).mean(),
        "avg_clv_prob": d["clv_prob"].mean(),
    })
    for name in ("model", "pick_market", "close_market"):
        row[f"{name}_brier"] = d[f"{name}_brier"].mean()
        row[f"{name}_log_loss"] = d[f"{name}_log_loss"].mean()
        row[f"{name}_accuracy"] = d[f"{name}_correct"].mean()
    row["model_minus_pick_market_brier"] = row["model_brier"] - row["pick_market_brier"]
    row["model_minus_close_market_brier"] = row["model_brier"] - row["close_market_brier"]
    row["close_minus_pick_market_brier"] = row["close_market_brier"] - row["pick_market_brier"]
    return row


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_row("all", detail)]
    rows += [_summary_row(f"week_{int(w):02d}", g) for w, g in detail.groupby("week")]
    return pd.DataFrame(rows)


def format_report(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    s = summary.iloc[0]
    if not s["n_games"]:
        return "pick-time vs close: no graded picks of record with a closing line"
    weeks = ", ".join(str(int(w)) for w in sorted(detail["week"].unique()))
    lines = [
        f"pick-time vs close: {int(s['n_games'])} graded picks of record (weeks {weeks}), "
        f"picked {s['avg_hours_before_kickoff']:.1f}h before kickoff on average",
        f"  market move to close: no-vig home prob {s['avg_abs_market_prob_move'] * 100:.1f} pts avg "
        f"(max {s['max_abs_market_prob_move'] * 100:.1f}); spread {s['avg_abs_spread_move']:.2f} pts avg, "
        f"moved in {s['share_spread_moved'] * 100:.0f}% of games",
        f"  toward the model's lean: {s['share_moved_toward_model'] * 100:.0f}% of {int(s['n_moved_with_lean'])} "
        f"moved games; avg CLV {s['avg_clv_prob'] * 100:+.2f} prob pts",
        f"  Brier  model {s['model_brier']:.4f} | pick-time market {s['pick_market_brier']:.4f} | "
        f"closing market {s['close_market_brier']:.4f}",
        f"  model - pick-time market {s['model_minus_pick_market_brier']:+.4f}; "
        f"model - closing market {s['model_minus_close_market_brier']:+.4f} (lower is better)",
    ]
    for _, r in summary.iloc[1:].iterrows():
        lines.append(f"  {r['scope']}: n={int(r['n_games'])} move {r['avg_abs_market_prob_move'] * 100:.1f} pts, "
                     f"toward model {r['share_moved_toward_model'] * 100:.0f}%, Brier model/pick/close "
                     f"{r['model_brier']:.4f}/{r['pick_market_brier']:.4f}/{r['close_market_brier']:.4f}")
    return "\n".join(lines)


def run(season: int, *, records_dir: Path = RECORDS_DIR, schedule_path: Path = SCHEDULE_PATH,
        out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(records_dir / f"picks_of_record_{season}.csv")
    schedule = pd.read_parquet(schedule_path)
    schedule = schedule[pd.to_numeric(schedule["season"], errors="coerce").eq(season)]
    detail = build_detail(records, schedule)
    summary = build_summary(detail)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "pick_time_vs_close_detail.csv", index=False, lineterminator="\n")
    summary.to_csv(out_dir / "pick_time_vs_close_summary.csv", index=False, lineterminator="\n")
    print(format_report(summary, detail))
    return detail, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    p.add_argument("--schedule", type=Path, default=SCHEDULE_PATH)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    run(a.season, records_dir=a.records_dir, schedule_path=a.schedule, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
