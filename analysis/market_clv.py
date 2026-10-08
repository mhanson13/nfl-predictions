from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PICK_SNAPSHOT_PATH = Path("predictions/predictions.csv")
DEFAULT_CLOSE_SOURCE_PATH = Path("data/processed/matchup_features.parquet")
DEFAULT_DETAIL_PATH = ANALYSIS_DIR / "market_clv_detail.csv"
DEFAULT_SUMMARY_PATH = ANALYSIS_DIR / "market_clv_summary.csv"


DETAIL_COLUMNS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "model_pick_side",
    "model_pick_team",
    "model_pick_prob",
    "moneyline_favorite_disagreement",
    "moneyline_policy_recommendation_flag",
    "moneyline_policy_status",
    "spread_policy_recommendation_flag",
    "spread_policy_status",
    "policy_primary_signal",
    "policy_tier",
    "policy_status",
    "pick_home_moneyline",
    "pick_away_moneyline",
    "close_home_moneyline",
    "close_away_moneyline",
    "model_side_pick_moneyline",
    "model_side_close_moneyline",
    "model_side_pick_decimal",
    "model_side_close_decimal",
    "model_side_pick_implied_prob_novig",
    "model_side_close_implied_prob_novig",
    "moneyline_clv_decimal_delta",
    "moneyline_clv_prob_delta",
    "moneyline_positive_clv",
    "model_vs_pick_prob_edge",
    "model_vs_close_prob_edge",
    "pick_selected_spread",
    "close_selected_spread",
    "spread_clv_points",
    "spread_positive_clv",
    "clv_status",
]


SUMMARY_COLUMNS = [
    "scope",
    "segment",
    "threshold",
    "n_games",
    "moneyline_clv_available",
    "avg_moneyline_clv_decimal_delta",
    "avg_moneyline_clv_prob_delta",
    "positive_moneyline_clv_rate",
    "avg_model_vs_close_prob_edge",
    "spread_clv_available",
    "avg_spread_clv_points",
    "positive_spread_clv_rate",
]


def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0


def american_to_implied_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path, columns=columns, engine="fastparquet")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _string_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _novig_side_prob(home_odds: pd.Series, away_odds: pd.Series, home_side: pd.Series) -> pd.Series:
    home_implied = home_odds.apply(american_to_implied_prob)
    away_implied = away_odds.apply(american_to_implied_prob)
    implied_sum = home_implied + away_implied
    home_novig = np.where(implied_sum > 0.0, home_implied / implied_sum, np.nan)
    away_novig = np.where(implied_sum > 0.0, away_implied / implied_sum, np.nan)
    return pd.Series(np.where(home_side, home_novig, away_novig), index=home_odds.index, dtype=float)


def load_pick_snapshot(path: Path, *, season: int | None = None, week: int | None = None) -> pd.DataFrame:
    picks = _read_csv_if_exists(path)
    if picks.empty:
        return picks
    if season is not None and "season" in picks.columns:
        picks = picks[pd.to_numeric(picks["season"], errors="coerce").eq(int(season))]
    if week is not None and "week" in picks.columns:
        picks = picks[pd.to_numeric(picks["week"], errors="coerce").eq(int(week))]
    return picks.copy()


def load_close_snapshot(path: Path, *, season: int | None = None, week: int | None = None) -> pd.DataFrame:
    columns = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_moneyline",
        "away_moneyline",
        "spread_line",
    ]
    close = _read_parquet(path, columns=[col for col in columns])
    if close.empty:
        return close
    if season is not None and "season" in close.columns:
        close = close[pd.to_numeric(close["season"], errors="coerce").eq(int(season))]
    if week is not None and "week" in close.columns:
        close = close[pd.to_numeric(close["week"], errors="coerce").eq(int(week))]
    return close.drop_duplicates("game_id", keep="last").copy()


def build_market_clv_detail(picks: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    if "game_id" not in picks.columns:
        raise ValueError("pick snapshot must include game_id")
    if close.empty or "game_id" not in close.columns:
        merged = picks.copy()
        for col in ["home_moneyline", "away_moneyline", "spread_line"]:
            merged[f"{col}_close"] = np.nan
    else:
        close_cols = [
            col
            for col in ["game_id", "home_moneyline", "away_moneyline", "spread_line"]
            if col in close.columns
        ]
        merged = picks.merge(close[close_cols], on="game_id", how="left", suffixes=("", "_close"))

    home_prob = _numeric(merged, "home_win_prob")
    model_home_pick = home_prob >= 0.5
    away_prob = 1.0 - home_prob
    model_pick_prob = pd.Series(np.where(model_home_pick, home_prob, away_prob), index=merged.index, dtype=float)

    pick_home_ml = _numeric(merged, "market_home_moneyline")
    pick_away_ml = _numeric(merged, "market_away_moneyline")
    close_home_ml = _numeric(merged, "home_moneyline")
    close_away_ml = _numeric(merged, "away_moneyline")

    model_side_pick_ml = pd.Series(np.where(model_home_pick, pick_home_ml, pick_away_ml), index=merged.index, dtype=float)
    model_side_close_ml = pd.Series(np.where(model_home_pick, close_home_ml, close_away_ml), index=merged.index, dtype=float)
    pick_decimal = model_side_pick_ml.apply(american_to_decimal)
    close_decimal = model_side_close_ml.apply(american_to_decimal)
    pick_side_novig = _novig_side_prob(pick_home_ml, pick_away_ml, model_home_pick)
    close_side_novig = _novig_side_prob(close_home_ml, close_away_ml, model_home_pick)

    moneyline_decimal_delta = pick_decimal - close_decimal
    moneyline_prob_delta = close_side_novig - pick_side_novig

    pick_home_spread = _numeric(merged, "market_home_spread")
    pick_away_spread = _numeric(merged, "market_away_spread")
    close_market_home_margin = _numeric(merged, "spread_line")
    close_home_spread = -close_market_home_margin
    close_away_spread = close_market_home_margin
    pick_selected_spread = pd.Series(
        np.where(model_home_pick, pick_home_spread, pick_away_spread),
        index=merged.index,
        dtype=float,
    )
    close_selected_spread = pd.Series(
        np.where(model_home_pick, close_home_spread, close_away_spread),
        index=merged.index,
        dtype=float,
    )
    spread_clv_points = pick_selected_spread - close_selected_spread

    moneyline_available = model_side_pick_ml.notna() & model_side_close_ml.notna()
    spread_available = pick_selected_spread.notna() & close_selected_spread.notna()
    movement = (
        moneyline_available
        & moneyline_decimal_delta.abs().gt(1e-9)
    ) | (
        spread_available
        & spread_clv_points.abs().gt(1e-9)
    )
    clv_status = np.select(
        [
            moneyline_available | spread_available,
            ~(moneyline_available | spread_available),
        ],
        [
            "movement_available",
            "unavailable_missing_pick_or_close",
        ],
        default="unavailable_missing_pick_or_close",
    )
    clv_status = pd.Series(clv_status, index=merged.index, dtype=object)
    clv_status.loc[(moneyline_available | spread_available) & ~movement] = "single_snapshot_or_no_movement"

    detail = pd.DataFrame(
        {
            "season": _numeric(merged, "season").astype("Int64"),
            "week": _numeric(merged, "week").astype("Int64"),
            "game_id": _string_series(merged, "game_id"),
            "home_team": _string_series(merged, "home_team"),
            "away_team": _string_series(merged, "away_team"),
            "model_pick_side": np.where(model_home_pick, "home", "away"),
            "model_pick_team": np.where(
                model_home_pick,
                _string_series(merged, "home_team"),
                _string_series(merged, "away_team"),
            ),
            "model_pick_prob": model_pick_prob,
            "moneyline_favorite_disagreement": _bool_series(merged, "market_moneyline_favorite_disagreement"),
            "moneyline_policy_recommendation_flag": _bool_series(
                merged, "market_moneyline_policy_recommendation_flag"
            ),
            "moneyline_policy_status": _string_series(merged, "market_moneyline_policy_status", "unavailable"),
            "spread_policy_recommendation_flag": _bool_series(
                merged, "market_spread_policy_recommendation_flag"
            ),
            "spread_policy_status": _string_series(merged, "market_spread_policy_status", "unavailable"),
            "policy_primary_signal": _string_series(merged, "market_policy_primary_signal", "none"),
            "policy_tier": _string_series(merged, "market_policy_tier", "no_bet"),
            "policy_status": _string_series(merged, "market_policy_status", "no_bet"),
            "pick_home_moneyline": pick_home_ml,
            "pick_away_moneyline": pick_away_ml,
            "close_home_moneyline": close_home_ml,
            "close_away_moneyline": close_away_ml,
            "model_side_pick_moneyline": model_side_pick_ml,
            "model_side_close_moneyline": model_side_close_ml,
            "model_side_pick_decimal": pick_decimal,
            "model_side_close_decimal": close_decimal,
            "model_side_pick_implied_prob_novig": pick_side_novig,
            "model_side_close_implied_prob_novig": close_side_novig,
            "moneyline_clv_decimal_delta": moneyline_decimal_delta,
            "moneyline_clv_prob_delta": moneyline_prob_delta,
            "moneyline_positive_clv": moneyline_decimal_delta.gt(0.0).where(moneyline_available),
            "model_vs_pick_prob_edge": model_pick_prob - pick_side_novig,
            "model_vs_close_prob_edge": model_pick_prob - close_side_novig,
            "pick_selected_spread": pick_selected_spread,
            "close_selected_spread": close_selected_spread,
            "spread_clv_points": spread_clv_points,
            "spread_positive_clv": spread_clv_points.gt(0.0).where(spread_available),
            "clv_status": clv_status,
        }
    )
    return detail[DETAIL_COLUMNS]


def _summary_record(
    scope: str,
    segment: str,
    selection: pd.DataFrame,
    *,
    threshold: float | None = None,
) -> dict[str, object]:
    n_games = int(len(selection))
    moneyline = selection[selection["moneyline_clv_decimal_delta"].notna()].copy()
    spread = selection[selection["spread_clv_points"].notna()].copy()
    return {
        "scope": scope,
        "segment": segment,
        "threshold": np.nan if threshold is None else float(threshold),
        "n_games": n_games,
        "moneyline_clv_available": int(len(moneyline)),
        "avg_moneyline_clv_decimal_delta": float(moneyline["moneyline_clv_decimal_delta"].mean())
        if not moneyline.empty
        else np.nan,
        "avg_moneyline_clv_prob_delta": float(moneyline["moneyline_clv_prob_delta"].mean())
        if not moneyline.empty
        else np.nan,
        "positive_moneyline_clv_rate": float(moneyline["moneyline_positive_clv"].mean())
        if not moneyline.empty
        else np.nan,
        "avg_model_vs_close_prob_edge": float(moneyline["model_vs_close_prob_edge"].mean())
        if not moneyline.empty
        else np.nan,
        "spread_clv_available": int(len(spread)),
        "avg_spread_clv_points": float(spread["spread_clv_points"].mean()) if not spread.empty else np.nan,
        "positive_spread_clv_rate": float(spread["spread_positive_clv"].mean()) if not spread.empty else np.nan,
    }


def summarize_market_clv(
    detail: pd.DataFrame,
    *,
    moneyline_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    moneyline_thresholds = moneyline_thresholds or [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
    records: list[dict[str, object]] = [
        _summary_record("overall", "all", detail),
        _summary_record(
            "overall",
            "moneyline_disagreement",
            detail[detail["moneyline_favorite_disagreement"].fillna(False).astype(bool)],
        ),
        _summary_record(
            "overall",
            "moneyline_policy_recommended",
            detail[detail["moneyline_policy_recommendation_flag"].fillna(False).astype(bool)],
        ),
        _summary_record(
            "overall",
            "spread_policy_recommended",
            detail[detail["spread_policy_recommendation_flag"].fillna(False).astype(bool)],
        ),
    ]
    disagreement = detail[detail["moneyline_favorite_disagreement"].fillna(False).astype(bool)].copy()
    pick_prob_edge = pd.to_numeric(
        disagreement.get("model_vs_pick_prob_edge", pd.Series(np.nan, index=disagreement.index)),
        errors="coerce",
    )
    for threshold in moneyline_thresholds:
        selection = disagreement[pick_prob_edge.ge(threshold)]
        records.append(
            _summary_record(
                "moneyline_threshold",
                f"prob_edge >= {threshold:.2f}",
                selection,
                threshold=threshold,
            )
        )
    for status, group in detail.groupby("policy_status", dropna=False, sort=True):
        records.append(_summary_record("policy_status", str(status), group))
    for status, group in detail.groupby("clv_status", dropna=False, sort=True):
        records.append(_summary_record("clv_status", str(status), group))
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def run(args: argparse.Namespace) -> None:
    picks = load_pick_snapshot(args.pick_snapshot, season=args.season, week=args.week)
    close = load_close_snapshot(args.close_source, season=args.season, week=args.week)
    detail = build_market_clv_detail(picks, close)
    summary = summarize_market_clv(detail, moneyline_thresholds=args.moneyline_thresholds)

    args.output_detail.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_detail, index=False)
    summary.to_csv(args.output_summary, index=False)

    print(f"[market_clv] wrote detail rows={len(detail)} -> {args.output_detail.resolve()}")
    print(f"[market_clv] wrote summary rows={len(summary)} -> {args.output_summary.resolve()}")
    if not summary.empty:
        display_cols = [
            "scope",
            "segment",
            "n_games",
            "moneyline_clv_available",
            "avg_moneyline_clv_prob_delta",
            "positive_moneyline_clv_rate",
            "spread_clv_available",
            "avg_spread_clv_points",
            "positive_spread_clv_rate",
        ]
        print(summary[display_cols].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build market CLV diagnostics from local odds snapshots.")
    parser.add_argument("--season", type=int, default=None, help="Season to filter.")
    parser.add_argument("--week", type=int, default=None, help="Week to filter.")
    parser.add_argument("--pick-snapshot", type=Path, default=DEFAULT_PICK_SNAPSHOT_PATH)
    parser.add_argument("--close-source", type=Path, default=DEFAULT_CLOSE_SOURCE_PATH)
    parser.add_argument("--output-detail", type=Path, default=DEFAULT_DETAIL_PATH)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--moneyline-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.08, 0.10, 0.15],
        help="Model-vs-pick probability-edge thresholds to summarize for CLV policy gating.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
