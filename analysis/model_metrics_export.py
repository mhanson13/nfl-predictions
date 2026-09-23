from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("analysis")
OVERALL_METRICS_PATH = Path("predictions/evaluation/overall_metrics.csv")
MARKET_BENCHMARK_SUMMARY_PATH = Path("analysis/market_benchmark_summary.csv")
MARKET_DISAGREEMENT_ROI_PATH = Path("analysis/market_disagreement_roi.csv")
MARKET_SPREAD_ROI_PATH = Path("analysis/market_roi_spread.csv")
MARKET_SPREAD_EDGE_BINS_PATH = Path("analysis/market_roi_spread_edge_bins.csv")


@dataclass(frozen=True)
class MetricRow:
    season: int
    week: int
    generated_at: str
    section: str
    sort_order: int
    metric_key: str
    label: str
    value: float | int | str
    display_value: str
    unit: str
    sample_size: int | str
    description: str
    source_artifact: str


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest_overall_metrics(path: Path) -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty:
        return None
    if "n_games" not in df.columns or "accuracy" not in df.columns:
        return None

    for col in ["n_games", "accuracy", "f1", "auc", "brier", "log_loss", "mae_margin", "rmse_margin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[df["n_games"].notna() & df["accuracy"].notna()]
    if valid.empty:
        return None
    return valid.iloc[-1]


def _select_market_row(path: Path, *, segment: str) -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty:
        return None
    selection = df[(df.get("scope") == "overall") & (df.get("segment") == segment)]
    if selection.empty:
        return None
    return selection.iloc[0]


def _select_threshold_row(path: Path, threshold: float) -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty or "threshold" not in df.columns:
        return None
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    selection = df[np.isclose(df["threshold"], threshold, equal_nan=False)]
    if selection.empty:
        return None
    return selection.iloc[0]


def _select_best_roi(path: Path, *, min_bets: int = 1, scope: str | None = None) -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty or "roi" not in df.columns:
        return None
    if scope is not None and "scope" in df.columns:
        df = df[df["scope"] == scope]
    if "n_bets" in df.columns:
        df["n_bets"] = pd.to_numeric(df["n_bets"], errors="coerce")
        df = df[df["n_bets"] >= min_bets]
    df["roi"] = pd.to_numeric(df["roi"], errors="coerce")
    df = df[df["roi"].notna()]
    if df.empty:
        return None
    return df.loc[df["roi"].idxmax()]


def _fmt_percent(value: float | int | str | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_decimal(value: float | int | str | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_points(value: float | int | str | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_count(value: float | int | str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(float(value)):,}"


def _value(row: pd.Series | None, key: str, default: float | int | str | None = None) -> float | int | str | None:
    if row is None or key not in row.index or pd.isna(row[key]):
        return default
    value = row[key]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _add_metric(
    rows: list[MetricRow],
    *,
    season: int,
    week: int,
    generated_at: str,
    section: str,
    sort_order: int,
    metric_key: str,
    label: str,
    value: float | int | str | None,
    display_value: str,
    unit: str,
    sample_size: int | str,
    description: str,
    source_artifact: Path,
) -> None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return
    rows.append(
        MetricRow(
            season=season,
            week=week,
            generated_at=generated_at,
            section=section,
            sort_order=sort_order,
            metric_key=metric_key,
            label=label,
            value=value,
            display_value=display_value,
            unit=unit,
            sample_size=sample_size,
            description=description,
            source_artifact=source_artifact.as_posix(),
        )
    )


def build_model_metrics_rows(
    *,
    season: int,
    week: int,
    generated_at: str,
    overall_metrics_path: Path = OVERALL_METRICS_PATH,
    market_benchmark_summary_path: Path = MARKET_BENCHMARK_SUMMARY_PATH,
    market_disagreement_roi_path: Path = MARKET_DISAGREEMENT_ROI_PATH,
    market_spread_roi_path: Path = MARKET_SPREAD_ROI_PATH,
    market_spread_edge_bins_path: Path = MARKET_SPREAD_EDGE_BINS_PATH,
    min_spread_edge_bin_bets: int = 50,
    start_season: int | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> pd.DataFrame:
    rows: list[MetricRow] = []

    overall = _latest_overall_metrics(overall_metrics_path)
    overall_n = int(float(_value(overall, "n_games", 0) or 0)) if overall is not None else ""

    _add_metric(
        rows,
        season=season,
        week=week,
        generated_at=generated_at,
        section="model_performance",
        sort_order=10,
        metric_key="evaluation_games",
        label="Evaluated games",
        value=overall_n,
        display_value=_fmt_count(overall_n),
        unit="games",
        sample_size=overall_n,
        description="Completed historical games included in the current walk-forward evaluation window.",
        source_artifact=overall_metrics_path,
    )
    for order, key, label, unit, formatter, description in [
        (20, "accuracy", "Straight-up accuracy", "percent", _fmt_percent, "Model accuracy for predicting outright winners."),
        (30, "f1", "F1 score", "percent", _fmt_percent, "Balanced precision/recall score for home-win classification."),
        (40, "auc", "AUC", "decimal", _fmt_decimal, "Rank-order separation for home-win probabilities."),
        (50, "brier", "Brier score", "decimal", _fmt_decimal, "Probability accuracy score; lower is better."),
        (60, "log_loss", "Log loss", "decimal", _fmt_decimal, "Penalty for overconfident wrong probabilities; lower is better."),
        (70, "mae_margin", "Margin MAE", "points", _fmt_points, "Average absolute error of predicted game margin."),
        (80, "rmse_margin", "Margin RMSE", "points", _fmt_points, "Root-mean-square margin error; penalizes large misses."),
    ]:
        value = _value(overall, key)
        _add_metric(
            rows,
            season=season,
            week=week,
            generated_at=generated_at,
            section="model_performance",
            sort_order=order,
            metric_key=key,
            label=label,
            value=value,
            display_value=formatter(value),
            unit=unit,
            sample_size=overall_n,
            description=description,
            source_artifact=overall_metrics_path,
        )

    market_all = _select_market_row(market_benchmark_summary_path, segment="all")
    market_disagree = _select_market_row(market_benchmark_summary_path, segment="disagree")
    disagreement_roi = _select_threshold_row(market_disagreement_roi_path, 0.0)
    market_n = int(float(_value(market_all, "n_games", 0) or 0)) if market_all is not None else ""
    disagreement_n = int(float(_value(market_disagree, "n_games", 0) or 0)) if market_disagree is not None else ""

    for order, row, key, metric_key, label, sample_size, description in [
        (
            110,
            market_all,
            "vegas_accuracy",
            "vegas_favorite_accuracy",
            "Vegas favorite accuracy",
            market_n,
            "No-vig Vegas favorite accuracy over the same evaluated games.",
        ),
        (
            120,
            market_all,
            "favorite_agreement_rate",
            "model_vegas_agreement_rate",
            "Model/Vegas agreement",
            market_n,
            "Share of games where model and no-vig Vegas favorite picked the same side.",
        ),
        (
            130,
            market_disagree,
            "n_games",
            "model_vegas_disagreement_games",
            "Disagreement games",
            disagreement_n,
            "Games where the model and no-vig Vegas favorite picked opposite sides.",
        ),
        (
            140,
            market_disagree,
            "model_accuracy",
            "disagreement_model_accuracy",
            "Model accuracy in disagreements",
            disagreement_n,
            "Model straight-up accuracy only in games where it disagreed with the Vegas favorite.",
        ),
        (
            150,
            market_disagree,
            "vegas_accuracy",
            "disagreement_vegas_accuracy",
            "Vegas accuracy in disagreements",
            disagreement_n,
            "No-vig Vegas favorite straight-up accuracy in model disagreement games.",
        ),
        (
            160,
            market_all,
            "model_side_roi",
            "overall_model_side_moneyline_roi",
            "Model-side moneyline ROI",
            market_n,
            "Flat one-unit ROI from betting the model-selected side at listed moneyline odds.",
        ),
        (
            170,
            disagreement_roi,
            "roi",
            "disagreement_moneyline_roi",
            "Disagreement moneyline ROI",
            disagreement_n,
            "Flat one-unit ROI when betting the model side only in favorite-disagreement games.",
        ),
    ]:
        value = _value(row, key)
        formatter = _fmt_count if metric_key == "model_vegas_disagreement_games" else _fmt_percent
        unit = "games" if metric_key == "model_vegas_disagreement_games" else "percent"
        _add_metric(
            rows,
            season=season,
            week=week,
            generated_at=generated_at,
            section="market_benchmark",
            sort_order=order,
            metric_key=metric_key,
            label=label,
            value=value,
            display_value=formatter(value),
            unit=unit,
            sample_size=sample_size,
            description=description,
            source_artifact=market_benchmark_summary_path
            if row is not disagreement_roi
            else market_disagreement_roi_path,
        )

    best_spread = _select_best_roi(market_spread_roi_path)
    best_edge_bin = _select_best_roi(
        market_spread_edge_bins_path,
        min_bets=min_spread_edge_bin_bets,
        scope="overall",
    )

    if best_spread is not None:
        side = str(_value(best_spread, "bet_side", "") or "")
        threshold = _value(best_spread, "threshold")
        spread_label = f"{side.title()} >{float(threshold):g} pts" if threshold is not None else side.title()
        best_spread_n = int(float(_value(best_spread, "n_bets", 0) or 0))
        for order, key, metric_key, label, unit, formatter, value in [
            (210, "bet_side", "best_spread_side", "Best spread side", "side", str, side),
            (220, "threshold", "best_spread_threshold_points", "Best spread edge threshold", "points", _fmt_points, threshold),
            (230, "roi", "best_spread_roi", "Best spread ROI", "percent", _fmt_percent, _value(best_spread, "roi")),
            (240, "n_bets", "best_spread_bets", "Best spread sample", "bets", _fmt_count, best_spread_n),
            (250, "hit_rate", "best_spread_hit_rate", "Best spread hit rate", "percent", _fmt_percent, _value(best_spread, "hit_rate")),
        ]:
            display = spread_label if metric_key == "best_spread_side" else formatter(value)
            _add_metric(
                rows,
                season=season,
                week=week,
                generated_at=generated_at,
                section="spread_edge",
                sort_order=order,
                metric_key=metric_key,
                label=label,
                value=value,
                display_value=display,
                unit=unit,
                sample_size=best_spread_n,
                description="Best corrected spread ROI row from the current backtest summary.",
                source_artifact=market_spread_roi_path,
            )

    if best_edge_bin is not None:
        edge_bin = str(_value(best_edge_bin, "edge_bin", "") or "")
        best_bin_n = int(float(_value(best_edge_bin, "n_bets", 0) or 0))
        for order, metric_key, label, unit, formatter, value in [
            (310, "best_spread_edge_bin", "Best spread edge bucket", "bucket", str, edge_bin),
            (320, "best_spread_edge_bin_roi", "Best spread edge bucket ROI", "percent", _fmt_percent, _value(best_edge_bin, "roi")),
            (330, "best_spread_edge_bin_bets", "Best spread edge bucket sample", "bets", _fmt_count, best_bin_n),
            (340, "best_spread_edge_bin_hit_rate", "Best spread edge bucket hit rate", "percent", _fmt_percent, _value(best_edge_bin, "hit_rate")),
        ]:
            display = f"{edge_bin} pts" if metric_key == "best_spread_edge_bin" else formatter(value)
            _add_metric(
                rows,
                season=season,
                week=week,
                generated_at=generated_at,
                section="spread_edge",
                sort_order=order,
                metric_key=metric_key,
                label=label,
                value=value,
                display_value=display,
                unit=unit,
                sample_size=best_bin_n,
                description="Best corrected absolute spread-edge bucket with minimum sample filter applied.",
                source_artifact=market_spread_edge_bins_path,
            )

    for order, metric_key, label, value, description in [
        (900, "evaluation_start_season", "Evaluation start season", start_season, "Lower season bound used for current evaluation artifacts."),
        (910, "excluded_from_season", "Live cutoff season", exclude_from_season, "Season excluded at and after the live cutoff."),
        (920, "excluded_from_week", "Live cutoff week", exclude_from_week, "Week excluded at and after the live cutoff."),
    ]:
        if value is not None:
            _add_metric(
                rows,
                season=season,
                week=week,
                generated_at=generated_at,
                section="metadata",
                sort_order=order,
                metric_key=metric_key,
                label=label,
                value=value,
                display_value=str(value),
                unit="",
                sample_size="",
                description=description,
                source_artifact=Path("tools/run_pipeline.py"),
            )

    columns = [field.name for field in MetricRow.__dataclass_fields__.values()]
    return pd.DataFrame([row.__dict__ for row in rows], columns=columns)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export website-ready model metrics for the active prediction week.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, required=True, help="Prediction season for the metrics export.")
    parser.add_argument("--week", type=int, required=True, help="Prediction week for the metrics export.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the output CSV.")
    parser.add_argument("--start-season", type=int, default=None, help="Evaluation lower-bound season metadata.")
    parser.add_argument("--exclude-from-season", type=int, default=None, help="Live cutoff season metadata.")
    parser.add_argument("--exclude-from-week", type=int, default=None, help="Live cutoff week metadata.")
    parser.add_argument(
        "--min-spread-edge-bin-bets",
        type=int,
        default=50,
        help="Minimum bets required for best spread edge-bin selection.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics = build_model_metrics_rows(
        season=args.season,
        week=args.week,
        generated_at=generated_at,
        min_spread_edge_bin_bets=args.min_spread_edge_bin_bets,
        start_season=args.start_season,
        exclude_from_season=args.exclude_from_season,
        exclude_from_week=args.exclude_from_week,
    )
    if metrics.empty:
        raise RuntimeError("No model metrics were available to export.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"model_metrics_week_{args.week:02d}.csv"
    metrics.sort_values(["sort_order", "section"]).to_csv(output_path, index=False)
    print(f"[model_metrics_export] Wrote {len(metrics)} metrics to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
