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
MARKET_CLV_SUMMARY_PATH = Path("analysis/market_clv_summary.csv")
MARKET_DISAGREEMENT_ROI_PATH = Path("analysis/market_disagreement_roi.csv")
MARKET_PROBABILITY_EDGE_CALIBRATION_PATH = Path("analysis/market_probability_edge_calibration.csv")
MARKET_BLEND_SUMMARY_PATH = Path("analysis/market_blend_summary.csv")
MARKET_BLEND_POLICY_PATH = Path("analysis/market_blend_policy.csv")
MARKET_SPREAD_ROI_PATH = Path("analysis/market_roi_spread.csv")
MARKET_SPREAD_EDGE_BINS_PATH = Path("analysis/market_roi_spread_edge_bins.csv")
MARKET_SPREAD_KEY_NUMBERS_PATH = Path("analysis/market_spread_key_numbers.csv")
MARKET_EDGE_POLICY_PATH = Path("analysis/market_edge_policy.csv")


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


def _select_probability_calibration_row(path: Path, *, bin_family: str, bin_label: str) -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty or "scope" not in df.columns or "bin_family" not in df.columns or "bin_label" not in df.columns:
        return None
    selection = df[
        df.get("scope").eq("overall")
        & df["bin_family"].eq(bin_family)
        & df["bin_label"].eq(bin_label)
    ]
    if selection.empty:
        return None
    return selection.iloc[0]


def _select_blend_summary_row(path: Path, *, segment: str = "all") -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty or "scope" not in df.columns or "segment" not in df.columns:
        return None
    selection = df[df["scope"].eq("overall") & df["segment"].eq(segment)]
    if selection.empty:
        return None
    return selection.iloc[0]


def _select_clv_summary_row(path: Path, *, scope: str = "overall", segment: str = "all") -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty or "scope" not in df.columns or "segment" not in df.columns:
        return None
    selection = df[df["scope"].astype(str).eq(scope) & df["segment"].astype(str).eq(segment)]
    if selection.empty:
        return None
    return selection.iloc[0]


def _select_spread_key_number_row(
    path: Path,
    *,
    key_number: float,
    edge_bucket: str = "2-3",
    segment: str = "key_window",
) -> pd.Series | None:
    df = _read_csv_if_exists(path)
    required = {"scope", "key_number", "edge_bucket", "segment"}
    if df.empty or not required.issubset(df.columns):
        return None
    df["key_number"] = pd.to_numeric(df["key_number"], errors="coerce")
    selection = df[
        df["scope"].eq("overall")
        & np.isclose(df["key_number"], float(key_number), atol=1e-9, equal_nan=False)
        & df["edge_bucket"].astype(str).eq(edge_bucket)
        & df["segment"].astype(str).eq(segment)
    ]
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


def _select_policy_row(path: Path, *, signal: str) -> pd.Series | None:
    df = _read_csv_if_exists(path)
    if df.empty or "signal" not in df.columns:
        return None
    selection = df[df["signal"].eq(signal)]
    if selection.empty:
        return None
    return selection.iloc[0]


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
    market_clv_summary_path: Path = MARKET_CLV_SUMMARY_PATH,
    market_disagreement_roi_path: Path = MARKET_DISAGREEMENT_ROI_PATH,
    market_probability_edge_calibration_path: Path = MARKET_PROBABILITY_EDGE_CALIBRATION_PATH,
    market_blend_summary_path: Path = MARKET_BLEND_SUMMARY_PATH,
    market_blend_policy_path: Path = MARKET_BLEND_POLICY_PATH,
    market_spread_roi_path: Path = MARKET_SPREAD_ROI_PATH,
    market_spread_edge_bins_path: Path = MARKET_SPREAD_EDGE_BINS_PATH,
    market_spread_key_numbers_path: Path = MARKET_SPREAD_KEY_NUMBERS_PATH,
    market_edge_policy_path: Path = MARKET_EDGE_POLICY_PATH,
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

    clv_disagreement = _select_clv_summary_row(market_clv_summary_path, segment="moneyline_disagreement")
    clv_policy = _select_clv_summary_row(market_clv_summary_path, segment="moneyline_policy_recommended")
    for order, row, key, metric_key, label, unit, formatter, description in [
        (
            180,
            clv_disagreement,
            "moneyline_clv_available",
            "clv_moneyline_disagreement_tracked_games",
            "CLV disagreement tracked games",
            "games",
            _fmt_count,
            "Favorite-disagreement games with both pick-time and close/latest moneyline snapshots.",
        ),
        (
            181,
            clv_disagreement,
            "avg_moneyline_clv_prob_delta",
            "clv_moneyline_disagreement_avg_prob_delta",
            "CLV disagreement avg prob delta",
            "percent",
            _fmt_percent,
            "Average no-vig close-minus-pick implied probability for the model-selected side; positive means the pick beat the close.",
        ),
        (
            182,
            clv_disagreement,
            "positive_moneyline_clv_rate",
            "clv_moneyline_disagreement_positive_rate",
            "CLV disagreement positive rate",
            "percent",
            _fmt_percent,
            "Share of tracked favorite-disagreement games where the pick-time price beat the close/latest price.",
        ),
        (
            183,
            clv_policy,
            "avg_moneyline_clv_prob_delta",
            "clv_moneyline_policy_avg_prob_delta",
            "CLV policy avg prob delta",
            "percent",
            _fmt_percent,
            "Average CLV probability delta for rows that passed the current moneyline policy recommendation flag.",
        ),
        (
            184,
            clv_policy,
            "positive_moneyline_clv_rate",
            "clv_moneyline_policy_positive_rate",
            "CLV policy positive rate",
            "percent",
            _fmt_percent,
            "Positive CLV rate for rows that passed the current moneyline policy recommendation flag.",
        ),
    ]:
        value = _value(row, key)
        sample_size = int(float(_value(row, "n_games", 0) or 0)) if row is not None else ""
        _add_metric(
            rows,
            season=season,
            week=week,
            generated_at=generated_at,
            section="market_clv",
            sort_order=order,
            metric_key=metric_key,
            label=label,
            value=value,
            display_value=formatter(value),
            unit=unit,
            sample_size=sample_size,
            description=description,
            source_artifact=market_clv_summary_path,
        )

    probability_calibration = _select_probability_calibration_row(
        market_probability_edge_calibration_path,
        bin_family="all",
        bin_label="all",
    )
    if probability_calibration is not None:
        calibration_n = int(float(_value(probability_calibration, "n_games", 0) or 0))
        for order, key, metric_key, label, description in [
            (
                172,
                "model_brier",
                "model_selected_side_probability_brier",
                "Model selected-side Brier",
                "Brier score for the model-selected side probability across market-matched games.",
            ),
            (
                173,
                "vegas_brier",
                "vegas_same_side_probability_brier",
                "Vegas same-side Brier",
                "Brier score for the no-vig Vegas probability on the model-selected side.",
            ),
            (
                174,
                "brier_delta_model_minus_vegas",
                "model_minus_vegas_probability_brier_delta",
                "Model minus Vegas Brier delta",
                "Negative means the model-selected-side probability beat no-vig Vegas on Brier.",
            ),
            (
                175,
                "model_calibration_error",
                "model_selected_side_calibration_error",
                "Model selected-side calibration error",
                "Average model-selected-side probability minus actual model-side win rate.",
            ),
            (
                176,
                "vegas_calibration_error",
                "vegas_same_side_calibration_error",
                "Vegas same-side calibration error",
                "Average no-vig Vegas same-side probability minus actual model-side win rate.",
            ),
        ]:
            value = _value(probability_calibration, key)
            _add_metric(
                rows,
                season=season,
                week=week,
                generated_at=generated_at,
                section="market_probability_calibration",
                sort_order=order,
                metric_key=metric_key,
                label=label,
                value=value,
                display_value=_fmt_decimal(value),
                unit="score",
                sample_size=calibration_n,
                description=description,
                source_artifact=market_probability_edge_calibration_path,
            )

    blend_summary = _select_blend_summary_row(market_blend_summary_path, segment="all")
    if blend_summary is not None:
        blend_n = int(float(_value(blend_summary, "n_games", 0) or 0))
        for order, key, metric_key, label, unit, formatter, description in [
            (
                177,
                "blend_brier_delta_vs_model",
                "market_blend_brier_delta_vs_model",
                "Blend vs model Brier delta",
                "score",
                _fmt_decimal,
                "Negative means the market-aware blend beat the raw model probability on Brier.",
            ),
            (
                178,
                "blend_brier_delta_vs_vegas",
                "market_blend_brier_delta_vs_vegas",
                "Blend vs Vegas Brier delta",
                "score",
                _fmt_decimal,
                "Negative means the market-aware blend beat no-vig Vegas probability on Brier.",
            ),
            (
                179,
                "blend_side_roi",
                "market_blend_moneyline_roi",
                "Blend selected-side moneyline ROI",
                "percent",
                _fmt_percent,
                "Flat one-unit ROI from betting the market-aware blend-selected side.",
            ),
        ]:
            value = _value(blend_summary, key)
            _add_metric(
                rows,
                season=season,
                week=week,
                generated_at=generated_at,
                section="market_blend",
                sort_order=order,
                metric_key=metric_key,
                label=label,
                value=value,
                display_value=formatter(value),
                unit=unit,
                sample_size=blend_n,
                description=description,
                source_artifact=market_blend_summary_path,
            )

    for order, signal, metric_key, label in [
        (181, "market_blend_probability", "market_blend_probability_policy", "Market blend probability policy"),
        (182, "market_blend_moneyline", "market_blend_moneyline_policy", "Market blend moneyline policy"),
    ]:
        policy = _select_policy_row(market_blend_policy_path, signal=signal)
        if policy is None:
            continue
        recommended_raw = _value(policy, "recommended", False)
        recommended = (
            bool(recommended_raw)
            if isinstance(recommended_raw, (bool, np.bool_))
            else str(recommended_raw).strip().lower() in {"true", "1", "yes"}
        )
        tier = str(_value(policy, "recommendation_tier", "") or "")
        selection = str(_value(policy, "selection", "") or "")
        policy_n = int(float(_value(policy, "n_games", 0) or 0))
        _add_metric(
            rows,
            season=season,
            week=week,
            generated_at=generated_at,
            section="market_blend",
            sort_order=order,
            metric_key=metric_key,
            label=label,
            value=selection if recommended else tier or "no_bet",
            display_value=selection if recommended else "No bet",
            unit="policy",
            sample_size=policy_n,
            description="Conservative market-aware blend policy; rows display No bet unless the blend beats both raw model and no-vig Vegas gates.",
            source_artifact=market_blend_policy_path,
        )

    policy_specs = [
        (180, "moneyline_disagreement", "moneyline_disagreement_policy", "Moneyline disagreement policy"),
        (190, "spread_threshold", "spread_threshold_policy", "Spread threshold policy"),
        (200, "spread_edge_bin", "spread_edge_bin_policy", "Spread edge-bin policy"),
    ]
    for order, signal, metric_key, label in policy_specs:
        policy = _select_policy_row(market_edge_policy_path, signal=signal)
        if policy is None:
            continue
        recommended_raw = _value(policy, "recommended", False)
        recommended = (
            bool(recommended_raw)
            if isinstance(recommended_raw, (bool, np.bool_))
            else str(recommended_raw).strip().lower() in {"true", "1", "yes"}
        )
        selection = str(_value(policy, "selection", "") or "")
        tier = str(_value(policy, "recommendation_tier", "") or "")
        display = selection if recommended else (
            "Review" if signal == "moneyline_disagreement" and tier == "review" else "No bet"
        )
        value = selection if recommended else tier or "no_bet"
        policy_n = int(float(_value(policy, "n_bets", 0) or 0))
        _add_metric(
            rows,
            season=season,
            week=week,
            generated_at=generated_at,
            section="market_policy",
            sort_order=order,
            metric_key=metric_key,
            label=label,
            value=value,
            display_value=display,
            unit="policy",
            sample_size=policy_n,
            description="Conservative historical market-edge policy; rows display No bet unless sample and ROI gates are met.",
            source_artifact=market_edge_policy_path,
        )
        _add_metric(
            rows,
            season=season,
            week=week,
            generated_at=generated_at,
            section="market_policy",
            sort_order=order + 1,
            metric_key=f"{metric_key}_roi",
            label=f"{label} ROI",
            value=_value(policy, "roi"),
            display_value=_fmt_percent(_value(policy, "roi")),
            unit="percent",
            sample_size=policy_n,
            description="Historical flat one-unit ROI for the policy candidate row.",
            source_artifact=market_edge_policy_path,
        )
        if signal == "moneyline_disagreement":
            stability_status = _value(policy, "stability_status")
            if stability_status is not None and not pd.isna(stability_status):
                qualified_seasons = _value(policy, "stability_qualified_seasons")
                positive_rate = _value(policy, "stability_positive_season_rate")
                recent_rate = _value(policy, "stability_recent_positive_season_rate")
                severe_rate = _value(policy, "stability_severe_negative_season_rate")
                for stability_order, metric_suffix, stability_label, stability_value, display_value, unit, description in [
                    (
                        2,
                        "stability_status",
                        "Moneyline disagreement stability",
                        stability_status,
                        str(stability_status),
                        "status",
                        "Season-stability gate status for the selected moneyline disagreement policy.",
                    ),
                    (
                        3,
                        "stability_qualified_seasons",
                        "Moneyline disagreement qualified seasons",
                        qualified_seasons,
                        _fmt_count(qualified_seasons),
                        "seasons",
                        "Number of seasons meeting the per-season minimum bet count for the selected policy.",
                    ),
                    (
                        4,
                        "stability_positive_season_rate",
                        "Moneyline disagreement positive-season rate",
                        positive_rate,
                        _fmt_percent(positive_rate),
                        "percent",
                        "Share of qualified seasons with positive ROI for the selected policy.",
                    ),
                    (
                        5,
                        "stability_recent_positive_season_rate",
                        "Moneyline disagreement recent positive-season rate",
                        recent_rate,
                        _fmt_percent(recent_rate),
                        "percent",
                        "Share of recent qualified seasons with positive ROI for the selected policy.",
                    ),
                    (
                        6,
                        "stability_severe_negative_season_rate",
                        "Moneyline disagreement severe-negative season rate",
                        severe_rate,
                        _fmt_percent(severe_rate),
                        "percent",
                        "Share of qualified seasons at or below the severe-negative ROI cutoff.",
                    ),
                ]:
                    _add_metric(
                        rows,
                        season=season,
                        week=week,
                        generated_at=generated_at,
                        section="market_policy",
                        sort_order=order + stability_order,
                        metric_key=f"{metric_key}_{metric_suffix}",
                        label=stability_label,
                        value=stability_value,
                        display_value=display_value,
                        unit=unit,
                        sample_size=policy_n,
                        description=description,
                        source_artifact=market_edge_policy_path,
                    )
            clv_status = _value(policy, "clv_status")
            if clv_status is not None and not pd.isna(clv_status):
                clv_tracked = _value(policy, "clv_tracked_games")
                clv_avg_delta = _value(policy, "clv_avg_prob_delta")
                clv_positive_rate = _value(policy, "clv_positive_rate")
                for clv_order, metric_suffix, clv_label, clv_value, display_value, unit, description in [
                    (
                        7,
                        "clv_status",
                        "Moneyline disagreement CLV gate",
                        clv_status,
                        str(clv_status),
                        "status",
                        "Current-week CLV gate status for the selected moneyline disagreement policy.",
                    ),
                    (
                        8,
                        "clv_tracked_games",
                        "Moneyline disagreement CLV tracked games",
                        clv_tracked,
                        _fmt_count(clv_tracked),
                        "games",
                        "Current-week games with both pick-time and close/latest moneyline snapshots at the selected policy threshold.",
                    ),
                    (
                        9,
                        "clv_avg_prob_delta",
                        "Moneyline disagreement CLV avg prob delta",
                        clv_avg_delta,
                        _fmt_percent(clv_avg_delta),
                        "percent",
                        "Average no-vig close-minus-pick implied probability for the model-selected side at the selected policy threshold.",
                    ),
                    (
                        10,
                        "clv_positive_rate",
                        "Moneyline disagreement CLV positive rate",
                        clv_positive_rate,
                        _fmt_percent(clv_positive_rate),
                        "percent",
                        "Share of tracked selected-threshold games where the pick-time price beat the close/latest price.",
                    ),
                ]:
                    _add_metric(
                        rows,
                        season=season,
                        week=week,
                        generated_at=generated_at,
                        section="market_policy",
                        sort_order=order + clv_order,
                        metric_key=f"{metric_key}_{metric_suffix}",
                        label=clv_label,
                        value=clv_value,
                        display_value=display_value,
                        unit=unit,
                        sample_size=policy_n,
                        description=description,
                        source_artifact=market_edge_policy_path,
                    )
        if signal == "spread_edge_bin":
            calibration_status = _value(policy, "spread_cover_calibration_status")
            if calibration_status is not None and not pd.isna(calibration_status):
                calibration_pass = _value(policy, "spread_cover_calibration_pass")
                max_abs_error = _value(policy, "spread_cover_calibration_max_abs_error")
                worst_bin = _value(policy, "spread_cover_calibration_worst_bin")
                for calibration_order, metric_suffix, calibration_label, calibration_value, display_value, unit, description in [
                    (
                        2,
                        "cover_calibration_status",
                        "Spread cover calibration gate",
                        calibration_status,
                        str(calibration_status),
                        "status",
                        "Empirical calibration gate for using raw spread cover probability as a confidence signal.",
                    ),
                    (
                        3,
                        "cover_calibration_pass",
                        "Spread cover calibration pass",
                        calibration_pass,
                        str(calibration_pass),
                        "boolean",
                        "Whether high raw spread cover probabilities are calibrated enough to promote spread edge-bin policy rows.",
                    ),
                    (
                        4,
                        "cover_calibration_max_abs_error",
                        "Spread cover max calibration error",
                        max_abs_error,
                        _fmt_percent(max_abs_error),
                        "percent",
                        "Worst absolute calibration error among sampled high-confidence spread cover-probability bins.",
                    ),
                    (
                        5,
                        "cover_calibration_worst_bin",
                        "Spread cover worst bin",
                        worst_bin,
                        str(worst_bin),
                        "bucket",
                        "High-confidence cover-probability bin with the largest empirical calibration error.",
                    ),
                ]:
                    _add_metric(
                        rows,
                        season=season,
                        week=week,
                        generated_at=generated_at,
                        section="market_policy",
                        sort_order=order + calibration_order,
                        metric_key=f"{metric_key}_{metric_suffix}",
                        label=calibration_label,
                        value=calibration_value,
                        display_value=display_value,
                        unit=unit,
                        sample_size=policy_n,
                        description=description,
                        source_artifact=market_edge_policy_path,
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

    for offset, key_number in enumerate([3.0, 6.0, 7.0, 10.0]):
        key_row = _select_spread_key_number_row(
            market_spread_key_numbers_path,
            key_number=key_number,
            edge_bucket="2-3",
            segment="key_window",
        )
        if key_row is None:
            continue
        key_n = int(float(_value(key_row, "n_bets", 0) or 0))
        _add_metric(
            rows,
            season=season,
            week=week,
            generated_at=generated_at,
            section="spread_edge",
            sort_order=350 + offset,
            metric_key=f"spread_key_{int(key_number)}_policy_bucket_roi",
            label=f"Spread key {int(key_number)} ROI",
            value=_value(key_row, "roi"),
            display_value=_fmt_percent(_value(key_row, "roi")),
            unit="percent",
            sample_size=key_n,
            description="ROI for current 2-3 point spread-edge policy bucket when the market/model line is near or crosses this NFL key number.",
            source_artifact=market_spread_key_numbers_path,
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
