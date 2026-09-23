from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.player_props.labels import DEFAULT_OUTPUT_PATH as DEFAULT_LABELS_PATH
from src.utils.io import read_df
from src.utils.teams import normalize_team_abbr
from src.utils.week_filter import filter_before_week


DEFAULT_PREDICTION_DIR = Path("predictions")
DEFAULT_OUTPUT_DIR = Path("predictions/evaluation/player_props")
BASELINE_PREDICTIONS_NAME = "baseline_predictions.csv"
BASELINE_METRICS_NAME = "baseline_metrics.csv"


@dataclass(frozen=True)
class ProjectionSpec:
    kind: str
    market: str
    projection_col: str


PROJECTION_SPECS: tuple[ProjectionSpec, ...] = (
    ProjectionSpec("qb", "qb_passing_yards", "projected_passing_yards"),
    ProjectionSpec("offense", "rb_rushing_yards", "projected_rushing_yards"),
    ProjectionSpec("offense", "wrte_receiving_yards", "projected_receiving_yards"),
    ProjectionSpec("defense", "def_sacks", "projected_sacks"),
)

CURRENT_FILES = {
    "qb": "predictions_players_qb.csv",
    "offense": "predictions_players_offense.csv",
    "defense": "predictions_players_defense.csv",
}

ARCHIVE_RE = re.compile(r"^w\d+_predictions_players_(qb|offense|defense)\.csv$", re.IGNORECASE)

PREDICTION_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "player_id",
    "player_name",
    "market",
    "baseline_projection",
    "projection_col",
    "games_sampled",
    "player_rank",
    "source_type",
    "source_file",
    "actual_value",
    "actual_over_zero",
    "label_available",
    "error",
    "abs_error",
    "squared_error",
    "prob_over_zero",
]

METRIC_COLUMNS = [
    "scope",
    "season",
    "market",
    "projection_col",
    "generated_at",
    "n_candidate_predictions",
    "n_available_labels",
    "n_scored",
    "prediction_scored_rate",
    "label_coverage",
    "mae",
    "rmse",
    "median_abs_error",
    "mean_error",
    "actual_mean",
    "projection_mean",
    "corr",
    "brier_over_zero",
    "logloss_over_zero",
    "auc_over_zero",
    "prob_over_zero_mean",
]


def _clean_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NAN", "NA", "<NA>"}:
        return None
    try:
        return normalize_team_abbr(text)
    except Exception:
        return text


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _filter_seasons(
    frame: pd.DataFrame,
    *,
    start_season: int | None = None,
    end_season: int | None = None,
) -> pd.DataFrame:
    if frame.empty or "season" not in frame.columns:
        return frame
    out = frame.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out = out[out["season"].notna()]
    out["season"] = out["season"].astype(int)
    if start_season is not None:
        out = out[out["season"] >= int(start_season)]
    if end_season is not None:
        out = out[out["season"] <= int(end_season)]
    return out


def discover_projection_files(
    prediction_dir: Path,
    *,
    include_current: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if include_current:
        for kind, filename in CURRENT_FILES.items():
            path = prediction_dir / filename
            if path.exists():
                rows.append(
                    {
                        "kind": kind,
                        "path": path,
                        "source_type": "current",
                        "source_priority": 1,
                    }
                )

    if prediction_dir.exists():
        for path in sorted(prediction_dir.glob("w*_predictions_players_*.csv")):
            match = ARCHIVE_RE.match(path.name)
            if not match:
                continue
            rows.append(
                {
                    "kind": match.group(1).lower(),
                    "path": path,
                    "source_type": "archive",
                    "source_priority": 0,
                }
            )

    return pd.DataFrame(rows, columns=["kind", "path", "source_type", "source_priority"])


def _read_projection_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect_baseline_projections(
    prediction_dir: Path,
    *,
    include_current: bool = True,
) -> pd.DataFrame:
    files = discover_projection_files(prediction_dir, include_current=include_current)
    if files.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS[:13] + ["source_priority"])

    records: list[pd.DataFrame] = []

    for file_row in files.itertuples(index=False):
        frame = _read_projection_csv(Path(file_row.path))
        if frame.empty:
            continue

        required = {"season", "week", "game_id", "team", "player_id"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{file_row.path} missing required columns: {sorted(missing)}")

        frame = frame.copy()
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
        frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
        frame = frame[frame["season"].notna() & frame["week"].notna()]
        if frame.empty:
            continue
        frame["season"] = frame["season"].astype(int)
        frame["week"] = frame["week"].astype(int)
        frame["game_id"] = _as_text(frame["game_id"])
        frame["team"] = frame["team"].apply(_clean_team)
        frame["player_id"] = _as_text(frame["player_id"])
        if "player_name" not in frame.columns:
            frame["player_name"] = pd.NA
        if "games_sampled" not in frame.columns:
            frame["games_sampled"] = pd.NA
        if "player_rank" not in frame.columns:
            frame["player_rank"] = pd.NA

        for spec in PROJECTION_SPECS:
            if spec.kind != file_row.kind or spec.projection_col not in frame.columns:
                continue

            subset = frame[
                [
                    "season",
                    "week",
                    "game_id",
                    "team",
                    "player_id",
                    "player_name",
                    "games_sampled",
                    "player_rank",
                    spec.projection_col,
                ]
            ].copy()
            subset["baseline_projection"] = pd.to_numeric(subset[spec.projection_col], errors="coerce")
            subset = subset[subset["baseline_projection"].notna()]
            if subset.empty:
                continue

            subset["market"] = spec.market
            subset["projection_col"] = spec.projection_col
            subset["source_type"] = file_row.source_type
            subset["source_file"] = Path(file_row.path).as_posix()
            subset["source_priority"] = int(file_row.source_priority)
            subset = subset.drop(columns=[spec.projection_col])
            records.append(subset)

    if not records:
        return pd.DataFrame(columns=PREDICTION_COLUMNS[:13] + ["source_priority"])

    projections = pd.concat(records, ignore_index=True)
    projections = projections.dropna(subset=["season", "week", "game_id", "team", "player_id", "market"])
    projections = projections.sort_values(
        ["season", "week", "game_id", "team", "player_id", "market", "source_priority", "source_file"],
        kind="stable",
    )
    projections = projections.drop_duplicates(
        subset=["season", "week", "game_id", "team", "player_id", "market"],
        keep="first",
    )
    return projections.reset_index(drop=True)


def _prepare_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(
            columns=["season", "week", "game_id", "team", "player_id", "market", "actual_value", "actual_over_zero"]
        )

    required = {"season", "week", "game_id", "team", "player_id", "market", "actual_value"}
    if missing := required - set(labels.columns):
        raise ValueError(f"Label data missing required columns: {sorted(missing)}")

    out = labels.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["actual_value"] = pd.to_numeric(out["actual_value"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna() & out["actual_value"].notna()]
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["game_id"] = _as_text(out["game_id"])
    out["team"] = out["team"].apply(_clean_team)
    out["player_id"] = _as_text(out["player_id"])
    out["market"] = _as_text(out["market"]).str.lower()
    if "actual_over_zero" not in out.columns:
        out["actual_over_zero"] = out["actual_value"] > 0
    else:
        out["actual_over_zero"] = out["actual_over_zero"].astype(bool)

    out = out.dropna(subset=["season", "week", "game_id", "team", "player_id", "market", "actual_value"])
    out = out.drop_duplicates(subset=["season", "week", "game_id", "team", "player_id", "market"], keep="last")
    return out[["season", "week", "game_id", "team", "player_id", "market", "actual_value", "actual_over_zero"]]


def _prediction_weeks(projections: pd.DataFrame) -> pd.DataFrame:
    if projections.empty:
        return pd.DataFrame(columns=["season", "week"])
    return projections[["season", "week"]].drop_duplicates()


def join_labels_to_projections(
    labels: pd.DataFrame,
    projections: pd.DataFrame,
    *,
    start_season: int | None = None,
    end_season: int | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_labels = _prepare_labels(labels)
    projections = projections.copy()

    prepared_labels = _filter_seasons(prepared_labels, start_season=start_season, end_season=end_season)
    projections = _filter_seasons(projections, start_season=start_season, end_season=end_season)
    prepared_labels = filter_before_week(prepared_labels, exclude_from_season, exclude_from_week)
    projections = filter_before_week(projections, exclude_from_season, exclude_from_week)

    weeks = _prediction_weeks(projections)
    if not weeks.empty:
        labels_for_eval = prepared_labels.merge(weeks, on=["season", "week"], how="inner")
    else:
        labels_for_eval = prepared_labels.iloc[0:0].copy()

    if projections.empty:
        joined = pd.DataFrame(columns=PREDICTION_COLUMNS)
        return joined, labels_for_eval

    joined = projections.merge(
        labels_for_eval,
        on=["season", "week", "game_id", "team", "player_id", "market"],
        how="left",
    )
    joined["label_available"] = joined["actual_value"].notna()
    joined["error"] = joined["baseline_projection"] - joined["actual_value"]
    joined["abs_error"] = joined["error"].abs()
    joined["squared_error"] = joined["error"] ** 2
    joined["prob_over_zero"] = np.nan
    sack_mask = joined["market"].eq("def_sacks") & joined["baseline_projection"].notna()
    joined.loc[sack_mask, "prob_over_zero"] = 1.0 - np.exp(-joined.loc[sack_mask, "baseline_projection"].clip(lower=0))

    output_cols = [col for col in PREDICTION_COLUMNS if col in joined.columns]
    return joined[output_cols], labels_for_eval


def _safe_corr(actual: pd.Series, projection: pd.Series) -> float:
    if len(actual) < 2 or actual.nunique(dropna=True) < 2 or projection.nunique(dropna=True) < 2:
        return float("nan")
    return float(actual.corr(projection))


def _binary_metrics(scored: pd.DataFrame) -> tuple[float, float, float, float]:
    if scored.empty or "prob_over_zero" not in scored.columns:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    valid = scored[scored["prob_over_zero"].notna()].copy()
    if valid.empty:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    y_true = valid["actual_over_zero"].astype(bool).astype(int)
    y_prob = valid["prob_over_zero"].clip(1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y_true, y_prob))
    logloss_value = float(log_loss(y_true, y_prob, labels=[0, 1]))
    auc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() == 2 else float("nan")
    return (brier, logloss_value, auc, float(y_prob.mean()))


def build_baseline_metrics(
    joined: pd.DataFrame,
    labels_for_eval: pd.DataFrame,
    *,
    generated_at: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    spec_by_market = {spec.market: spec for spec in PROJECTION_SPECS}
    markets = sorted(set(spec_by_market) | set(joined.get("market", [])) | set(labels_for_eval.get("market", [])))

    def append_row(scope: str, season: int | None, market: str) -> None:
        pred_group = joined[joined["market"].eq(market)] if not joined.empty else joined
        label_group = labels_for_eval[labels_for_eval["market"].eq(market)] if not labels_for_eval.empty else labels_for_eval
        if season is not None:
            pred_group = pred_group[pred_group["season"].eq(season)]
            label_group = label_group[label_group["season"].eq(season)]

        scored = pred_group[pred_group["label_available"].fillna(False)].copy() if not pred_group.empty else pred_group
        n_candidates = int(len(pred_group))
        n_labels = int(len(label_group))
        n_scored = int(len(scored))
        if n_candidates == 0 and n_labels == 0:
            return

        if n_scored:
            actual = scored["actual_value"].astype(float)
            projection = scored["baseline_projection"].astype(float)
            errors = projection - actual
            abs_error = errors.abs()
            squared_error = errors.pow(2)
            mae = float(abs_error.mean())
            rmse = float(np.sqrt(squared_error.mean()))
            median_abs_error = float(abs_error.median())
            mean_error = float(errors.mean())
            actual_mean = float(actual.mean())
            projection_mean = float(projection.mean())
            corr = _safe_corr(actual, projection)
        else:
            mae = rmse = median_abs_error = mean_error = actual_mean = projection_mean = corr = float("nan")

        brier, logloss_value, auc, prob_over_zero_mean = _binary_metrics(scored)
        spec = spec_by_market.get(market)
        rows.append(
            {
                "scope": scope,
                "season": season if season is not None else pd.NA,
                "market": market,
                "projection_col": spec.projection_col if spec else "",
                "generated_at": generated_at,
                "n_candidate_predictions": n_candidates,
                "n_available_labels": n_labels,
                "n_scored": n_scored,
                "prediction_scored_rate": n_scored / n_candidates if n_candidates else np.nan,
                "label_coverage": n_scored / n_labels if n_labels else np.nan,
                "mae": mae,
                "rmse": rmse,
                "median_abs_error": median_abs_error,
                "mean_error": mean_error,
                "actual_mean": actual_mean,
                "projection_mean": projection_mean,
                "corr": corr,
                "brier_over_zero": brier,
                "logloss_over_zero": logloss_value,
                "auc_over_zero": auc,
                "prob_over_zero_mean": prob_over_zero_mean,
            }
        )

    for market in markets:
        append_row("overall", None, market)

    seasons = sorted(
        set(pd.to_numeric(joined.get("season", pd.Series(dtype=int)), errors="coerce").dropna().astype(int))
        | set(pd.to_numeric(labels_for_eval.get("season", pd.Series(dtype=int)), errors="coerce").dropna().astype(int))
    )
    for season in seasons:
        for market in markets:
            append_row("season", int(season), market)

    if not rows:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def evaluate_baselines(
    *,
    labels: pd.DataFrame,
    projections: pd.DataFrame,
    generated_at: str,
    start_season: int | None = None,
    end_season: int | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined, labels_for_eval = join_labels_to_projections(
        labels,
        projections,
        start_season=start_season,
        end_season=end_season,
        exclude_from_season=exclude_from_season,
        exclude_from_week=exclude_from_week,
    )
    metrics = build_baseline_metrics(joined, labels_for_eval, generated_at=generated_at)
    return joined, metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate current rate-based player projection CSVs against player prop labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="Player prop label parquet/csv.")
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR, help="Prediction CSV directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Evaluation output directory.")
    parser.add_argument("--start-season", type=int, default=None, help="Earliest season to evaluate.")
    parser.add_argument("--end-season", type=int, default=None, help="Latest season to evaluate.")
    parser.add_argument("--exclude-from-season", type=int, default=None, help="Exclude this season/week and later rows.")
    parser.add_argument("--exclude-from-week", type=int, default=None, help="Exclude this season/week and later rows.")
    parser.add_argument("--no-current", action="store_true", help="Ignore current predictions_players_*.csv files.")
    parser.add_argument("--debug", action="store_true", help="Print baseline metric summary.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now(timezone.utc).isoformat()

    labels = read_df(args.labels) if args.labels.exists() else pd.DataFrame()
    projections = collect_baseline_projections(args.prediction_dir, include_current=not args.no_current)
    joined, metrics = evaluate_baselines(
        labels=labels,
        projections=projections,
        generated_at=generated_at,
        start_season=args.start_season,
        end_season=args.end_season,
        exclude_from_season=args.exclude_from_season,
        exclude_from_week=args.exclude_from_week,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / BASELINE_PREDICTIONS_NAME
    metrics_path = args.output_dir / BASELINE_METRICS_NAME
    joined.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    if args.debug and not metrics.empty:
        summary = metrics[metrics["scope"].eq("overall")][
            ["market", "n_scored", "mae", "rmse", "label_coverage", "brier_over_zero", "auc_over_zero"]
        ]
        print("[player_props.baselines] overall metrics:")
        print(summary.to_string(index=False))

    print(f"[player_props.baselines] wrote {len(joined)} prediction rows -> {predictions_path.resolve()}")
    print(f"[player_props.baselines] wrote {len(metrics)} metric rows -> {metrics_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
