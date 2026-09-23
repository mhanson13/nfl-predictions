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

"""Generate leaderboards/plots comparing all logged model runs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

RUNS_CSV = Path("predictions/evaluation/overall_metrics.csv")
OUTPUT_DIR = Path("analysis/run_comparisons")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame to markdown, falling back to plain text if needed."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _fmt(value, digits: int = 3) -> str:
    """Format floats with ``digits`` decimal places while keeping NaNs readable."""
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _load_runs(path: Path) -> pd.DataFrame:
    """Load and basic-clean the evaluation CSV (raises if missing/empty)."""
    if not path.exists():
        raise FileNotFoundError(f"Runs CSV not found at {path}.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("overall_metrics.csv is empty; no runs to compare.")

    df = df.copy()
    alias_map = {
        "accuracy": "acc",
        "acc": "acc",
        "brier": "brier",
        "log_loss": "logloss",
        "logloss": "logloss",
        "mae": "mae",
        "mae_margin": "mae",
        "rmse": "rmse",
        "rmse_margin": "rmse",
        "n_games": "n_samples",
        "samples": "n_samples",
        "precision": "precision",
        "recall": "recall",
        "specificity": "specificity",
        "f1": "f1",
    }
    for src, target in alias_map.items():
        if src in df.columns:
            values = pd.to_numeric(df[src], errors="coerce")
            if target in df.columns:
                df[target] = df[target].combine_first(values)
            else:
                df[target] = values
    numeric_cols = [
        "n_samples",
        "acc",
        "auc",
        "brier",
        "logloss",
        "mae",
        "rmse",
        "precision",
        "recall",
        "specificity",
        "f1",
        "pos_rate",
        "proba_min",
        "proba_max",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    created = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in ("created_at", "run_timestamp", "timestamp"):
        if col in df.columns:
            candidate = pd.to_datetime(df[col], errors="coerce", utc=True)
            if candidate.notna().any():
                created = created.combine_first(candidate)
    if created.isna().all():
        created = pd.date_range("2000-01-01", periods=len(df), freq="h")
    else:
        created = created.ffill().bfill()
    df["created_at"] = created
    if "run_id" not in df.columns:
        df["run_id"] = [f"run_{i}" for i in range(len(df))]
    else:
        df["run_id"] = df["run_id"].fillna(method="ffill").fillna(method="bfill")
    if "model_name" not in df.columns:
        df["model_name"] = "unknown_model"
    else:
        df["model_name"] = df["model_name"].fillna("unknown_model")
    return df.sort_values("created_at").reset_index(drop=True)


def _current_evaluation_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer current-schema full evaluation rows over calibration snapshots and old leaky rows."""
    mask = pd.Series(True, index=df.index)
    if "stage" in df.columns:
        mask &= df["stage"].isna()
    for col in ("auc", "brier", "n_samples"):
        if col in df.columns:
            mask &= df[col].notna()
    if "f1" in df.columns and df["f1"].notna().any():
        mask &= df["f1"].notna()
    filtered = df.loc[mask].copy()
    return filtered if not filtered.empty else df.copy()


def _save_line_plot(df: pd.DataFrame, metric: str, path: Path) -> None:
    """Persist a simple metric-over-time line chart for the requested column."""
    if metric not in df.columns:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(df["created_at"], df[metric], marker="o")
    plt.title(f"{metric.upper()} over time")
    plt.xlabel("run time")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _top_runs(df: pd.DataFrame, sort_cols: List[str], ascending: List[bool], n: int = 5) -> pd.DataFrame:
    """Return the top-N runs for the provided sort ordering."""
    cols = [c for c in sort_cols if c in df.columns]
    if not cols:
        return df.head(n)
    return df.sort_values(by=cols, ascending=ascending).head(n)


def _metric_table(df: pd.DataFrame, metric: str, ascending: bool) -> str | None:
    """Build a markdown snippet describing the best runs for one metric."""
    if metric not in df.columns:
        return None
    cols = ["run_id", "model_name", "created_at", metric]
    working = df[cols].sort_values(metric, ascending=ascending).head(5)
    return _df_to_markdown(working)


def generate_report(df: pd.DataFrame) -> str:
    """Create markdown report text plus save sidecar plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_count = len(df)
    df = _current_evaluation_runs(df.copy()).sort_values("created_at").reset_index(drop=True)
    filtered_count = len(df)
    leaderboard_cols = [
        "run_id",
        "model_name",
        "created_at",
        "n_samples",
        "acc",
        "auc",
        "brier",
        "logloss",
        "mae",
        "rmse",
    ]
    existing_cols = [c for c in leaderboard_cols if c in df.columns]
    leaderboard = df[existing_cols].sort_values("created_at", ascending=False).head(10)
    leaderboard_md = _df_to_markdown(leaderboard)

    best_auc = _top_runs(df, ["auc"], [False], n=1).iloc[0]
    prev_best_df = _top_runs(df, ["auc"], [False], n=2)
    prev_best = prev_best_df.iloc[1] if len(prev_best_df) > 1 else None
    latest_run = df.iloc[-1]

    summary_lines: List[str] = []
    best_date = best_auc.get("created_at")
    best_date_str = best_date.strftime("%Y-%m-%d") if pd.notna(best_date) else "unknown date"
    best_acc = _fmt(best_auc.get("acc"))
    best_auc_val = _fmt(best_auc.get("auc"))
    best_brier_val = _fmt(best_auc.get("brier"))
    best_logloss_val = _fmt(best_auc.get("logloss"))
    best_n = best_auc.get("n_samples")
    best_n_str = f"{int(best_n)}" if pd.notna(best_n) else "NA"
    summary_lines.append(
        (
            f"Best run so far: **{best_auc.get('model_name')} ({best_auc.get('run_id')})** "
            f"on {best_date_str}. AUC={best_auc_val}, Brier={best_brier_val}, "
            f"LogLoss={best_logloss_val}, Accuracy={best_acc}, n={best_n_str}."
        )
    )
    if prev_best is not None:
        prev_auc = prev_best.get("auc")
        prev_brier = prev_best.get("brier")
        if pd.notna(prev_auc) and pd.notna(best_auc.get("auc")) and pd.notna(prev_brier) and pd.notna(best_auc.get("brier")):
            summary_lines.append(
                (
                    f"It edges the previous leader ({prev_best.get('model_name')} / {prev_best.get('run_id')}) "
                    f"by {best_auc.get('auc') - prev_auc:+.3f} AUC and "
                    f"{prev_brier - best_auc.get('brier'):+.3f} Brier."
                )
            )
    if latest_run["run_id"] != best_auc["run_id"]:
        summary_lines.append(
            (
                f"The most recent run ({latest_run['model_name']} / {latest_run['run_id']}) "
                f"delivered AUC={_fmt(latest_run.get('auc'))} and Brier={_fmt(latest_run.get('brier'))}."
            )
        )

    metric_sections: List[str] = []
    auc_table = _metric_table(df, "auc", ascending=False)
    brier_table = _metric_table(df, "brier", ascending=True)
    logloss_table = _metric_table(df, "logloss", ascending=True)
    if auc_table:
        metric_sections.extend(["### AUC leaders", auc_table, ""])
    if brier_table:
        metric_sections.extend(["### Brier leaders", brier_table, ""])
    if logloss_table:
        metric_sections.extend(["### LogLoss leaders", logloss_table, ""])

    if {"auc", "brier", "logloss"} <= set(df.columns):
        df["combined_score"] = 0.4 * df["auc"] - 0.3 * df["brier"] - 0.3 * df["logloss"]
        combo_table = df[["run_id", "model_name", "created_at", "combined_score"]].sort_values(
            "combined_score", ascending=False
        ).head(5)
        metric_sections.extend(
            ["### Composite score (0.4*AUC - 0.3*Brier - 0.3*LogLoss)", _df_to_markdown(combo_table), ""]
        )

    if not metric_sections:
        metric_sections.append("_Metric columns not available in overall_metrics.csv._")

    report_lines = [
        "# Run Comparison Report",
        f"Generated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M:%S UTC}",
        "",
        "## Run Leaderboard",
        f"_Using {filtered_count} current-schema evaluation runs from {raw_count} total metric rows._",
        "",
        leaderboard_md,
        "",
        "## Metric Leaders",
        *metric_sections,
        "## Highlights",
        "\n".join(summary_lines),
        "",
    ]

    # Metric trend plots
    _save_line_plot(df, "auc", OUTPUT_DIR / "auc_over_time.png")
    _save_line_plot(df, "brier", OUTPUT_DIR / "brier_over_time.png")
    _save_line_plot(df, "logloss", OUTPUT_DIR / "logloss_over_time.png")
    report_lines.append("## Metric Trends")
    report_lines.append("![](./auc_over_time.png)")
    report_lines.append("![](./brier_over_time.png)")
    report_lines.append("![](./logloss_over_time.png)")
    report_lines.append("")
    return "\n".join(report_lines)


def main() -> None:
    """CLI entry point used by the pipeline to regenerate comparison artefacts."""
    try:
        df = _load_runs(RUNS_CSV)
    except FileNotFoundError as exc:
        print(f"[compare_runs] {exc}")
        return
    except ValueError as exc:
        print(f"[compare_runs] {exc}")
        return

    report = generate_report(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "run_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[compare_runs] Wrote report to {report_path}")


if __name__ == "__main__":
    main()
