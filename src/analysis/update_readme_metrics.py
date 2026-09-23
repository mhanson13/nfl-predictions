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

"""Update the README's model-performance section from logged evaluation metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

RUNS_CSV = Path("predictions/evaluation/overall_metrics.csv")
README_PATH = Path("README.md")
_ENCODINGS = ("utf-8", "cp1252")


def read_text_with_fallback(path: Path) -> str:
    """
    Read text from ``path`` using UTF-8 first, then CP1252 if required.

    Prints a short message when the fallback encoding is used.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    for idx, encoding in enumerate(_ENCODINGS):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            if idx < len(_ENCODINGS) - 1:
                print(f"[update_readme_metrics] Falling back to {_ENCODINGS[idx + 1]} for {path} (decode error).")
            continue
    return path.read_text(errors="ignore")


def write_text_with_fallback(path: Path, text: str) -> None:
    """
    Write ``text`` to ``path`` using UTF-8 first, falling back to CP1252 if needed.

    A brief message is logged whenever the fallback encoding is exercised.
    """
    for idx, encoding in enumerate(_ENCODINGS):
        try:
            path.write_text(text, encoding=encoding)
            return
        except UnicodeEncodeError:
            if idx < len(_ENCODINGS) - 1:
                print(f"[update_readme_metrics] Falling back to {_ENCODINGS[idx + 1]} for {path} (encode error).")
            continue
    path.write_text(text, encoding="utf-8", errors="ignore")


def _load_runs(path: Path) -> pd.DataFrame:
    """Return the metrics DataFrame from ``path`` with normalized column aliases."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Metrics CSV is empty; run the pipeline first.")
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
        "actual_positive_rate": "actual_positive_rate",
        "pred_positive_rate": "pred_positive_rate",
    }
    for src, target in alias_map.items():
        if src in df.columns:
            values = pd.to_numeric(df[src], errors="coerce")
            if target in df.columns:
                df[target] = df[target].combine_first(values)
            else:
                df[target] = values
    timestamp_cols = ["created_at", "run_timestamp", "timestamp"]
    created = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in timestamp_cols:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True)
            if ts.notna().any():
                created = created.combine_first(ts)
    if created.isna().all():
        created = pd.date_range("2000-01-01", periods=len(df), freq="h")
    df["created_at"] = created
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
        "actual_positive_rate",
        "pred_positive_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "run_id" not in df.columns:
        df["run_id"] = [f"run_{i}" for i in range(len(df))]
    else:
        df["run_id"] = df["run_id"].fillna(method="ffill").fillna(method="bfill")
    if "model_name" not in df.columns:
        df["model_name"] = "winprob_model"
    else:
        df["model_name"] = df["model_name"].fillna("winprob_model")
    return df


def _current_evaluation_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Return current-schema full evaluation rows, excluding calibration snapshots and stale runs."""
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


def _best_run(df: pd.DataFrame) -> pd.Series:
    """Pick the run with highest AUC / lowest Brier / lowest LogLoss."""
    sort_columns: list[str] = []
    ascending: list[bool] = []
    for col, asc in (("auc", False), ("brier", True), ("logloss", True)):
        if col in df.columns:
            sort_columns.append(col)
            ascending.append(asc)
    if not sort_columns:
        raise ValueError("overall_metrics.csv lacks auc/brier/logloss columns.")
    df_sorted = df.sort_values(by=sort_columns, ascending=ascending)
    return df_sorted.iloc[0]


def _latest_run(df: pd.DataFrame) -> pd.Series:
    """Pick the most recent run after metric filtering."""
    if "created_at" not in df.columns:
        raise ValueError("overall_metrics.csv lacks created_at/run_timestamp/timestamp columns.")
    return df.sort_values("created_at").iloc[-1]


def _baseline_run(df: pd.DataFrame, best_idx: int) -> Optional[pd.Series]:
    """Return the earliest run that is not the best run, used for comparison text."""
    df_chron = df.sort_values("created_at")
    for _, row in df_chron.iterrows():
        if row.name != best_idx:
            return row
    return None


def _format_value(value: Optional[float], digits: int = 3) -> str:
    """Format metric values while preserving '-' for NaN."""
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def _build_metrics_table(best: pd.Series) -> str:
    """Render the markdown metrics table for the README."""
    rows = [
        ("Accuracy", _format_value(best.get("acc"))),
        ("AUC", _format_value(best.get("auc"))),
        ("Brier", _format_value(best.get("brier"))),
        ("LogLoss", _format_value(best.get("logloss"))),
        ("MAE", _format_value(best.get("mae"), digits=2)),
        ("RMSE", _format_value(best.get("rmse"), digits=2)),
        ("n", _format_value(best.get("n_samples"), digits=0)),
    ]
    table = ["| Metric | Value |", "|--------|-------|"]
    table.extend([f"| {label} | {value} |" for label, value in rows])
    return "\n".join(table)


def _compose_section(best: pd.Series, baseline: Optional[pd.Series]) -> str:
    """Compose the full README section describing the current validated run."""
    best_date = best.get("created_at")
    best_date_str = best_date.strftime("%Y-%m-%d") if pd.notna(best_date) else "unknown date"
    summary = (
        f"The latest validated win-probability evaluation run is **{best.get('model_name')} ({best.get('run_id')})** "
        f"from {best_date_str}. It logged AUC={_format_value(best.get('auc'))}, Brier={_format_value(best.get('brier'))}, "
        f"LogLoss={_format_value(best.get('logloss'))}, Accuracy={_format_value(best.get('acc'))}, "
        f"F1={_format_value(best.get('f1'))}, "
        f"MAE={_format_value(best.get('mae'), digits=2)}, and RMSE={_format_value(best.get('rmse'), digits=2)}."
    )
    comparison = ""
    if baseline is not None:
        delta_auc = best.get("auc") - baseline.get("auc") if pd.notna(baseline.get("auc")) else None
        delta_brier = best.get("brier") - baseline.get("brier") if pd.notna(baseline.get("brier")) else None
        if delta_auc is not None and delta_brier is not None:
            if abs(delta_auc) < 0.002 and abs(delta_brier) < 0.002:
                comparison = (
                    f" Compared to the earliest current-schema baseline "
                    f"({baseline.get('model_name')} / {baseline.get('run_id')}), performance is essentially flat."
                )
            else:
                comparison = (
                    f" Compared to the earliest current-schema baseline "
                    f"({baseline.get('model_name')} / {baseline.get('run_id')}), "
                    f"AUC changed by {delta_auc:+.3f} and Brier changed by {delta_brier:+.3f} "
                    "(lower Brier is better)."
                )
    bullets = [
        f"- **AUC ~{_format_value(best.get('auc'))}** - ranking quality for winners vs. losers.",
        f"- **Brier ~{_format_value(best.get('brier'))}** - probability calibration/error for game winners.",
        f"- **F1 ~{_format_value(best.get('f1'))}** - balance between precision and recall on home-win calls.",
    ]
    metrics_table = _build_metrics_table(best)
    shap_note = (
        "### Explainability (GPU SHAP)\n"
        "We compute GPU-accelerated TreeSHAP values (`analysis/shap/*.png`) to confirm which engineered "
        "signals (QB availability deltas, passing EPA trends, opponent-adjusted efficiency, red-zone execution, "
        "and pressure metrics) drove these gains."
    )
    section = [
        "## Model Performance",
        "",
        summary + comparison,
        "",
        "### Metrics Snapshot",
        metrics_table,
        "",
        "### Why it matters",
        "\n".join(bullets),
        "",
        shap_note,
        "",
    ]
    return "\n".join(section).strip() + "\n\n"


def _replace_section(text: str, section: str) -> str:
    """Replace the existing README 'Model Performance' section with ``section``."""
    # Match the current section until the next H2 header (or EOF).
    pattern = re.compile(r"## Model Performance.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(text):
        return pattern.sub(section, text, count=1)
    first_header = re.search(r"^# .*$", text, re.MULTILINE)
    insert_at = first_header.end() if first_header else 0
    prefix = text[:insert_at]
    suffix = text[insert_at:]
    if prefix and not prefix.endswith("\n\n"):
        prefix = prefix.rstrip("\n") + "\n\n"
    return prefix + section + suffix


def main() -> None:
    """Entrypoint for CLI usage; updates README.md with latest metrics."""
    try:
        df = _load_runs(RUNS_CSV)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[update_readme_metrics] {exc}")
        return
    current_runs = _current_evaluation_runs(df)
    latest = _latest_run(current_runs)
    baseline = _baseline_run(current_runs, latest.name)
    section = _compose_section(latest, baseline)
    try:
        text = read_text_with_fallback(README_PATH)
    except FileNotFoundError as exc:
        print(f"[update_readme_metrics] {exc}")
        return
    new_text = _replace_section(text, section)
    write_text_with_fallback(README_PATH, new_text)
    print(
        f"[update_readme_metrics] README updated with latest valid run {latest.get('run_id')} "
        f"(AUC={latest.get('auc'):.3f}, Brier={latest.get('brier'):.3f})."
    )


if __name__ == "__main__":
    main()
