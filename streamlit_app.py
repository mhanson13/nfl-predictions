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

from __future__ import annotations



import json

import os

import re

import subprocess

import textwrap

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Tuple



import numpy as np

import pandas as pd

import streamlit as st

from dateutil import tz



BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_DIR = BASE_DIR / "predictions"

ANALYSIS_DIR = BASE_DIR / "analysis"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPORTRADAR_DIR = PROCESSED_DIR / "sportradar"
PLAYER_ACTUALS_PATH = PROCESSED_DIR / "player_actuals.parquet"


MOUNTAIN_TZ = tz.gettz("America/Denver")



ARTIFACTS = {

    PREDICTIONS_DIR / "predictions_full.csv": "Upcoming team predictions",

    PREDICTIONS_DIR / "predictions_players_offense.csv": "Upcoming offensive projections",

    PREDICTIONS_DIR / "predictions_players_qb.csv": "Upcoming QB projections",

    PREDICTIONS_DIR / "predictions_players_defense.csv": "Upcoming defensive projections",

    PREDICTIONS_DIR / "evaluation" / "overall_metrics.csv": "Evaluation metrics history",

    ANALYSIS_DIR / "volatility_classifier_metrics.json": "Volatility classifier report",

    BASE_DIR / "models" / "isotonic_calibrator.pkl": "Win probability calibrator",

}



PIPELINE_TEMPLATES: Dict[str, str] = {

    "Run full pipeline": "python -m tools.run_pipeline --debug",

    "Refresh data only": "python -m tools.run_pipeline --debug --skip-evaluation --skip-market-roi",

    "Rebuild matchup features": "python -m src.features.build_features --season 2002 2025 --debug",

    "Retrain win probability model": "python -m src.models.train --target win_prob --use-gpu --debug",

    "Generate upcoming predictions": "python -m src.predict.predict_upcoming --season 2025 --week 11 --overwrite --debug",

    "Generate historical predictions": "python -m src.predict.predict_history --seasons 2025 --overwrite --debug",

}



TRANSPARENCY_DOCS = {
    "feature_importance_winprob_full.csv": "Permutation feature importance for the current win probability model (higher = bigger impact).",
    "feature_importance_winprob_baseline.csv": "Legacy win probability importances to compare against the current feature mix.",
    "feature_importance_spread_full.csv": "Permutation feature importance for the point-spread regression model.",
    "feature_importance_spread_baseline.csv": "Older spread model importances kept for before/after comparisons.",
    "feature_lift_summary.json": "Quick summary of how the latest feature set improves lift versus the baseline configuration.",
    "market_benchmark.csv": "Per-game comparison of model win probability versus no-vig Vegas implied probability.",
    "market_benchmark_summary.csv": "Overall and season-level model-vs-Vegas agreement and disagreement metrics.",
    "market_disagreement_roi.csv": "Moneyline ROI when the model disagrees with the Vegas favorite, by probability-edge threshold.",
    "market_roi_moneyline.csv": "Backtest of model picks versus closing moneyline odds, including hit rate and ROI.",
    "market_roi_spread.csv": "Backtest of spread edges, showing cover rate and return on investment.",
    "market_roi_spread_edge_bins.csv": "Spread ROI by absolute model-vs-market margin edge bucket.",
    "market_roi_summary.json": "High-level summary of the market ROI experiments (moneyline + spread).",
    "volatility_classifier_dataset.csv": "Full dataset used to train the active volatility classifier (logreg blend).",
    "volatility_classifier_dataset_logreg.csv": "Logistic regression-ready version of the volatility dataset.",
    "volatility_classifier_dataset_xgb.csv": "Gradient boosting ready volatility dataset (feature engineered for tree models).",
    "volatility_classifier_importance.csv": "Average feature importances for the volatility ensemble.",
    "volatility_classifier_importance_logreg.csv": "Feature contributions for the logistic-regression volatility model.",
    "volatility_classifier_importance_xgb.csv": "Feature gains for the XGBoost volatility model.",
    "volatility_classifier_metrics.json": "Headline precision/recall/AUC metrics for the active volatility classifier.",
    "volatility_classifier_metrics_logreg.json": "Detailed metrics for the logistic-regression volatility model.",
    "volatility_classifier_metrics_xgb.json": "Detailed metrics for the XGBoost volatility model.",
    "volatility_shrink_grid.csv": "Grid search results for how strongly to shrink picks in volatile spots.",
    "volatility_shrink_grid.json": "Summary of the shrink grid sweep—best thresholds and shrink factors.",
    "volatility_slice_metrics.csv": "Breakdowns of model error by weather, travel, rest, and other volatility slices.",
}




st.set_page_config(

    page_title="NFL Analytics Command Center",

    layout="wide",

    initial_sidebar_state="expanded",

)



st.markdown(

    """

    <style>

        body {background-color: #10121a; color: #f5f7ff;}

        .stApp {background-color: #10121a;}

        .stMetric-label, .stMetric-value {color: #f5f7ff !important;}

        .stTabs [role="tablist"] button {background-color: #161a27; color: #f5f7ff;}

        .stTabs [role="tablist"] button[aria-selected="true"] {background-color: #1f2435;}

    </style>

    """,

    unsafe_allow_html=True,

)



st.title("NFL Analytics Command Center")

st.caption("Operational control, transparency, predictions, and performance retrospectives in one place.")





def _read_json(path: Path) -> Dict[str, Any]:

    if not path.exists():

        return {}

    try:

        return json.loads(path.read_text())

    except Exception:

        return {}





def _load_parquet(path: Path) -> Optional[pd.DataFrame]:

    if not path.exists():

        return None

    try:

        return pd.read_parquet(path)

    except Exception:

        return None





@st.cache_data(show_spinner=False)

def load_csv(path: Path) -> Optional[pd.DataFrame]:

    if not path.exists():

        return None

    try:

        df = pd.read_csv(path)

    except Exception:

        return None

    return df





@st.cache_data(show_spinner=False)

def list_predictions_history() -> List[Path]:

    history_dir = PREDICTIONS_DIR / "history"

    if not history_dir.exists():

        return []

    return sorted(history_dir.glob("w*_predictions_history_*.csv"))





@st.cache_data(show_spinner=False)

def load_team_history() -> pd.DataFrame:

    frames: List[pd.DataFrame] = []

    for path in list_predictions_history():

        df = load_csv(path)

        if df is not None and not df.empty:

            df["source_file"] = path.name

            frames.append(df)

    if not frames:

        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    for col in ("season", "week"):

        if col in data.columns:

            data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")

    return data





@st.cache_data(show_spinner=False)
def load_player_predictions(kind: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    # Include the consolidated latest export if it exists.
    latest_path = PREDICTIONS_DIR / f"predictions_players_{kind}.csv"
    latest_df = load_csv(latest_path)
    if latest_df is not None and not latest_df.empty:
        latest_df = latest_df.copy()
        latest_df["source_file"] = latest_path.name
        frames.append(latest_df)

    # Pull historical week-specific exports (w##_predictions_players_*.csv).
    pattern = f"w*_predictions_players_{kind}.csv"
    for path in sorted(PREDICTIONS_DIR.glob(pattern)):
        df = load_csv(path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["source_file"] = data["source_file"].astype(str)

    # Prefer consolidated exports over per-week files when duplicates appear.
    priority_flag = data["source_file"].eq(f"predictions_players_{kind}.csv").astype(int)
    data["_priority"] = priority_flag
    # Stable sort so higher priority rows appear first.
    data = data.sort_values(["_priority", "source_file"], ascending=[False, True])

    dedupe_keys = [col for col in ["season", "week", "game_id", "player_id", "player_name", "team"] if col in data.columns]
    if dedupe_keys:
        data = data.drop_duplicates(subset=dedupe_keys, keep="first")
    data = data.drop(columns="_priority", errors="ignore")

    for col in ("season", "week"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")
    for col in ("player_rank", "games_sampled"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data




def _parse_stats(value: Any) -> Dict[str, Any]:

    if isinstance(value, dict):

        return value

    if isinstance(value, str):

        try:

            return json.loads(value)

        except Exception:

            return {}

    return {}





def _clean_last_name(name: str) -> str:

    cleaned = re.sub(r"[^A-Za-z\s\-]", "", name or "")

    parts = cleaned.strip().split()

    return re.sub(r"[^A-Za-z]", "", parts[-1]).upper() if parts else ""





def _first_initial(name: str) -> str:

    cleaned = name.strip().replace("-", " ")

    return cleaned[0].upper() if cleaned else ""





def _tokenize_prediction_name(name: str) -> Tuple[str, str]:

    if not isinstance(name, str):

        return "", ""

    if "." in name:

        first, last = name.split(".", 1)

    else:

        parts = name.split()

        first, last = (parts[0], parts[-1]) if parts else ("", "")

    return _first_initial(first), re.sub(r"[^A-Za-z]", "", last.upper())





def _tokenize_actual_name(name: str) -> Tuple[str, str]:

    if not isinstance(name, str):

        return "", ""

    cleaned = name.replace("'", "").replace("-", " ").strip()

    parts = cleaned.split()

    if not parts:

        return "", ""

    first_initial = parts[0][0].upper()

    last_name = re.sub(r"[^A-Za-z]", "", parts[-1]).upper()

    return first_initial, last_name





@st.cache_data(show_spinner=False)
def load_player_actuals() -> Optional[pd.DataFrame]:
    if PLAYER_ACTUALS_PATH.exists():
        stats = _load_parquet(PLAYER_ACTUALS_PATH)
        if stats is None or stats.empty:
            return None
        stats = stats.copy()
        if "stats_dict" not in stats.columns:
            stats["stats_dict"] = stats["stats"].apply(_parse_stats)
        stats["team_alias"] = stats["team_alias"].astype(str).str.upper()
        stats["first_initial"] = stats["player_name"].apply(_first_initial)
        stats["last_name"] = stats["player_name"].apply(_clean_last_name)
        return stats

    stats_path = SPORTRADAR_DIR / "game_player_stats.parquet"
    roster_path = SPORTRADAR_DIR / "roster_players.parquet"
    if not stats_path.exists() or not roster_path.exists():
        return None
    stats = _load_parquet(stats_path)
    roster = _load_parquet(roster_path)
    if stats is None or roster is None or stats.empty or roster.empty:
        return None

    alias_map = roster[["team_id", "team_alias"]].drop_duplicates()
    stats = stats.merge(alias_map, on="team_id", how="left")
    stats["stats_dict"] = stats["stats"].apply(_parse_stats)
    stats["first_initial"] = stats["player_name"].apply(_first_initial)
    stats["last_name"] = stats["player_name"].apply(_clean_last_name)
    return stats




def _aggregate_offense_actuals(stats: pd.DataFrame) -> pd.DataFrame:

    rushing = stats[stats["stat_category"] == "rushing"].copy()

    rushing["rush_yards"] = rushing["stats_dict"].apply(lambda d: float(d.get("yards", 0) or 0))



    receiving = stats[stats["stat_category"] == "receiving"].copy()

    receiving["rec_yards"] = receiving["stats_dict"].apply(lambda d: float(d.get("yards", 0) or 0))



    defense = (

        pd.concat(

            [rushing[["game_id", "team_alias", "player_name", "first_initial", "last_name", "rush_yards"]],

             receiving[["game_id", "team_alias", "player_name", "first_initial", "last_name", "rec_yards"]]],

            axis=0,

        )

        .groupby(["game_id", "team_alias", "player_name", "first_initial", "last_name"], as_index=False)

        .sum(min_count=1)

    )

    defense["actual_total_yards"] = defense.get("rush_yards", 0).fillna(0) + defense.get("rec_yards", 0).fillna(0)

    return defense





def _aggregate_passing_actuals(stats: pd.DataFrame) -> pd.DataFrame:

    passing = stats[stats["stat_category"] == "passing"].copy()

    passing["pass_yards"] = passing["stats_dict"].apply(lambda d: float(d.get("yards", 0) or 0))

    return passing[

        ["game_id", "team_alias", "player_name", "first_initial", "last_name", "pass_yards"]

    ].rename(columns={"pass_yards": "actual_passing_yards"})





def _aggregate_defense_actuals(stats: pd.DataFrame) -> pd.DataFrame:

    defense = stats[stats["stat_category"] == "defense"].copy()

    defense["sacks"] = defense["stats_dict"].apply(lambda d: float(d.get("sacks", 0) or 0))

    defense["qb_hits"] = defense["stats_dict"].apply(lambda d: float(d.get("qb_hits", 0) or 0))

    return defense[

        ["game_id", "team_alias", "player_name", "first_initial", "last_name", "sacks", "qb_hits"]

    ].rename(columns={"sacks": "actual_sacks", "qb_hits": "actual_qb_hits"})





def _merge_prediction_actuals(

    preds: pd.DataFrame,

    actuals: pd.DataFrame,

    team_col: str = "team",

    value_map: Optional[Dict[str, str]] = None,

) -> pd.DataFrame:

    if preds.empty or actuals is None or actuals.empty:

        return pd.DataFrame()



    preds = preds.copy()

    preds["team"] = preds[team_col].astype(str).str.upper()

    preds["first_initial"], preds["last_name"] = zip(*preds["player_name"].map(_tokenize_prediction_name))



    actuals = actuals.copy()

    actuals = actuals.rename(columns={"team_alias": "team"})

    merged = preds.merge(

        actuals,

        on=["game_id", "team", "first_initial", "last_name"],

        how="left",

        suffixes=("", "_actual"),

    )

    if value_map:

        for pred_col, actual_col in value_map.items():

            if pred_col in merged.columns and actual_col in merged.columns:

                merged[f"{pred_col}_error"] = merged[actual_col] - merged[pred_col]

    return merged





@st.cache_data(show_spinner=False)

def build_offense_evaluation() -> pd.DataFrame:

    preds = load_player_predictions("offense")

    stats = load_player_actuals()

    if preds.empty or stats is None:

        return pd.DataFrame()

    offense_actual = _aggregate_offense_actuals(stats)

    merged = _merge_prediction_actuals(

        preds,

        offense_actual,

        value_map={"projected_total_yards": "actual_total_yards"},

    )

    return merged





@st.cache_data(show_spinner=False)

def build_qb_evaluation() -> pd.DataFrame:

    preds = load_player_predictions("qb")

    stats = load_player_actuals()

    if preds.empty or stats is None:

        return pd.DataFrame()

    passing_actual = _aggregate_passing_actuals(stats)

    merged = _merge_prediction_actuals(

        preds,

        passing_actual,

        value_map={"projected_passing_yards": "actual_passing_yards"},

    )

    return merged





@st.cache_data(show_spinner=False)

def build_defense_evaluation() -> pd.DataFrame:

    preds = load_player_predictions("defense")

    stats = load_player_actuals()

    if preds.empty or stats is None:

        return pd.DataFrame()

    defense_actual = _aggregate_defense_actuals(stats)

    merged = _merge_prediction_actuals(

        preds,

        defense_actual,

        value_map={"projected_sacks": "actual_sacks"},

    )

    return merged





def _calculate_confidence(series: pd.Series) -> pd.Series:

    if series is None or series.empty:

        return series

    max_val = series.max()

    if not max_val or np.isnan(max_val):

        return pd.Series(np.nan, index=series.index)

    return np.round(np.clip(series / max_val, 0, 1), 3)





def run_command(cmd: str) -> Tuple[int, str, str]:

    try:

        result = subprocess.run(

            cmd,

            cwd=BASE_DIR,

            shell=True,

            capture_output=True,

            text=True,

            encoding="utf-8",

        )

        return result.returncode, result.stdout, result.stderr

    except Exception as exc:  # pragma: no cover - defensive

        return 1, "", str(exc)





def artifact_summary() -> pd.DataFrame:
    rows = []

    for path, description in ARTIFACTS.items():

        if path.exists():

            stat = path.stat()

            modified = datetime.fromtimestamp(stat.st_mtime)

            rows.append(

                {

                    "artifact": path.name,

                    "description": description,

                    "location": str(path.relative_to(BASE_DIR)),

                    "last_modified": modified.strftime("%Y-%m-%d %H:%M:%S"),

                    "age_hours": round((datetime.now() - modified).total_seconds() / 3600, 1),

                    "size_kb": round(stat.st_size / 1024, 1),

                }

            )

        else:

            rows.append(

                {

                    "artifact": path.name,

                    "description": description,

                    "location": str(path.relative_to(BASE_DIR)),

                    "last_modified": "missing",

                    "age_hours": None,

                    "size_kb": None,

                }

            )

    return pd.DataFrame(rows)


def _coerce_metric(value: Any) -> Optional[float]:
    """Convert assorted metric representations to float or None."""
    if value is None:
        return None
    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(coerced) if pd.notna(coerced) else None


def _format_metric(value: Any, places: int = 3) -> str:
    num = _coerce_metric(value)
    if num is None:
        return "–"
    return f"{num:.{places}f}"


METRIC_COLUMNS = ["accuracy", "auc", "brier", "log_loss", "mae_margin", "rmse_margin"]


def _prepare_metrics_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["stage_norm"] = working.get("stage", "").fillna("").astype(str).str.lower()
    working["run_timestamp"] = pd.to_datetime(working.get("run_timestamp"), errors="coerce")
    working["timestamp"] = pd.to_datetime(working.get("timestamp"), errors="coerce")
    sort_key = working["run_timestamp"].where(working["run_timestamp"].notna(), working["timestamp"])
    working["_sort_key"] = sort_key.fillna(pd.Timestamp.utcnow())
    return working


def _pick_metric_row(
    working: pd.DataFrame,
    column: str,
    prefer_stage: Optional[str] = "calibrated",
) -> Optional[pd.Series]:
    if column not in working.columns:
        return None
    subset = working[working[column].notna()]
    if subset.empty:
        return None
    if prefer_stage:
        preferred = subset[subset["stage_norm"] == prefer_stage]
        if not preferred.empty:
            subset = preferred
    subset = subset.sort_values("_sort_key", ascending=False)
    return subset.iloc[0]


def _latest_metrics_summary(
    df: pd.DataFrame,
    prefer_stage: Optional[str] = "calibrated",
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if df is None or df.empty:
        return summary
    working = _prepare_metrics_frame(df)

    def _value(col: str) -> Optional[Any]:
        row = _pick_metric_row(working, col, prefer_stage)
        return None if row is None else row.get(col)

    for field in ["samples", "n_games"] + METRIC_COLUMNS:
        summary[field] = _value(field)

    context_row = None
    for key in ("brier", "accuracy", "log_loss", "mae_margin", "auc"):
        row = _pick_metric_row(working, key, prefer_stage)
        if row is not None:
            context_row = row
            break
    if context_row is None:
        context_row = working.sort_values("_sort_key", ascending=False).iloc[0]

    summary["stage"] = context_row.get("stage")
    summary["timestamp"] = context_row.get("timestamp") or context_row.get("run_timestamp")
    return summary


def _format_count(value: Any) -> str:
    num = _coerce_metric(value)
    if num is None:
        return "–"
    return f"{int(round(num)):,}"




def render_command_runner():

    st.subheader("Command runner")

    template = st.selectbox("Pick a template command", list(PIPELINE_TEMPLATES.keys()))

    command = st.text_area(

        "Command",

        value=PIPELINE_TEMPLATES[template],

        height=70,

        help="Commands run from the repository root. Adjust arguments as needed before launching.",

    )

    if st.button("Execute", type="primary"):

        with st.spinner("Running command..."):

            code, stdout, stderr = run_command(command)

        st.write(f"Exit code: {code}")

        if stdout:

            st.markdown("**stdout**")

            st.code(stdout, language="bash")

        if stderr:

            st.markdown("**stderr**")

            st.code(stderr, language="bash")

        if code != 0:

            st.error("Command reported a non-zero exit code. Check logs above.")





def render_pipeline_tab():
    col_status, col_controls = st.columns([1.4, 1.0], gap="large")
    with col_status:
        st.subheader("Operational snapshot")
        df = artifact_summary()
        st.dataframe(
            df.sort_values("last_modified", ascending=False),
            use_container_width=True,
            height=350,
        )
        metrics_path = PREDICTIONS_DIR / "evaluation" / "overall_metrics.csv"
        metrics = load_csv(metrics_path)
        summary = _latest_metrics_summary(metrics)
        if summary:
            st.markdown("**Latest calibrated performance**")
            cols = st.columns(4)
            cols[0].metric("Accuracy", _format_metric(summary.get("accuracy")))
            cols[1].metric("AUC", _format_metric(summary.get("auc")))
            cols[2].metric("Brier", _format_metric(summary.get("brier")))
            cols[3].metric("LogLoss", _format_metric(summary.get("log_loss")))
    with col_controls:
        render_command_runner()




def render_transparency_tab():
    st.subheader("Model diagnostics & documentation")
    metrics = load_csv(PREDICTIONS_DIR / "evaluation" / "overall_metrics.csv")
    summary = _latest_metrics_summary(metrics)
    if summary:
        st.markdown("**Latest calibrated run**")
        cols = st.columns(6)
        samples = summary.get("samples") or summary.get("n_games")
        with cols[0]:
            st.metric(
                label="Samples",
                value=_format_count(samples),
                help="Number of games included in the calibrated evaluation window.",
            )
        with cols[1]:
            st.metric(
                label="Accuracy",
                value=_format_metric(summary.get("accuracy")),
                help="Share of games where the predicted winner matched the actual winner.",
            )
        with cols[2]:
            st.metric(
                label="AUC",
                value=_format_metric(summary.get("auc")),
                help="Area under the ROC curve for win probability discrimination.",
            )
        with cols[3]:
            st.metric(
                label="Brier",
                value=_format_metric(summary.get("brier")),
                help="Average squared error between predicted win probability and actual outcome.",
            )
        with cols[4]:
            st.metric(
                label="LogLoss",
                value=_format_metric(summary.get("log_loss")),
                help="Cross-entropy error for win probability calibration (lower is better).",
            )
        with cols[5]:
            st.metric(
                label="MAE margin",
                value=_format_metric(summary.get("mae_margin")),
                help="Mean absolute error for predicted home margin versus actual margin.",
            )
    else:
        st.info("No calibrated evaluation rows with populated metrics were found.")


    st.markdown("### Analysis folder reference")
    st.caption(
        "Hover the metric cards above for quick tooltips. In plain English: accuracy tells us how often we picked the "
        "right winner; AUC shows how well the model separates favorites from underdogs; Brier measures average miss on "
        "the win odds; LogLoss penalizes bad confidence; MAE margin is the typical spread miss in points."
    )
    analysis_files = sorted(ANALYSIS_DIR.glob("*.csv"))

    if not analysis_files:

        st.info("No analysis CSV files found.")

    else:

        for path in analysis_files:

            name = path.name

            explanation = TRANSPARENCY_DOCS.get(name, "Description pending.")

            with st.expander(f"{name}"):

                st.caption(explanation)

                df = load_csv(path)

                if df is None or df.empty:

                    st.info("File is empty or could not be parsed.")

                else:

                    preview_rows = min(len(df), 200)

                    st.dataframe(df.head(preview_rows), use_container_width=True, height=240)





def render_predictions_tab():

    st.subheader("Upcoming week outlook")

    offense = load_player_predictions("offense")

    qbs = load_player_predictions("qb")

    defense = load_player_predictions("defense")

    if offense.empty and qbs.empty and defense.empty:

        st.info("Upcoming prediction files are not available. Generate them from the pipeline tab first.")

        return



    season = int(offense["season"].max()) if not offense.empty else int(qbs["season"].max())

    week = int(offense["week"].max()) if not offense.empty else int(qbs["week"].max())

    st.caption(f"Showing projections for season **{season}**, week **{week}**.")



    min_games = st.slider(
        "Minimum games sampled",
        1,
        100,
        16,
        help="Filter out projections backed by very small samples.",
    )
    max_rank = st.slider("Maximum player rank", 1, 50, 2)


    st.divider()

    st.markdown("### Player projection leaders")

    if not offense.empty:
        off = offense[
            (offense["games_sampled"] >= min_games)
            & (offense["player_rank"] <= max_rank)
            & (offense["season"] == season)
            & (offense["week"] == week)
        ].copy()
        if not off.empty:
            off["confidence"] = _calculate_confidence(off["games_sampled"])
            off_cols = [
                "player_name",
                "team",
                "kickoff_mt",
                "projected_total_yards",
                "projected_rushing_yards",
                "projected_receiving_yards",
                "projected_total_tds",
                "projected_rushing_tds",
                "projected_receiving_tds",
                "games_sampled",
                "player_rank",
                "confidence",
            ]
            available_off_cols = [c for c in off_cols if c in off.columns]
            top = off.sort_values("projected_total_yards", ascending=False).head(15)[available_off_cols]
            st.markdown("**Top total yards (rush + receive)**")
            st.dataframe(top, use_container_width=True, height=520)

            td_cols = [
                "player_name",
                "team",
                "kickoff_mt",
                "projected_total_tds",
                "projected_rushing_tds",
                "projected_receiving_tds",
                "games_sampled",
                "player_rank",
                "confidence",
            ]
            available_td_cols = [c for c in td_cols if c in off.columns]
            if available_td_cols:
                td_table = off.sort_values("projected_total_tds", ascending=False).head(15)[available_td_cols]
                st.markdown("**Top projected total touchdowns**")
                st.dataframe(td_table, use_container_width=True, height=480)
        else:
            st.info("No offensive projections after applying filters.")
    else:
        st.info("Upcoming offensive projections are unavailable.")

    st.divider()

    if not qbs.empty:
        qb = qbs[
            (qbs["games_sampled"] >= min_games)
            & (qbs["player_rank"] <= max_rank)
            & (qbs["season"] == season)
            & (qbs["week"] == week)
        ].copy()
        if not qb.empty:
            qb["confidence"] = _calculate_confidence(qb["games_sampled"])
            qb_cols = [
                "player_name",
                "team",
                "kickoff_mt",
                "projected_passing_yards",
                "projected_rushing_yards",
                "projected_passing_tds",
                "projected_rushing_tds",
                "projected_interceptions",
                "projected_completions",
                "games_sampled",
                "player_rank",
                "confidence",
            ]
            available_qb_cols = [c for c in qb_cols if c in qb.columns]
            top = qb.sort_values("projected_passing_yards", ascending=False).head(15)[available_qb_cols]
            st.markdown("**Top projected passing yards**")
            st.dataframe(top, use_container_width=True, height=520)
        else:
            st.info("No QB projections after applying filters.")
    else:
        st.info("Upcoming QB projections are unavailable.")

    st.divider()

    if not defense.empty:
        df_def = defense[
            (defense["games_sampled"] >= min_games)
            & (defense["player_rank"] <= max_rank)
            & (defense["season"] == season)
            & (defense["week"] == week)
        ].copy()
        if not df_def.empty:
            df_def["confidence"] = _calculate_confidence(df_def["games_sampled"])
            df_def_cols = [
                "player_name",
                "team",
                "kickoff_mt",
                "projected_sacks",
                "projected_qb_hits",
                "projected_tfl",
                "games_sampled",
                "player_rank",
                "confidence",
            ]
            available_def_cols = [c for c in df_def_cols if c in df_def.columns]
            top = df_def.sort_values("projected_sacks", ascending=False).head(15)[available_def_cols]
            st.markdown("**Top projected sacks**")
            st.dataframe(top, use_container_width=True, height=520)
        else:
            st.info("No defensive projections after applying filters.")
    else:
        st.info("Upcoming defensive projections are unavailable.")


    st.divider()



    team_preds = load_csv(PREDICTIONS_DIR / "predictions_full.csv")

    if team_preds is None or team_preds.empty:

        st.info("Team predictions file (`predictions_full.csv`) is unavailable.")

        return



    team = team_preds.copy()

    team["season"] = pd.to_numeric(team.get("season"), errors="coerce")

    team["week"] = pd.to_numeric(team.get("week"), errors="coerce")

    latest_season = team["season"].dropna().max()

    latest_week = team.loc[team["season"] == latest_season, "week"].dropna().max()

    subset = team[(team["season"] == latest_season) & (team["week"] == latest_week)].copy()
    if subset.empty:
        st.info("No team predictions available for the latest week.")
        return

    # Prefer the most recently generated prediction per game.
    sort_cols = []
    for cand in ("prediction_generated_at", "generated_at", "created_at"):
        if cand in subset.columns:
            subset[cand] = pd.to_datetime(subset[cand], errors="coerce")
            sort_cols.append(cand)
    if sort_cols:
        subset = subset.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    subset = subset.drop_duplicates(subset=["game_id"], keep="first")

    subset["home_win_prob"] = pd.to_numeric(subset.get("home_win_prob"), errors="coerce")
    subset["away_win_prob"] = 1 - subset["home_win_prob"]
    subset["favorite"] = np.where(

        subset["home_win_prob"] >= 0.5, subset["home_team"], subset["away_team"]

    )

    subset["favorite_prob"] = subset[["home_win_prob", "away_win_prob"]].max(axis=1)

    subset["favorite_margin"] = pd.to_numeric(subset.get("pred_home_margin"), errors="coerce")

    subset.loc[subset["favorite"] == subset["away_team"], "favorite_margin"] *= -1



    st.markdown(

        f"### Team win probabilities (season {int(latest_season)} week {int(latest_week)})"

    )

    team_cols = [

        "favorite",

        "favorite_prob",

        "favorite_margin",

        "home_team",

        "away_team",

        "home_win_prob",

        "pred_home_margin",

        "volatility_prob",

        "kickoff_mt",

        "pick_expl",

    ]

    available_team_cols = [c for c in team_cols if c in subset.columns]

    st.dataframe(

        subset.sort_values("favorite_prob", ascending=False)[available_team_cols],

        use_container_width=True,

        height=420,

    )






def summarise_prediction_quality(df: pd.DataFrame, pred_col: str, actual_col: str) -> pd.DataFrame:
    if df.empty or pred_col not in df.columns or actual_col not in df.columns:
        return pd.DataFrame()
    valid = df[pd.notna(df[actual_col])]
    if valid.empty:
        return pd.DataFrame()
    valid = valid.copy()
    valid["abs_error"] = (valid[pred_col] - valid[actual_col]).abs()
    valid["squared_error"] = (valid[pred_col] - valid[actual_col]) ** 2
    by_week = (
        valid.groupby(["season", "week"], as_index=False)
        .agg(
            count=("player_name", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        )
    )
    return by_week

def render_history_tab():

    team_history = load_team_history()

    offense_eval = build_offense_evaluation()

    qb_eval = build_qb_evaluation()

    defense_eval = build_defense_evaluation()



    team_tab, offense_tab, qb_tab, defense_tab = st.tabs(["Team", "Offense", "QB", "Defense"])



    with team_tab:

        if team_history.empty:

            st.info("Team history files are unavailable.")

        else:

            df = team_history.copy()

            df["home_win"] = (pd.to_numeric(df["home_margin"], errors="coerce") > 0).astype(int)

            df["pred_win"] = (pd.to_numeric(df["home_win_prob"], errors="coerce") >= 0.5).astype(int)

            df["brier_component"] = (pd.to_numeric(df["home_win_prob"], errors="coerce") - df["home_win"]) ** 2

            df["correct"] = (df["home_win"] == df["pred_win"]).astype(int)

            df["margin_error"] = (pd.to_numeric(df["pred_home_margin"], errors="coerce") - pd.to_numeric(df["home_margin"], errors="coerce")).abs()

            summary = (

                df.groupby(["season", "week"])

                .agg(

                    games=("game_id", "count"),

                    accuracy=("correct", "mean"),

                    brier=("brier_component", "mean"),

                    mae_margin=("margin_error", "mean"),

                )

                .reset_index()

            )

            st.markdown("**Weekly performance (team predictions)**")

            sort_cols = [c for c in ["season", "week"] if c in summary.columns]

            summary_sorted = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else summary

            st.dataframe(summary_sorted, use_container_width=True, height=420)

            st.markdown("**Trend – Accuracy**")

            trend_df = summary.sort_values(["season", "week"]).copy()

            trend_df["season_week"] = trend_df["season"].astype(str) + "-W" + trend_df["week"].astype(str)

            trend_df = trend_df.set_index("season_week")

            st.line_chart(trend_df[["accuracy"]])

            st.markdown("**Trend – MAE Margin**")

            st.line_chart(trend_df[["mae_margin"]])



    with offense_tab:
        if offense_eval.empty:
            st.info("Need actual data or offensive projections to compute evaluation.")
        else:
            summary = summarise_prediction_quality(offense_eval, "projected_total_yards", "actual_total_yards")
            coverage = offense_eval["actual_total_yards"].notna().mean()
            st.metric("Coverage (matched players)", f"{coverage * 100:.1f}%")
            if summary.empty:
                st.info("No matched offensive players with actual stats yet.")
            else:
                sort_cols = [c for c in ("season", "week") if c in summary.columns]
                summary_view = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else summary
                st.dataframe(summary_view, use_container_width=True, height=420)
            seasons = (
                [int(x) for x in sorted(offense_eval["season"].dropna().unique())]
                if "season" in offense_eval.columns and not offense_eval.empty
                else []
            )
            if seasons:
                st.markdown("**Detailed comparison**")
                sel_season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="off_season_select")
                weeks_raw = offense_eval.loc[offense_eval["season"] == sel_season, "week"].dropna().unique()
                weeks = [int(w) for w in sorted(weeks_raw)]
                sel_week = st.selectbox("Week", weeks, index=len(weeks) - 1, key="off_week_select")
                detail = offense_eval[

                    (offense_eval["season"] == sel_season) & (offense_eval["week"] == sel_week)

                ].copy()

                if detail.empty:

                    st.info("No offensive projections matched for the selected season/week.")

                else:

                    detail = detail[pd.notna(detail["actual_total_yards"])].copy()

                    if detail.empty:

                        st.info("Actual stats not yet available for the selected week.")

                    else:

                        if "projected_total_yards_error" in detail.columns:

                            detail["total_yards_error"] = detail["projected_total_yards_error"]

                            detail["abs_error"] = detail["total_yards_error"].abs()

                        cols = [

                            "player_name",

                            "team",

                            "kickoff_mt",

                            "projected_total_yards",

                            "actual_total_yards",

                            "total_yards_error",

                            "abs_error",

                            "projected_total_tds",

                            "games_sampled",

                            "player_rank",

                        ]

                        available_cols = [c for c in cols if c in detail.columns]

                        st.dataframe(

                            detail[available_cols].sort_values("abs_error"),

                            use_container_width=True,

                            height=360,

                        )



    with qb_tab:
        if qb_eval.empty:
            st.info("Need actual data or QB projections to compute evaluation.")
        else:
            summary = summarise_prediction_quality(qb_eval, "projected_passing_yards", "actual_passing_yards")
            coverage = qb_eval["actual_passing_yards"].notna().mean()
            st.metric("Coverage (matched players)", f"{coverage * 100:.1f}%")
            if summary.empty:
                st.info("No matched QB stats available for evaluation yet.")
            else:
                sort_cols = [c for c in ("season", "week") if c in summary.columns]
                summary_view = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else summary
                st.dataframe(summary_view, use_container_width=True, height=420)
            seasons = (
                [int(x) for x in sorted(qb_eval["season"].dropna().unique())]
                if "season" in qb_eval.columns and not qb_eval.empty
                else []
            )
            if seasons:
                st.markdown("**Detailed comparison**")
                sel_season = st.selectbox("Season ", seasons, index=len(seasons) - 1, key="qb_season_select")
                weeks_raw = qb_eval.loc[qb_eval["season"] == sel_season, "week"].dropna().unique()
                weeks = [int(w) for w in sorted(weeks_raw)]
                sel_week = st.selectbox("Week ", weeks, index=len(weeks) - 1, key="qb_week_select")
                detail = qb_eval[

                    (qb_eval["season"] == sel_season) & (qb_eval["week"] == sel_week)

                ].copy()

                if detail.empty or detail["actual_passing_yards"].isna().all():

                    st.info("Actual stats not yet available for the selected week.")

                else:

                    detail = detail[pd.notna(detail["actual_passing_yards"])].copy()

                    if "projected_passing_yards_error" in detail.columns:

                        detail["passing_yards_error"] = detail["projected_passing_yards_error"]

                        detail["abs_error"] = detail["passing_yards_error"].abs()

                    cols = [

                        "player_name",

                        "team",

                        "kickoff_mt",

                        "projected_passing_yards",

                        "actual_passing_yards",

                        "passing_yards_error",

                        "abs_error",

                        "projected_passing_tds",

                        "projected_interceptions",

                        "games_sampled",

                        "player_rank",

                    ]

                    available_cols = [c for c in cols if c in detail.columns]

                    st.dataframe(

                        detail[available_cols].sort_values("abs_error"),

                        use_container_width=True,

                        height=360,

                    )



    with defense_tab:
        if defense_eval.empty:
            st.info("Need actual data or defensive projections to compute evaluation.")
        else:
            summary = summarise_prediction_quality(defense_eval, "projected_sacks", "actual_sacks")
            coverage = defense_eval["actual_sacks"].notna().mean()
            st.metric("Coverage (matched players)", f"{coverage * 100:.1f}%")
            if summary.empty:
                st.info("No matched defensive stats available for evaluation yet.")
            else:
                sort_cols = [c for c in ("season", "week") if c in summary.columns]
                summary_view = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else summary
                st.dataframe(summary_view, use_container_width=True, height=420)
            seasons = (
                [int(x) for x in sorted(defense_eval["season"].dropna().unique())]
                if "season" in defense_eval.columns and not defense_eval.empty
                else []
            )
            if seasons:
                st.markdown("**Detailed comparison**")
                sel_season = st.selectbox("Season  ", seasons, index=len(seasons) - 1, key="def_season_select")
                weeks_raw = defense_eval.loc[defense_eval["season"] == sel_season, "week"].dropna().unique()
                weeks = [int(w) for w in sorted(weeks_raw)]
                sel_week = st.selectbox("Week  ", weeks, index=len(weeks) - 1, key="def_week_select")
                detail = defense_eval[

                    (defense_eval["season"] == sel_season) & (defense_eval["week"] == sel_week)

                ].copy()

                if detail.empty or detail["actual_sacks"].isna().all():

                    st.info("Actual stats not yet available for the selected week.")

                else:

                    detail = detail[pd.notna(detail["actual_sacks"])].copy()

                    if "projected_sacks_error" in detail.columns:

                        detail["sacks_error"] = detail["projected_sacks_error"]

                        detail["abs_error"] = detail["sacks_error"].abs()

                    cols = [

                        "player_name",

                        "team",

                        "kickoff_mt",

                        "projected_sacks",

                        "actual_sacks",

                        "sacks_error",

                        "abs_error",

                        "projected_qb_hits",

                        "games_sampled",

                        "player_rank",

                    ]

                    available_cols = [c for c in cols if c in detail.columns]

                    st.dataframe(

                        detail[available_cols].sort_values("abs_error"),

                        use_container_width=True,

                        height=360,

                    )





def render_validation_tab():
    """Render live validation metrics dashboard."""
    st.header("🎯 Live Validation Dashboard")
    st.caption("Real-time tracking of model performance, calibration, and market edge")
    
    # Season selector
    current_season = 2025
    season = st.selectbox("Select Season", [2025, 2024, 2023, 2022, 2021], index=0)
    
    # Load validation data
    validation_dir = Path("analysis")
    
    # === SECTION 1: Current Season Performance ===
    st.subheader("📊 Current Season Performance")
    
    # Try to load live tracking metrics
    predictions_log_dir = Path("predictions_log") / str(season)
    
    if predictions_log_dir.exists():
        # Find all weekly metrics
        weekly_metrics = []
        for metrics_file in sorted(predictions_log_dir.glob("week_*_metrics.json")):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    metrics['week'] = int(metrics_file.stem.split('_')[1])
                    weekly_metrics.append(metrics)
            except Exception:
                continue
        
        if weekly_metrics:
            # Display current season summary
            col1, col2, col3, col4 = st.columns(4)
            
            total_games = sum(m.get('n_predictions', 0) for m in weekly_metrics)
            avg_accuracy = np.mean([m.get('accuracy', 0) for m in weekly_metrics if 'accuracy' in m])
            avg_brier = np.mean([m.get('brier_score', 0) for m in weekly_metrics if 'brier_score' in m])
            avg_auc = np.mean([m.get('auc', 0) for m in weekly_metrics if 'auc' in m])
            
            with col1:
                st.metric("Total Games", f"{total_games}")
            with col2:
                st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
            with col3:
                st.metric("Avg Brier Score", f"{avg_brier:.4f}")
            with col4:
                st.metric("Avg AUC", f"{avg_auc:.3f}")
            
            # Weekly trend chart
            st.markdown("**Weekly Performance Trend**")
            weekly_df = pd.DataFrame(weekly_metrics)
            if 'accuracy' in weekly_df.columns and 'week' in weekly_df.columns:
                chart_data = weekly_df[['week', 'accuracy', 'brier_score']].set_index('week')
                st.line_chart(chart_data)
        else:
            st.info(f"No weekly metrics found for {season}. Run live tracking to generate metrics.")
    else:
        st.info(f"No predictions log found for {season}. Start logging predictions with `live_tracking.py`")
    
    # === SECTION 2: Calibration Health ===
    st.subheader("🎯 Calibration Health")
    
    calibration_dir = validation_dir / "calibration_monitoring"
    calibration_file = calibration_dir / f"{season}_season_metrics.json"
    
    if calibration_file.exists():
        try:
            with open(calibration_file) as f:
                cal_metrics = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                brier = cal_metrics.get('brier_score', 0)
                st.metric("Brier Score", f"{brier:.4f}",
                         delta=f"{brier - 0.20:.4f}" if brier else None,
                         delta_color="inverse")
            
            with col2:
                cal_error = cal_metrics.get('calibration_error', 0)
                st.metric("Calibration Error", f"{cal_error:.4f}",
                         delta="Good" if cal_error < 0.02 else "Check",
                         delta_color="normal" if cal_error < 0.02 else "inverse")
            
            with col3:
                resolution = cal_metrics.get('brier_resolution', 0)
                st.metric("Resolution", f"{resolution:.4f}",
                         delta="High" if resolution > 0.04 else "Moderate")
            
            with col4:
                max_error = cal_metrics.get('max_calibration_error', 0)
                st.metric("Max Cal Error", f"{max_error:.4f}",
                         delta="OK" if max_error < 0.10 else "High",
                         delta_color="normal" if max_error < 0.10 else "inverse")
            
            # Check for drift alerts
            drift_file = calibration_dir / f"{season}_drift_alerts.json"
            if drift_file.exists():
                try:
                    with open(drift_file) as f:
                        alerts = json.load(f)
                    if alerts:
                        st.warning(f"⚠️ {len(alerts)} calibration drift alert(s) detected!")
                        for alert in alerts:
                            st.write(f"- **{alert['metric']}**: {alert['drift']:+.4f} ({alert['severity']})")
                except Exception:
                    pass
            
            # Display reliability diagram if available
            reliability_plot = calibration_dir / f"{season}_season_reliability.png"
            if reliability_plot.exists():
                st.markdown("**Reliability Diagram**")
                st.image(str(reliability_plot), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading calibration metrics: {e}")
    else:
        st.info(f"No calibration metrics for {season}. Run `calibration_monitor.py --season {season} --generate-report`")
    
    # === SECTION 3: Closing Line Value (CLV) ===
    st.subheader("💰 Closing Line Value (CLV)")
    
    clv_dir = validation_dir / "clv_tracking"
    clv_file = clv_dir / f"{season}_moneyline_clv.csv"
    
    if clv_file.exists():
        try:
            clv_df = pd.read_csv(clv_file)
            
            # CLV summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_clv = clv_df['clv'].mean()
                st.metric("Avg CLV", f"{avg_clv:+.2%}",
                         delta="Positive Edge" if avg_clv > 0 else "No Edge",
                         delta_color="normal" if avg_clv > 0 else "inverse")
            
            with col2:
                positive_clv_pct = (clv_df['clv'] > 0).mean()
                st.metric("Positive CLV %", f"{positive_clv_pct:.1%}")
            
            with col3:
                win_rate = clv_df['won'].mean()
                st.metric("Win Rate", f"{win_rate:.1%}",
                         delta="Above Break-even" if win_rate > 0.524 else "Below",
                         delta_color="normal" if win_rate > 0.524 else "inverse")
            
            with col4:
                high_clv_games = (clv_df['clv'] > 0.02).sum()
                st.metric("High CLV Games", f"{high_clv_games}",
                         help="Games with CLV > 2%")
            
            # CLV distribution
            st.markdown("**CLV Distribution**")
            st.bar_chart(clv_df['clv'].value_counts(bins=20).sort_index())
            
        except Exception as e:
            st.error(f"Error loading CLV data: {e}")
    else:
        st.info(f"No CLV data for {season}. Run `clv_tracker.py --season {season} --generate-report`")
    
    # === SECTION 4: Paper Trading Results ===
    st.subheader("📈 Paper Trading Results")
    
    paper_trading_dir = validation_dir / "paper_trading"
    comparison_file = paper_trading_dir / f"{season}_comparison.json"
    
    if comparison_file.exists():
        try:
            with open(comparison_file) as f:
                pt_results = json.load(f)
            
            # Display strategy comparison
            st.markdown("**Strategy Performance Comparison**")
            
            strategies_data = []
            for strategy_name, metrics in pt_results.items():
                if isinstance(metrics, dict) and 'roi' in metrics:
                    strategies_data.append({
                        'Strategy': strategy_name.title(),
                        'ROI': f"{metrics['roi']:.2%}",
                        'Win Rate': f"{metrics['win_rate']:.1%}",
                        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                        'Max Drawdown': f"{metrics['max_drawdown']:.1%}",
                        'Total Bets': metrics['n_bets'],
                        'Final Bankroll': f"${metrics['final_bankroll']:.2f}"
                    })
            
            if strategies_data:
                st.dataframe(pd.DataFrame(strategies_data), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading paper trading results: {e}")
    else:
        st.info(f"No paper trading results for {season}. Run `paper_trading.py --season {season} --compare-strategies`")
    
    # === SECTION 5: Benchmark Comparison ===
    st.subheader("🏆 Benchmark Comparison")
    
    benchmark_dir = validation_dir / "benchmark_comparison"
    benchmark_file = benchmark_dir / f"{season}_comparison.json"
    
    if benchmark_file.exists():
        try:
            with open(benchmark_file) as f:
                bench_results = json.load(f)
            
            # Model vs benchmarks table
            st.markdown("**Model vs Public Benchmarks**")
            
            model_metrics = bench_results.get('model', {})
            benchmarks = bench_results.get('benchmarks', {})
            
            comparison_data = [{
                'Model': 'Our Model',
                'Accuracy': f"{model_metrics.get('accuracy', 0):.3f}",
                'AUC': f"{model_metrics.get('auc', 0):.3f}",
                'Brier Score': f"{model_metrics.get('brier_score', 0):.4f}",
            }]
            
            for bench_name, bench_metrics in benchmarks.items():
                comparison_data.append({
                    'Model': bench_metrics.get('name', bench_name),
                    'Accuracy': f"{bench_metrics.get('accuracy', 0):.3f}",
                    'AUC': f"{bench_metrics.get('auc', 0):.3f}",
                    'Brier Score': f"{bench_metrics.get('brier_score', 0):.4f}",
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Statistical significance
            comparisons = bench_results.get('comparisons', [])
            if comparisons:
                st.markdown("**Statistical Significance Tests**")
                sig_data = []
                for comp in comparisons:
                    sig_data.append({
                        'Comparison': f"{comp['model_a']} vs {comp['model_b']}",
                        'Metric': comp['metric'],
                        'Difference': f"{comp['difference']:+.4f}",
                        'P-value': f"{comp['p_value']:.4f}",
                        'Significant': "✅" if comp['significant'] else "❌",
                        'Test': comp['test_name']
                    })
                st.dataframe(pd.DataFrame(sig_data), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading benchmark comparison: {e}")
    else:
        st.info(f"No benchmark comparison for {season}. Run `benchmark_comparison.py --season {season} --generate-report`")
    
    # === SECTION 6: Walk-Forward Validation ===
    st.subheader("🔄 Walk-Forward Validation")
    
    walk_forward_dir = validation_dir / "walk_forward_validation"
    summary_file = walk_forward_dir / "walk_forward_summary.json"
    
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                wf_results = json.load(f)
            
            # Display aggregated metrics
            agg_metrics = wf_results.get('aggregated_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mean_acc = agg_metrics.get('winprob', {}).get('mean_accuracy', 0)
                std_acc = agg_metrics.get('winprob', {}).get('std_accuracy', 0)
                st.metric("Mean Accuracy", f"{mean_acc:.3f} ± {std_acc:.3f}")
            
            with col2:
                mean_auc = agg_metrics.get('winprob', {}).get('mean_auc', 0)
                std_auc = agg_metrics.get('winprob', {}).get('std_auc', 0)
                st.metric("Mean AUC", f"{mean_auc:.3f} ± {std_auc:.3f}")
            
            with col3:
                mean_brier = agg_metrics.get('winprob', {}).get('mean_brier', 0)
                std_brier = agg_metrics.get('winprob', {}).get('std_brier', 0)
                st.metric("Mean Brier", f"{mean_brier:.4f} ± {std_brier:.4f}")
            
            with col4:
                n_windows = len(wf_results.get('windows', []))
                st.metric("Test Windows", f"{n_windows}")
            
            # Per-window results
            windows = wf_results.get('windows', [])
            if windows:
                st.markdown("**Per-Window Results**")
                window_data = []
                for window in windows:
                    window_data.append({
                        'Test Season': window['test_season'],
                        'Train Period': f"{window['train_start']}-{window['train_end']}",
                        'Accuracy': f"{window.get('accuracy', 0):.3f}",
                        'AUC': f"{window.get('auc', 0):.3f}",
                        'Brier': f"{window.get('brier', 0):.4f}",
                    })
                st.dataframe(pd.DataFrame(window_data), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading walk-forward validation: {e}")
    else:
        st.info("No walk-forward validation results. Run `walk_forward_validation.py`")
    
    # === SECTION 7: Quick Actions ===
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh All Metrics"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Generate Reports"):
            st.info("Run validation scripts to generate reports")
    
    with col3:
        if st.button("📥 Export Data"):
            st.info("Export functionality coming soon")


tab_labels = ["Pipeline Ops", "Transparency", "Upcoming Predictions", "Performance Retro", "Live Validation"]
tab_pipeline, tab_transparency, tab_predictions, tab_history, tab_validation = st.tabs(tab_labels)

with tab_pipeline:
    render_pipeline_tab()

with tab_transparency:
    render_transparency_tab()

with tab_predictions:
    render_predictions_tab()

with tab_history:
    render_history_tab()

with tab_validation:
    render_validation_tab()
