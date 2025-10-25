import math
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

COLUMN_DOCS = {
    "season": "Season year",
    "week": "NFL week number",
    "matchup": "Formatted as 'Away @ Home'",
    "home_team": "Home team abbreviation",
    "away_team": "Away team abbreviation",
    "home_win_prob": "Model-predicted probability the home team wins",
    "pred_home_margin": "Predicted home margin (positive favors the home team)",
    "pred_home_margin_lo": "Lower bound of predicted margin interval",
    "pred_home_margin_hi": "Upper bound of predicted margin interval",
    "news_count7_home": "News/injury keyword hits impacting the home team in the last 7 days",
    "news_count7_away": "News/injury keyword hits impacting the away team in the last 7 days",
    "inj_out_home": "Home players currently ruled out",
    "inj_out_away": "Away players currently ruled out",
    "weather_temp_kickoff_f": "Kickoff temperature (°F)",
    "weather_wind_speed_mph": "Kickoff wind speed (mph)",
    "offense_team": "Team whose offense is being evaluated",
    "defense_team": "Opponent defense facing that offense",
    "mismatch_score": "Composite offense-versus-defense mismatch score (higher favors the offense)",
    "offense_signal": "Aggregated offensive signal derived from keyword metrics",
    "defense_vulnerability": "Aggregated defensive weakness signal",
    "win_prob": "Win probability for the highlighted offense's team",
    "predicted_margin": "Predicted margin for the highlighted offense's team",
    "news_7d": "News/injury hits (7-day window) affecting the highlighted offense's team",
    "injury_count": "Count of out/doubtful/questionable players for the highlighted offense's team",
    "run_mismatch": "Composite rushing mismatch score",
    "rush_signal": "Aggregated rushing offense signal",
    "defensive_rush_vulnerability": "Rush defense vulnerability signal",
    "top_rush_metric": "Representative rushing metric (yards, attempts, etc.)",
    "rush_allowed_metric": "Representative rushing allowed metric for the defense",
    "receiving_mismatch": "Composite receiving mismatch score",
    "receiving_signal": "Aggregated passing/receiving offense signal",
    "def_pass_vulnerability": "Pass defense vulnerability signal",
    "top_receiving_metric": "Representative receiving metric (yards, targets, etc.)",
    "coverage_allowed_metric": "Representative coverage yards allowed metric for the defense",
    "favorite_team": "Model favourite in the matchup",
    "opponent": "Underdog opponent",
    "favorite_win_prob": "Favourite's win probability",
    "favorite_margin": "Favourite's predicted margin",
    "injury_pressure": "Injury burden applied to the favourite",
    "news_pressure": "News/injury keyword hits affecting the favourite",
    "depth_pressure": "Depth chart/snap stress indicator for the favourite",
    "weather_penalty": "Weather-based risk penalty",
    "risk_score": "Composite upset-risk score (higher indicates more external pressure)",
    "home_news_hits": "News/injury hits impacting the home team",
    "away_news_hits": "News/injury hits impacting the away team",
    "home_injuries": "Injury burden on the home team",
    "away_injuries": "Injury burden on the away team",
    "combined_pressure": "Combined availability/news pressure on both teams",
    "weather_temp": "Kickoff temperature (°F)",
    "weather_wind": "Wind speed or descriptor at kickoff",
    "weather_notes": "Additional weather details (joined text)",
}


st.set_page_config(page_title="NFL Predictions", layout="wide")
st.title("NFL Predictions Intelligence Center")


@st.cache_data
def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("season", "week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("home_win_prob", "pred_home_margin"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _hist_counts(series: pd.Series, bins: int) -> pd.DataFrame:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame({"bin": [], "count": []})
    counts, edges = np.histogram(series.to_numpy(dtype=float), bins=bins)
    labels = [f"{edges[i]:.3f}-{edges[i+1]:.3f}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"bin": labels, "count": counts})


def _safe_float(val: float | str | pd.Series | None) -> float | None:
    try:
        num = float(val)  # type: ignore[arg-type]
        return num if math.isfinite(num) else None
    except Exception:
        return None


def _safe_int(val: float | str | pd.Series | None) -> int | None:
    num = _safe_float(val)
    if num is None:
        return None
    try:
        return int(num)
    except Exception:
        return None


OFFENSE_KEYWORDS = ["pass", "rush", "receiv", "yards", "scoring", "epa", "plays", "off_"]
OFFENSE_EXCLUDE = ["def_", "news_", "inj_", "depth_", "roster_", "snap_", "weather", "margin", "win_prob"]
DEFENSE_WEAK_KEYWORDS = ["def_", "allowed", "pass", "rush", "epa", "yards"]
DEFENSE_EXCLUDE = ["news_", "inj_", "depth_", "roster_", "snap_", "weather"]

RUN_OFF_KEYWORDS = ["rush", "rushing", "rush_yards", "rush_yp", "carry"]
RUN_DEF_KEYWORDS = ["def_", "rush", "ground", "yards_allowed", "tfl"]

REC_OFF_KEYWORDS = ["pass", "receiv", "target", "air", "catch", "yards"]
REC_DEF_KEYWORDS = ["def_", "pass", "receiv", "coverage", "yards_allowed"]

DEPTH_KEYWORDS = ["depth_", "roster_", "snaps_"]


def _side_keyword_score(row: pd.Series, side: str, include: Iterable[str], exclude: Iterable[str]) -> float:
    suffix = f"_{side}"
    total = 0.0
    hits = 0
    for col, val in row.items():
        if not isinstance(col, str) or not col.endswith(suffix):
            continue
        low = col.lower()
        if any(excl in low for excl in exclude):
            continue
        if any(inc in low for inc in include):
            num = _safe_float(val)
            if num is None:
                continue
            total += num
            hits += 1
    return total / hits if hits else math.nan


def _collect_feature(row: pd.Series, side: str, keywords: Iterable[str]) -> float | None:
    suffix = f"_{side}"
    best_val = None
    best_score = -math.inf
    for col, val in row.items():
        if not isinstance(col, str):
            continue
        low = col.lower()
        if suffix not in low:
            continue
        if not any(key in low for key in keywords):
            continue
        num = _safe_float(val)
        if num is None:
            continue
        score = abs(num)
        if score > best_score:
            best_score = score
            best_val = num
    return best_val


def _weather_penalty(row: pd.Series) -> float:
    penalty = 0.0
    for col, val in row.items():
        if not isinstance(col, str):
            continue
        low = col.lower()
        if not low.startswith("weather"):
            continue
        if "wind" in low:
            wind = _safe_float(val)
            if wind and wind > 15:
                penalty += 0.5
            if wind and wind > 25:
                penalty += 0.5
        if any(token in low for token in ("precip", "rain", "snow")):
            if isinstance(val, str):
                label = val.strip().lower()
                if label and label not in {"0", "none", "clear", "no"}:
                    penalty += 0.5
            else:
                amt = _safe_float(val)
                if amt and amt > 0:
                    penalty += 0.5
        if "temp" in low:
            temp = _safe_float(val)
            if temp is not None and (temp < 32 or temp > 90):
                penalty += 0.2
    return penalty


def _build_offense_mismatch_table(df: pd.DataFrame) -> pd.DataFrame:
    records: List[dict] = []
    for _, row in df.iterrows():
        season = _safe_int(row.get("season"))
        week = _safe_int(row.get("week"))
        home = row.get("home_team", "HOME")
        away = row.get("away_team", "AWAY")
        matchup = f"{away} @ {home}"
        for side, opp in (("home", "away"), ("away", "home")):
            offense_score = _side_keyword_score(row, side, OFFENSE_KEYWORDS, OFFENSE_EXCLUDE)
            defense_weak = _side_keyword_score(row, opp, DEFENSE_WEAK_KEYWORDS, DEFENSE_EXCLUDE)
            if math.isnan(offense_score) and math.isnan(defense_weak):
                continue
            off_val = 0.0 if math.isnan(offense_score) else offense_score
            def_val = 0.0 if math.isnan(defense_weak) else defense_weak
            mismatch = off_val + def_val
            if side == "home":
                offence_team, defence_team = home, away
                win_prob = _safe_float(row.get("home_win_prob"))
                margin = _safe_float(row.get("pred_home_margin"))
                news = _safe_float(row.get("news_count7_home"))
                injuries = sum(
                    max(_safe_float(row.get(col)) or 0.0, 0.0)
                    for col in ("inj_out_home", "inj_doubtful_home", "inj_questionable_home")
                )
            else:
                offence_team, defence_team = away, home
                base_prob = _safe_float(row.get("home_win_prob"))
                win_prob = 1 - base_prob if base_prob is not None else None
                base_margin = _safe_float(row.get("pred_home_margin"))
                margin = -base_margin if base_margin is not None else None
                news = _safe_float(row.get("news_count7_away"))
                injuries = sum(
                    max(_safe_float(row.get(col)) or 0.0, 0.0)
                    for col in ("inj_out_away", "inj_doubtful_away", "inj_questionable_away")
                )
            records.append(
                {
                    "season": season,
                    "week": week,
                    "matchup": matchup,
                    "offense_team": offence_team,
                    "defense_team": defence_team,
                    "mismatch_score": mismatch,
                    "offense_signal": off_val,
                    "defense_vulnerability": def_val,
                    "win_prob": win_prob,
                    "predicted_margin": margin,
                    "news_7d": news,
                    "injury_count": injuries,
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("mismatch_score", ascending=False)


def _build_run_mismatch_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df.iterrows():
        season = _safe_int(row.get("season"))
        week = _safe_int(row.get("week"))
        home = row.get("home_team", "HOME")
        away = row.get("away_team", "AWAY")
        matchup = f"{away} @ {home}"
        for side, opp in (("home", "away"), ("away", "home")):
            off_score = _side_keyword_score(row, side, RUN_OFF_KEYWORDS, OFFENSE_EXCLUDE)
            def_score = _side_keyword_score(row, opp, RUN_DEF_KEYWORDS, DEFENSE_EXCLUDE)
            if math.isnan(off_score) and math.isnan(def_score):
                continue
            off_val = 0.0 if math.isnan(off_score) else off_score
            def_val = 0.0 if math.isnan(def_score) else def_score
            mismatch = off_val + def_val
            top_rush = _collect_feature(row, side, ["rush", "ground", "carry", "rushing"])
            def_allowed = _collect_feature(row, opp, ["def_rush", "rush_allowed", "rushing"])
            offence_team, defence_team = (home, away) if side == "home" else (away, home)
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "matchup": matchup,
                    "offense_team": offence_team,
                    "defense_team": defence_team,
                    "run_mismatch": mismatch,
                    "rush_signal": off_val,
                    "defensive_rush_vulnerability": def_val,
                    "top_rush_metric": top_rush,
                    "rush_allowed_metric": def_allowed,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("run_mismatch", ascending=False)


def _build_receiving_mismatch_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df.iterrows():
        season = _safe_int(row.get("season"))
        week = _safe_int(row.get("week"))
        home = row.get("home_team", "HOME")
        away = row.get("away_team", "AWAY")
        matchup = f"{away} @ {home}"
        for side, opp in (("home", "away"), ("away", "home")):
            off_score = _side_keyword_score(row, side, REC_OFF_KEYWORDS, OFFENSE_EXCLUDE)
            def_score = _side_keyword_score(row, opp, REC_DEF_KEYWORDS, DEFENSE_EXCLUDE)
            if math.isnan(off_score) and math.isnan(def_score):
                continue
            off_val = 0.0 if math.isnan(off_score) else off_score
            def_val = 0.0 if math.isnan(def_score) else def_score
            mismatch = off_val + def_val
            top_recv = _collect_feature(row, side, ["receiv", "target", "pass", "air"])
            def_allowed = _collect_feature(row, opp, ["def_pass", "def_receiv", "coverage", "yards_allowed"])
            offence_team, defence_team = (home, away) if side == "home" else (away, home)
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "matchup": matchup,
                    "offense_team": offence_team,
                    "defense_team": defence_team,
                    "receiving_mismatch": mismatch,
                    "receiving_signal": off_val,
                    "def_pass_vulnerability": def_val,
                    "top_receiving_metric": top_recv,
                    "coverage_allowed_metric": def_allowed,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("receiving_mismatch", ascending=False)


def _build_availability_watch(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df.iterrows():
        season = _safe_int(row.get("season"))
        week = _safe_int(row.get("week"))
        home = row.get("home_team", "HOME")
        away = row.get("away_team", "AWAY")
        matchup = f"{away} @ {home}"
        home_news = sum(
            max(_safe_float(row.get(col)) or 0.0, 0.0)
            for col in ("news_count7_home", "news_injury_kw7_home")
        )
        away_news = sum(
            max(_safe_float(row.get(col)) or 0.0, 0.0)
            for col in ("news_count7_away", "news_injury_kw7_away")
        )
        home_inj = sum(
            max(_safe_float(row.get(col)) or 0.0, 0.0)
            for col in ("inj_out_home", "inj_doubtful_home", "inj_questionable_home")
        )
        away_inj = sum(
            max(_safe_float(row.get(col)) or 0.0, 0.0)
            for col in ("inj_out_away", "inj_doubtful_away", "inj_questionable_away")
        )
        rows.append(
            {
                "season": season,
                "week": week,
                "matchup": matchup,
                "home_team": home,
                "away_team": away,
                "home_news_hits": home_news,
                "away_news_hits": away_news,
                "home_injuries": home_inj,
                "away_injuries": away_inj,
                "combined_pressure": home_news + away_news + home_inj + away_inj,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("combined_pressure", ascending=False)


def _build_upset_watch(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df.iterrows():
        season = _safe_int(row.get("season"))
        week = _safe_int(row.get("week"))
        home = row.get("home_team", "HOME")
        away = row.get("away_team", "AWAY")
        matchup = f"{away} @ {home}"
        home_prob = _safe_float(row.get("home_win_prob"))
        margin = _safe_float(row.get("pred_home_margin"))
        if home_prob is None and margin is None:
            continue
        if (home_prob is not None and home_prob >= 0.55) or (margin is not None and margin >= 2):
            fav_side = "home"
            fav_team = home
            dog_team = away
            fav_prob = home_prob
            fav_margin = margin
        elif (home_prob is not None and home_prob <= 0.45) or (margin is not None and margin <= -2):
            fav_side = "away"
            fav_team = away
            dog_team = home
            fav_prob = (1 - home_prob) if home_prob is not None else None
            fav_margin = (-margin) if margin is not None else None
        else:
            continue
        injuries = sum(
            max(_safe_float(row.get(f"{col}_{fav_side}")) or 0.0, 0.0)
            for col in ("inj_out", "inj_doubtful", "inj_questionable")
        )
        news = sum(
            max(_safe_float(row.get(f"{col}_{fav_side}")) or 0.0, 0.0)
            for col in ("news_count7", "news_injury_kw7")
        )
        depth_pressure = _side_keyword_score(row, fav_side, DEPTH_KEYWORDS, [])
        if math.isnan(depth_pressure):
            depth_pressure = 0.0
        weather_penalty = _weather_penalty(row)
        margin_lo = _safe_float(row.get("pred_home_margin_lo"))
        margin_hi = _safe_float(row.get("pred_home_margin_hi"))
        if fav_side == "away" and margin_lo is not None and margin_hi is not None:
            margin_lo, margin_hi = -margin_hi, -margin_lo
        interval_penalty = (
            1.0 if margin_lo is not None and margin_hi is not None and margin_lo <= 0 <= margin_hi else 0.0
        )
        risk_score = injuries * 0.5 + news * 0.3 + max(depth_pressure, 0.0) * 0.2 + weather_penalty + interval_penalty
        if risk_score <= 0:
            continue
        rows.append(
            {
                "season": season,
                "week": week,
                "matchup": matchup,
                "favorite_team": fav_team,
                "opponent": dog_team,
                "favorite_win_prob": fav_prob,
                "favorite_margin": fav_margin,
                "injury_pressure": injuries,
                "news_pressure": news,
                "depth_pressure": depth_pressure,
                "weather_penalty": weather_penalty,
                "risk_score": risk_score,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)


def _build_weather_watch(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df.iterrows():
        penalty = _weather_penalty(row)
        if penalty <= 0:
            continue
        season = _safe_int(row.get("season"))
        week = _safe_int(row.get("week"))
        home = row.get("home_team", "HOME")
        away = row.get("away_team", "AWAY")
        matchup = f"{away} @ {home}"
        rows.append(
            {
                "season": season,
                "week": week,
                "matchup": matchup,
                "weather_penalty": penalty,
                "weather_temp": _safe_float(row.get("weather_temp_kickoff_f")),
                "weather_wind": _safe_float(
                    next((row.get(col) for col in row.index if isinstance(col, str) and "weather" in col.lower() and "wind" in col.lower()), None)
                ),
                "weather_notes": " | ".join(
                    str(row.get(col))
                    for col in row.index
                    if isinstance(col, str) and col.startswith("weather") and pd.notna(row.get(col))
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("weather_penalty", ascending=False)


def _build_column_config(df: pd.DataFrame) -> dict:
    config = {}
    for col in df.columns:
        help_text = COLUMN_DOCS.get(col)
        if not help_text:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            config[col] = st.column_config.NumberColumn(col, help=help_text)
        else:
            config[col] = st.column_config.Column(col, help=help_text)
    return config


st.sidebar.header("Predictions Source")
source_col1, source_col2 = st.sidebar.columns([3, 1])
with source_col1:
    preds_path = st.text_input("CSV path", value="predictions.csv")
with source_col2:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

preds: pd.DataFrame | None = None
if uploaded is not None:
    preds = pd.read_csv(uploaded)
elif Path(preds_path).exists():
    preds = load_preds(preds_path)

if preds is None or preds.empty:
    st.info("Run the pipeline to create `predictions.csv`, or upload a file in the sidebar.")
    st.stop()


st.sidebar.header("Model Artefacts")
model_status = {}
for name in ("winprob_gb.pkl", "spread_gb.pkl"):
    path = Path("models") / name
    ok = path.exists()
    model_status[name] = ok
    st.sidebar.write(f"{name}: {'✅ ready' if ok else '⚠️ missing'}")

if model_status.get("winprob_gb.pkl"):
    try:
        artefact = joblib.load(Path("models/winprob_gb.pkl"))
        calibrated = artefact.get("calibrated", False) if isinstance(artefact, dict) else False
        st.sidebar.caption(f"Win probability calibrated: {'Yes' if calibrated else 'No'}")
    except Exception:
        st.sidebar.caption("Win probability calibrated: Unknown")

if model_status.get("spread_gb.pkl"):
    try:
        artefact = joblib.load(Path("models/spread_gb.pkl"))
        quantiles = artefact.get("quantile_models", {}) if isinstance(artefact, dict) else {}
        st.sidebar.caption(f"Spread quantiles: {'Yes' if isinstance(quantiles, dict) and quantiles else 'No'}")
    except Exception:
        st.sidebar.caption("Spread quantiles: Unknown")

analysis_top_n = st.sidebar.slider("Rows to display in insights", min_value=5, max_value=30, value=10, step=5)


st.subheader("Filters")
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    seasons = sorted([int(val) for val in preds.get("season", pd.Series(dtype="Int64")).dropna().unique().tolist()])
    season_sel = st.selectbox("Season", seasons, index=len(seasons) - 1 if seasons else 0, disabled=not seasons)
with filter_col2:
    if {"season", "week"}.issubset(preds.columns):
        week_options = (
            preds.loc[preds["season"] == season_sel, "week"].dropna().astype(int).unique().tolist()
        )
        week_options.sort()
    else:
        week_options = []
    week_sel = st.selectbox("Week", week_options, index=len(week_options) - 1 if week_options else 0, disabled=not week_options)

filtered = preds.copy()
if "season" in filtered.columns:
    filtered = filtered[filtered["season"].astype("Int64") == season_sel]
if "week" in filtered.columns:
    filtered = filtered[filtered["week"].astype("Int64") == week_sel]

team_options = sorted(
    set(filtered.get("home_team", pd.Series(dtype=str)).dropna())
    | set(filtered.get("away_team", pd.Series(dtype=str)).dropna())
)
team_filter = st.multiselect("Optional team filter", team_options, default=[])
if team_filter:
    mask = filtered["home_team"].isin(team_filter) | filtered["away_team"].isin(team_filter)
    filtered = filtered[mask]

sort_opt = st.selectbox(
    "Sort matchups by",
    ["home_win_prob desc", "pred_home_margin desc", "home_team", "away_team"],
)
if sort_opt == "home_win_prob desc" and "home_win_prob" in filtered.columns:
    filtered = filtered.sort_values("home_win_prob", ascending=False)
elif sort_opt == "pred_home_margin desc" and "pred_home_margin" in filtered.columns:
    filtered = filtered.sort_values("pred_home_margin", ascending=False)
elif sort_opt == "home_team" and {"home_team", "away_team"}.issubset(filtered.columns):
    filtered = filtered.sort_values(["home_team", "away_team"])
elif sort_opt == "away_team" and {"home_team", "away_team"}.issubset(filtered.columns):
    filtered = filtered.sort_values(["away_team", "home_team"])


st.subheader("Summary")
summary_cols = st.columns(5)
summary_cols[0].metric("Games", len(filtered))
if "home_win_prob" in filtered.columns and not filtered.empty:
    avg_prob = float(np.nanmean(filtered["home_win_prob"]))
    summary_cols[1].metric("Avg home win prob", f"{avg_prob:.1%}" if math.isfinite(avg_prob) else "-")
if "pred_home_margin" in filtered.columns and not filtered.empty:
    avg_margin = float(np.nanmean(filtered["pred_home_margin"]))
    summary_cols[2].metric("Avg home margin", f"{avg_margin:+.2f}" if math.isfinite(avg_margin) else "-")
if {"pred_home_margin_lo", "pred_home_margin_hi"}.issubset(filtered.columns):
    band = filtered["pred_home_margin_hi"] - filtered["pred_home_margin_lo"]
    avg_band = float(np.nanmean(pd.to_numeric(band, errors="coerce")))
    summary_cols[3].metric("Avg margin band", f"{avg_band:.2f} pts" if math.isfinite(avg_band) else "-")
news_cols = [col for col in filtered.columns if col.startswith("news_count7")]
if news_cols:
    total_news = float(filtered[news_cols].fillna(0).sum().sum())
    summary_cols[4].metric("News hits (7d)", f"{int(total_news)}")


st.subheader("Distributions")
dist_cols = st.columns(3)
with dist_cols[0]:
    if "home_win_prob" in filtered.columns and filtered["home_win_prob"].dropna().any():
        st.bar_chart(_hist_counts(filtered["home_win_prob"], bins=10).set_index("bin"))
    else:
        st.caption("No win probability values.")
with dist_cols[1]:
    if "pred_home_margin" in filtered.columns and filtered["pred_home_margin"].dropna().any():
        st.bar_chart(_hist_counts(filtered["pred_home_margin"], bins=12).set_index("bin"))
    else:
        st.caption("No margin values.")
with dist_cols[2]:
    if news_cols and filtered[news_cols].any(axis=None):
        st.bar_chart(_hist_counts(filtered[news_cols].sum(axis=1), bins=10).set_index("bin"))
    else:
        st.caption("No news volume detected.")


scoreboard_columns = [
    "season",
    "week",
    "home_team",
    "away_team",
    "home_win_prob",
    "pred_home_margin",
    "pred_home_margin_lo",
    "pred_home_margin_hi",
    "news_count7_home",
    "news_count7_away",
    "inj_out_home",
    "inj_out_away",
    "weather_temp_kickoff_f",
    "weather_wind_speed_mph",
]
overview_df = filtered[[col for col in scoreboard_columns if col in filtered.columns]].copy()

offense_table = _build_offense_mismatch_table(filtered)
run_table = _build_run_mismatch_table(filtered)
receiving_table = _build_receiving_mismatch_table(filtered)
availability_table = _build_availability_watch(filtered)
upset_table = _build_upset_watch(filtered)
weather_table = _build_weather_watch(filtered)

tabs = st.tabs(
    [
        "Overview",
        "Offense vs Defense",
        "Run Game Spotlight",
        "Receiving Spotlight",
        "Upset Watch",
        "Availability & Weather",
    ]
)

with tabs[0]:
    st.markdown("### Matchup Overview")
    if overview_df.empty:
        st.info("No standard columns available for overview.")
    else:
        st.dataframe(
            overview_df,
            use_container_width=True,
            column_config=_build_column_config(overview_df),
        )

with tabs[1]:
    st.markdown("### Offensive mismatches vs vulnerable defenses")
    if offense_table.empty:
        st.info("Offensive mismatch signals are unavailable for this week.")
    else:
        display_cols = [
            "season",
            "week",
            "matchup",
            "offense_team",
            "defense_team",
            "mismatch_score",
            "offense_signal",
            "defense_vulnerability",
            "win_prob",
            "predicted_margin",
            "news_7d",
            "injury_count",
        ]
        subset = offense_table.head(analysis_top_n)[display_cols]
        st.dataframe(
            subset,
            use_container_width=True,
            column_config=_build_column_config(subset),
        )

with tabs[2]:
    st.markdown("### Run game mismatches & league-leading rushers")
    if run_table.empty:
        st.info("Run game signals are unavailable for this week.")
    else:
        display_cols = [
            "season",
            "week",
            "matchup",
            "offense_team",
            "defense_team",
            "run_mismatch",
            "rush_signal",
            "defensive_rush_vulnerability",
            "top_rush_metric",
            "rush_allowed_metric",
        ]
        subset = run_table.head(analysis_top_n)[display_cols]
        st.dataframe(
            subset,
            use_container_width=True,
            column_config=_build_column_config(subset),
        )

with tabs[3]:
    st.markdown("### Receiving mismatches vs thin defensive depth")
    if receiving_table.empty:
        st.info("Receiving mismatch signals are unavailable for this week.")
    else:
        display_cols = [
            "season",
            "week",
            "matchup",
            "offense_team",
            "defense_team",
            "receiving_mismatch",
            "receiving_signal",
            "def_pass_vulnerability",
            "top_receiving_metric",
            "coverage_allowed_metric",
        ]
        subset = receiving_table.head(analysis_top_n)[display_cols]
        st.dataframe(
            subset,
            use_container_width=True,
            column_config=_build_column_config(subset),
        )

with tabs[4]:
    st.markdown("### Upset watch (external pressure on favourites)")
    if upset_table.empty:
        st.info("No high-risk favourites detected.")
    else:
        display_cols = [
            "season",
            "week",
            "matchup",
            "favorite_team",
            "opponent",
            "favorite_win_prob",
            "favorite_margin",
            "injury_pressure",
            "news_pressure",
            "depth_pressure",
            "weather_penalty",
            "risk_score",
        ]
        subset = upset_table.head(analysis_top_n)[display_cols]
        st.dataframe(
            subset,
            use_container_width=True,
            column_config=_build_column_config(subset),
        )

with tabs[5]:
    st.markdown("### Availability radar")
    sub_cols = [
        "season",
        "week",
        "matchup",
        "home_team",
        "away_team",
        "home_news_hits",
        "home_injuries",
        "away_news_hits",
        "away_injuries",
        "combined_pressure",
    ]
    if availability_table.empty:
        st.info("No availability signals detected.")
    else:
        subset = availability_table.head(analysis_top_n)[sub_cols]
        st.dataframe(
            subset,
            use_container_width=True,
            column_config=_build_column_config(subset),
        )
    st.markdown("### Weather watch")
    if weather_table.empty:
        st.caption("No weather risks flagged.")
    else:
        display_cols = [
            "season",
            "week",
            "matchup",
            "weather_penalty",
            "weather_temp",
            "weather_wind",
            "weather_notes",
        ]
        subset = weather_table.head(analysis_top_n)[display_cols]
        st.dataframe(
            subset,
            use_container_width=True,
            column_config=_column_config(subset.columns),
        )


st.subheader("Download")
st.download_button(
    "Download filtered predictions",
    data=filtered.to_csv(index=False),
    file_name="predictions_filtered.csv",
    mime="text/csv",
)
