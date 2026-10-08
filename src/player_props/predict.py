from __future__ import annotations

import argparse
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src.player_props.features import (
    COUNT_MARKETS,
    NOVELTY_MARKETS,
    build_player_prop_novelty_prediction_features,
    build_player_prop_prediction_features,
)
from src.player_props.labels import (
    DEFAULT_NOVELTY_TD_OUTPUT_PATH as DEFAULT_NOVELTY_LABELS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_LABELS_PATH,
)
from src.player_props.over_probability import (
    DEFAULT_INPUT_PATH as DEFAULT_OVER_PROBABILITY_HISTORY_PATH,
    attach_current_over_probabilities,
)
from src.player_props.train import DEFAULT_MODELS_DIR
from src.player_availability import (
    DEFAULT_OUTPUT_PATH as DEFAULT_AVAILABILITY_PATH,
    attach_player_availability,
)
from src.predict.utils import moneyline_to_prob
from src.utils.io import PROC_DIR, RAW_DIR, read_df
from src.utils.teams import get_team_abbr_from_name, normalize_team_abbr


DEFAULT_PREDICTION_DIR = Path("predictions")
DEFAULT_QB_OUTPUT = DEFAULT_PREDICTION_DIR / "player_props_qb.csv"
DEFAULT_OFFENSE_OUTPUT = DEFAULT_PREDICTION_DIR / "player_props_offense.csv"
DEFAULT_DEFENSE_OUTPUT = DEFAULT_PREDICTION_DIR / "player_props_defense.csv"
DEFAULT_NOVELTY_OUTPUT = DEFAULT_PREDICTION_DIR / "player_props_novelty.csv"
DEFAULT_ROSTERS_PATH = RAW_DIR / "nfl_rosters.parquet"
DEFAULT_INJURIES_PATH = RAW_DIR / "nfl_injuries.parquet"
DEFAULT_BDL_INJURIES_PATH = RAW_DIR / "balldontlie_injuries.parquet"
DEFAULT_PLAYER_IDENTITY_PATH = PROC_DIR / "player_identity.parquet"
DEFAULT_EDGE_POLICY_PATH = Path("predictions/evaluation/player_props/edge_recommendation_policy.csv")
DEFAULT_ODDS_VENDORS = ("draftkings", "hardrock", "fanduel")
DEFAULT_ODDS_VENDOR = DEFAULT_ODDS_VENDORS[0]
LINE_BALANCE_DISTANCE_LIMIT = 0.15
YARD_LINE_VARIANCE_THRESHOLD = 5.0
COUNT_LINE_VARIANCE_THRESHOLD = 0.5
_SKLEARN_VERSION_WARNING_EMITTED = False

MODEL_MARKET_TO_PROPLINE_MARKET = {
    "qb_passing_yards": "player_pass_yds",
    "qb_passing_tds": "player_pass_tds",
    "rb_rushing_yards": "player_rush_yds",
    "rb_rushing_tds": "player_rush_tds",
    "wrte_receiving_yards": "player_reception_yds",
    "wrte_receiving_tds": "player_reception_tds",
    "anytime_td": "player_anytime_td",
    "first_td": "player_1st_td",
    "last_td": "player_last_td",
    "two_plus_tds": "player_2plus_td",
    "three_plus_tds": "player_3plus_td",
    "def_sacks": "player_sacks",
}

PROPLINE_MARKET_TO_MODEL_MARKET = {value: key for key, value in MODEL_MARKET_TO_PROPLINE_MARKET.items()}
ONE_SIDED_TOUCHDOWN_MARKET_LINES = {
    "player_anytime_td": 0.5,
    "player_1st_td": 0.5,
    "player_last_td": 0.5,
    "player_2plus_td": 0.5,
    "player_3plus_td": 0.5,
}
ANYTIME_TD_FALLBACK_LINE_TYPE = "anytime_td_fallback"
ANYTIME_TD_FALLBACK_TARGET_MARKETS = frozenset({"rb_rushing_tds", "wrte_receiving_tds"})
NON_BETTABLE_LINE_TYPES = frozenset({ANYTIME_TD_FALLBACK_LINE_TYPE})

MARKET_TO_BDL_PROP_TYPE = {
    "qb_passing_yards": "passing_yards",
    "rb_rushing_yards": "rushing_yards",
    "wrte_receiving_yards": "receiving_yards",
    "anytime_td": "anytime_td",
}

QB_MARKETS = frozenset({"qb_passing_yards", "qb_passing_tds"})
OFFENSE_OUTPUT_MARKETS = frozenset(
    {"rb_rushing_yards", "rb_rushing_tds", "wrte_receiving_yards", "wrte_receiving_tds", "anytime_td"}
)
NOVELTY_OUTPUT_MARKETS = NOVELTY_MARKETS

BDL_PROP_TYPE_TO_MARKET = {value: key for key, value in MARKET_TO_BDL_PROP_TYPE.items()}

OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "team_side",
    "player_id",
    "player_name",
    "position",
    "position_group",
    "market",
    "projection",
    "prob_over",
    "confidence_tier",
    "model_name",
    "baseline_projection",
    "model_projection",
    "sample_size",
    "feature_count",
    "line",
    "line_type",
    "implied_probability",
    "prob_over_raw",
    "prob_edge",
    "prob_over_method",
    "prob_over_sample_size",
    "over_break_even_probability",
    "under_break_even_probability",
    "over_ev",
    "under_ev",
    "edge_side",
    "edge_odds",
    "edge_break_even_probability",
    "edge_ev",
    "edge_probability_edge",
    "edge_recommendation_flag",
    "edge_recommendation_tier",
    "edge_policy_scope",
    "edge_policy_strategy",
    "edge_policy_threshold",
    "edge_policy_roi",
    "edge_policy_n_bets",
    "edge",
    "sportsbook",
    "over_odds",
    "under_odds",
    "line_updated_at",
    "line_consensus",
    "line_min",
    "line_max",
    "line_range",
    "line_std",
    "line_book_count",
    "line_options_count",
    "line_variance_flag",
    "injury_status",
    "injury_practice_status",
    "injury_body_part",
    "injury_report_week",
    "injury_source",
    "injury_last_updated",
    "injury_reported",
    "injury_exclusion_flag",
    "prop_quality_flag",
    "current_team",
    "active_current_roster",
    "roster_validation_flag",
    "roster_validation_source",
    "availability_status",
    "depth_chart_position",
    "depth_chart_rank",
    "bettable_flag",
    "games_sampled",
    "player_rank",
    "generated_at",
]

LINE_STAT_COLUMNS = [
    "line_consensus",
    "line_min",
    "line_max",
    "line_range",
    "line_std",
    "line_book_count",
    "line_options_count",
    "line_variance_flag",
]

AVAILABILITY_COLUMNS = [
    "current_team",
    "active_current_roster",
    "roster_validation_flag",
    "roster_validation_source",
    "availability_status",
    "depth_chart_position",
    "depth_chart_rank",
]

INJURY_COLUMNS = [
    "injury_status",
    "injury_practice_status",
    "injury_body_part",
    "injury_report_week",
    "injury_source",
    "injury_last_updated",
    "injury_reported",
    "injury_exclusion_flag",
    "prop_quality_flag",
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


def _team_abbr_from_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NAN", "NA", "<NA>"}:
        return None
    abbr = get_team_abbr_from_name(text) or _clean_team(text)
    return "LA" if abbr == "LAR" else abbr


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _game_identity_parts(game_id: object) -> tuple[int | None, int | None, list[str]]:
    text = str(game_id).strip().upper()
    parts = text.split("_")
    if len(parts) < 4:
        return None, None, []
    try:
        season = int(parts[0])
        week = int(parts[1])
    except ValueError:
        return None, None, []
    teams: list[str] = []
    for raw in parts[-2:]:
        team = _clean_team(raw)
        if team and team not in teams:
            teams.append(team)
    return season, week, teams


def _prepare_player_identity(identity: pd.DataFrame | None) -> pd.DataFrame:
    if identity is None or identity.empty:
        return pd.DataFrame()
    required = {"season", "week", "team", "player_id"}
    if missing := required - set(identity.columns):
        raise ValueError(f"Player identity table missing required columns: {sorted(missing)}")

    out = identity.copy()
    out["identity_season"] = pd.to_numeric(out["season"], errors="coerce")
    out["identity_week"] = pd.to_numeric(out["week"], errors="coerce")
    out["identity_team"] = out["team"].apply(_clean_team)
    out["identity_player_id"] = _as_text(out["player_id"])
    if "alias_name_key" not in out.columns:
        out["alias_name_key"] = out.get("alias_name", pd.Series(pd.NA, index=out.index)).apply(_player_name_key)
    if "alias_prefix_key" not in out.columns:
        out["alias_prefix_key"] = out.get("alias_name", pd.Series(pd.NA, index=out.index)).apply(_player_name_prefix_key)
    out["alias_name_key"] = _as_text(out["alias_name_key"])
    out["alias_prefix_key"] = _as_text(out["alias_prefix_key"])
    if "alias_confidence" not in out.columns:
        out["alias_confidence"] = 0.0
    out["alias_confidence"] = pd.to_numeric(out["alias_confidence"], errors="coerce").fillna(0.0)
    out = out.dropna(
        subset=[
            "identity_season",
            "identity_week",
            "identity_team",
            "identity_player_id",
            "alias_name_key",
            "alias_prefix_key",
        ]
    ).copy()
    if out.empty:
        return pd.DataFrame()
    out["identity_season"] = out["identity_season"].astype(int)
    out["identity_week"] = out["identity_week"].astype(int)
    identity_keys = ["identity_season", "identity_week", "identity_team", "alias_name_key", "alias_prefix_key"]
    counts = (
        out.groupby(identity_keys, dropna=False)["identity_player_id"]
        .nunique(dropna=True)
        .reset_index(name="_identity_player_count")
    )
    out = out.merge(counts, on=identity_keys, how="left")
    out = out[out["_identity_player_count"].eq(1)].copy()
    out = out.sort_values(identity_keys + ["alias_confidence"], ascending=[True, True, True, True, True, False], kind="stable")
    out = out.drop_duplicates(identity_keys, keep="first")
    return out[identity_keys + ["identity_player_id"]].reset_index(drop=True)


def _normalize_vendors(vendor: str | Sequence[str] | None) -> list[str]:
    if vendor is None:
        return []
    if isinstance(vendor, str):
        raw_values: Sequence[str] = vendor.split(",")
    else:
        raw_values = vendor
    return [str(value).strip().lower() for value in raw_values if str(value).strip()]


def _clean_injury_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NAN", "NA", "<NA>", "NULL"}:
        return None
    return text


def _injury_severity(value: object) -> int:
    text = str(value or "").strip().lower()
    if any(token in text for token in ["injured reserve", "reserve", "ir", "pup", "nfi", "out"]):
        return 4
    if "doubtful" in text:
        return 3
    if "questionable" in text:
        return 2
    if text:
        return 1
    return 0


def _injury_exclusion_flag(status: object) -> bool:
    text = str(status or "").strip().lower()
    exclusion_tokens = [
        "out",
        "doubtful",
        "injured reserve",
        "reserve",
        "ir",
        "pup",
        "nfi",
        "suspended",
        "inactive",
    ]
    return any(token in text for token in exclusion_tokens)


def _injury_reported(row: pd.Series) -> bool:
    status = _clean_injury_text(row.get("injury_status"))
    body_part = _clean_injury_text(row.get("injury_body_part"))
    practice_status = (_clean_injury_text(row.get("injury_practice_status")) or "").strip().lower()
    full_practice = "full participation" in practice_status
    return bool(
        status
        or "limited" in practice_status
        or "did not participate" in practice_status
        or (body_part and not full_practice)
    )


def _injury_quality_flag(row: pd.Series) -> str:
    if bool(row.get("injury_exclusion_flag")):
        return "injury_exclusion"
    if bool(row.get("injury_reported")):
        return "injury_review"
    return "ok"


def _line_balance_distance(frame: pd.DataFrame) -> pd.Series:
    over_prob = moneyline_to_prob(frame["over_odds"])
    under_prob = moneyline_to_prob(frame["under_odds"])
    total_prob = over_prob + under_prob
    implied = np.where(total_prob > 0, over_prob / total_prob, np.nan)
    has_two_way = pd.to_numeric(frame["over_odds"], errors="coerce").notna() & pd.to_numeric(
        frame["under_odds"], errors="coerce"
    ).notna()
    return pd.Series(np.where(has_two_way, np.abs(implied - 0.5), np.inf), index=frame.index)


def _american_profit_per_unit(values: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    odds = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    profit = np.full_like(odds, np.nan, dtype=float)
    pos = odds > 0
    neg = odds < 0
    profit[pos] = odds[pos] / 100.0
    profit[neg] = 100.0 / np.abs(odds[neg])
    return profit


def _line_selection_price_rank(frame: pd.DataFrame) -> pd.Series:
    over_odds = pd.to_numeric(frame.get("over_odds", pd.Series(np.nan, index=frame.index)), errors="coerce")
    under_odds = pd.to_numeric(frame.get("under_odds", pd.Series(np.nan, index=frame.index)), errors="coerce")
    projection = pd.to_numeric(frame.get("projection", pd.Series(np.nan, index=frame.index)), errors="coerce")
    line = pd.to_numeric(frame.get("line", pd.Series(np.nan, index=frame.index)), errors="coerce")

    use_under = projection.lt(line) & under_odds.notna()
    preferred_odds = pd.Series(np.where(use_under, under_odds, over_odds), index=frame.index, dtype="float64")
    fallback_odds = pd.concat([over_odds, under_odds], axis=1).max(axis=1, skipna=True)
    preferred_odds = preferred_odds.where(preferred_odds.notna(), fallback_odds)
    return (-preferred_odds).fillna(np.inf)


def _sportsbook_selection_rank(values: pd.Series) -> pd.Series:
    priority = {book: rank for rank, book in enumerate(DEFAULT_ODDS_VENDORS)}
    return values.apply(_sportsbook_policy_key).map(priority).fillna(len(priority)).astype(float)


def _add_prop_betting_edges(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    out = predictions.copy()
    for col in [
        "over_break_even_probability",
        "under_break_even_probability",
        "over_ev",
        "under_ev",
        "edge_side",
        "edge_odds",
        "edge_break_even_probability",
        "edge_ev",
    ]:
        if col not in out.columns:
            out[col] = pd.NA if col == "edge_side" else np.nan

    if not {"prob_over", "over_odds", "under_odds"}.issubset(out.columns):
        return out

    prob_over = pd.to_numeric(out["prob_over"], errors="coerce")
    over_odds = pd.to_numeric(out["over_odds"], errors="coerce")
    under_odds = pd.to_numeric(out["under_odds"], errors="coerce")
    over_profit = _american_profit_per_unit(over_odds)
    under_profit = _american_profit_per_unit(under_odds)
    out["over_break_even_probability"] = moneyline_to_prob(over_odds)
    out["under_break_even_probability"] = moneyline_to_prob(under_odds)
    out["over_ev"] = (prob_over * over_profit) - (1.0 - prob_over)
    under_prob = 1.0 - prob_over
    out["under_ev"] = (under_prob * under_profit) - prob_over

    valid_over = prob_over.notna() & over_odds.notna()
    valid_under = prob_over.notna() & under_odds.notna()
    choose_under = valid_under & (~valid_over | (out["under_ev"] > out["over_ev"]))
    choose_over = valid_over & ~choose_under
    out.loc[choose_over, "edge_side"] = "over"
    out.loc[choose_under, "edge_side"] = "under"
    out.loc[choose_over, "edge_odds"] = over_odds.loc[choose_over]
    out.loc[choose_under, "edge_odds"] = under_odds.loc[choose_under]
    out.loc[choose_over, "edge_break_even_probability"] = out.loc[choose_over, "over_break_even_probability"]
    out.loc[choose_under, "edge_break_even_probability"] = out.loc[choose_under, "under_break_even_probability"]
    out.loc[choose_over, "edge_ev"] = out.loc[choose_over, "over_ev"]
    out.loc[choose_under, "edge_ev"] = out.loc[choose_under, "under_ev"]
    implied_over = pd.to_numeric(out.get("implied_probability", pd.Series(np.nan, index=out.index)), errors="coerce")
    over_probability_edge = prob_over - implied_over
    under_probability_edge = (1.0 - prob_over) - (1.0 - implied_over)
    out["edge_probability_edge"] = np.nan
    out.loc[choose_over, "edge_probability_edge"] = over_probability_edge.loc[choose_over]
    out.loc[choose_under, "edge_probability_edge"] = under_probability_edge.loc[choose_under]

    if {"prob_over_method", "prob_edge"}.issubset(out.columns):
        neutral_fallback = out["prob_over_method"].eq("market_implied_fallback") & pd.to_numeric(
            out["prob_edge"], errors="coerce"
        ).abs().le(1e-12)
        out.loc[neutral_fallback, "edge_side"] = pd.NA
        out.loc[neutral_fallback, "edge_odds"] = np.nan
        out.loc[neutral_fallback, "edge_break_even_probability"] = np.nan
        out.loc[neutral_fallback, "edge_ev"] = 0.0
        out.loc[neutral_fallback, "edge_probability_edge"] = np.nan
    return out


def _sportsbook_policy_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "nan", "na", "<na>"}:
        return None
    aliases = {
        "hardrockbet": "hardrock",
        "hard_rock": "hardrock",
        "hard rock": "hardrock",
    }
    return aliases.get(text, text)


EDGE_RECOMMENDATION_COLUMNS = [
    "edge_recommendation_flag",
    "edge_recommendation_tier",
    "edge_policy_scope",
    "edge_policy_strategy",
    "edge_policy_threshold",
    "edge_policy_roi",
    "edge_policy_n_bets",
]


def _default_edge_recommendation_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    defaults: dict[str, object] = {
        "edge_recommendation_flag": False,
        "edge_recommendation_tier": "research_only",
        "edge_policy_scope": pd.NA,
        "edge_policy_strategy": pd.NA,
        "edge_policy_threshold": np.nan,
        "edge_policy_roi": np.nan,
        "edge_policy_n_bets": np.nan,
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    return out


def _prepare_edge_policy(policy: pd.DataFrame | None) -> pd.DataFrame:
    if policy is None or policy.empty:
        return pd.DataFrame()
    required = {"scope", "market", "threshold", "roi", "n_bets"}
    if not required.issubset(policy.columns):
        return pd.DataFrame()
    out = policy.copy()
    out = out[out["scope"].isin(["market_sportsbook", "market"])].copy()
    out["market"] = out["market"].astype("string").str.strip()
    out["sportsbook"] = out.get("sportsbook", pd.Series(pd.NA, index=out.index)).apply(_sportsbook_policy_key)
    out["threshold"] = pd.to_numeric(out["threshold"], errors="coerce")
    out["roi"] = pd.to_numeric(out["roi"], errors="coerce")
    out["n_bets"] = pd.to_numeric(out["n_bets"], errors="coerce")
    if "strategy" not in out.columns:
        out["strategy"] = "prob_edge"
    if "recommendation_tier" not in out.columns:
        out["recommendation_tier"] = np.where(out["roi"].ge(0.05), "strong", "lean")
    out = out.dropna(subset=["market", "threshold", "roi", "n_bets"])
    out = out[out["strategy"].eq("prob_edge")]
    if out.empty:
        return pd.DataFrame()
    out["_scope_rank"] = np.where(out["scope"].eq("market_sportsbook"), 0, 1)
    out = out.sort_values(
        ["_scope_rank", "market", "sportsbook", "threshold", "roi", "n_bets"],
        ascending=[True, True, True, False, False, False],
        kind="stable",
    )
    out = out.drop_duplicates(["scope", "market", "sportsbook"], keep="first")
    return out[
        [
            "scope",
            "market",
            "sportsbook",
            "strategy",
            "threshold",
            "roi",
            "n_bets",
            "recommendation_tier",
        ]
    ].reset_index(drop=True)


def _attach_edge_recommendations(predictions: pd.DataFrame, policy: pd.DataFrame | None) -> pd.DataFrame:
    if predictions.empty:
        return _default_edge_recommendation_values(predictions)
    out = _default_edge_recommendation_values(predictions)
    prepared_policy = _prepare_edge_policy(policy)
    if prepared_policy.empty:
        return out

    out["_edge_market_key"] = out["market"].astype("string").str.strip()
    out["_edge_sportsbook_key"] = out.get("sportsbook", pd.Series(pd.NA, index=out.index)).apply(_sportsbook_policy_key)
    out["_edge_row_id"] = np.arange(len(out))
    policy_cols = {
        "scope": "edge_policy_scope",
        "strategy": "edge_policy_strategy",
        "threshold": "edge_policy_threshold",
        "roi": "edge_policy_roi",
        "n_bets": "edge_policy_n_bets",
        "recommendation_tier": "_edge_policy_tier",
    }

    attached_ids: set[int] = set()
    matches: list[pd.DataFrame] = []
    sportsbook_policy = prepared_policy[prepared_policy["scope"].eq("market_sportsbook")].dropna(subset=["sportsbook"])
    if not sportsbook_policy.empty:
        sportsbook_matches = out.merge(
            sportsbook_policy.rename(columns={"market": "_edge_market_key", "sportsbook": "_edge_sportsbook_key"}),
            on=["_edge_market_key", "_edge_sportsbook_key"],
            how="inner",
            suffixes=("", "_policy"),
        )
        if not sportsbook_matches.empty:
            matches.append(sportsbook_matches)
            attached_ids.update(sportsbook_matches["_edge_row_id"].astype(int).tolist())

    market_policy = prepared_policy[prepared_policy["scope"].eq("market")].copy()
    if not market_policy.empty:
        market_base = out[~out["_edge_row_id"].isin(attached_ids)].copy()
        market_matches = market_base.merge(
            market_policy.drop(columns=["sportsbook"], errors="ignore").rename(columns={"market": "_edge_market_key"}),
            on="_edge_market_key",
            how="inner",
            suffixes=("", "_policy"),
        )
        if not market_matches.empty:
            matches.append(market_matches)

    if matches:
        policy_matches = pd.concat(matches, ignore_index=True, sort=False)
        policy_matches = policy_matches.sort_values(["_edge_row_id", "threshold", "roi"], ascending=[True, False, False])
        policy_matches = policy_matches.drop_duplicates("_edge_row_id", keep="first")
        update_cols = ["_edge_row_id", *policy_cols.keys()]
        update_cols = [col for col in update_cols if col in policy_matches.columns]
        policy_matches = policy_matches[update_cols].rename(columns=policy_cols)
        out = out.merge(policy_matches, on="_edge_row_id", how="left", suffixes=("", "_policy_value"))
        for col in EDGE_RECOMMENDATION_COLUMNS:
            value_col = f"{col}_policy_value"
            if value_col in out.columns:
                out[col] = out[value_col].where(out[value_col].notna(), out[col])
                out = out.drop(columns=value_col)
        if "_edge_policy_tier" in out.columns:
            out["_edge_policy_tier"] = out["_edge_policy_tier"].where(out["_edge_policy_tier"].notna(), pd.NA)
    else:
        out["_edge_policy_tier"] = pd.NA

    edge_probability = pd.to_numeric(out.get("edge_probability_edge", pd.Series(np.nan, index=out.index)), errors="coerce")
    edge_ev = pd.to_numeric(out.get("edge_ev", pd.Series(np.nan, index=out.index)), errors="coerce")
    threshold = pd.to_numeric(out["edge_policy_threshold"], errors="coerce")
    bettable = out.get("bettable_flag", pd.Series(False, index=out.index)).astype("boolean").fillna(False).astype(bool)
    has_policy = threshold.notna()
    recommended = bettable & has_policy & edge_probability.ge(threshold) & edge_ev.gt(0)
    out["edge_recommendation_flag"] = recommended
    out["edge_recommendation_tier"] = np.select(
        [
            recommended & out.get("_edge_policy_tier", pd.Series(pd.NA, index=out.index)).eq("strong"),
            recommended,
            has_policy,
        ],
        ["strong", "lean", "below_threshold"],
        default="research_only",
    )
    out.loc[has_policy & ~bettable, "edge_recommendation_tier"] = "unavailable"
    return out.drop(columns=["_edge_market_key", "_edge_sportsbook_key", "_edge_row_id", "_edge_policy_tier"], errors="ignore")


def _position_group(position: object) -> str | None:
    if position is None or pd.isna(position):
        return None
    pos = str(position).strip().upper()
    if not pos or pos in {"NONE", "NAN", "NA", "<NA>"}:
        return None
    if pos == "QB":
        return "QB"
    if pos in {"RB", "FB"}:
        return "RB"
    if pos == "WR":
        return "WR"
    if pos == "TE":
        return "TE"
    if pos in {"DE", "DT", "NT", "DL", "EDGE"}:
        return "DL"
    if pos in {"LB", "ILB", "OLB", "MLB"}:
        return "LB"
    if pos in {"CB", "DB", "S", "FS", "SS"}:
        return "DB"
    return pos


def _player_name_parts(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = re.sub(r"\s+\([A-Z]{2,4}\)\s*$", "", text)
    if "," in text:
        parts = [part.strip() for part in text.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            text = f"{parts[1]} {parts[0]}"
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return text.split()


def _player_name_key(value: object) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first_initial = parts[0][0]
    last = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{first_initial}:{last}"


def _player_name_prefix_key(value: object, *, prefix_len: int = 2) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first = parts[0][: max(1, int(prefix_len))]
    last = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{first}:{last}"


def _player_initial_last_key(value: object) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first_initial = parts[0][0]
    last = parts[-1]
    return f"{first_initial}:{last}"


def _full_name(first: object, last: object) -> str | None:
    pieces = [str(part).strip() for part in [first, last] if part is not None and not pd.isna(part) and str(part).strip()]
    return " ".join(pieces) if pieces else None


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _infer_opponent(game_id: object, team: object) -> str | None:
    team_clean = _clean_team(team)
    if not team_clean or game_id is None or pd.isna(game_id):
        return None
    parts = str(game_id).split("_")
    if len(parts) < 4:
        return None
    away = _clean_team(parts[-2])
    home = _clean_team(parts[-1])
    if team_clean == home:
        return away
    if team_clean == away:
        return home
    return None


def _infer_team_side(game_id: object, team: object) -> str | None:
    team_clean = _clean_team(team)
    if not team_clean or game_id is None or pd.isna(game_id):
        return None
    parts = str(game_id).split("_")
    if len(parts) < 4:
        return None
    away = _clean_team(parts[-2])
    home = _clean_team(parts[-1])
    if team_clean == away:
        return "away"
    if team_clean == home:
        return "home"
    return None


def _load_position_lookup(labels: pd.DataFrame, rosters_path: Path = DEFAULT_ROSTERS_PATH) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    if not labels.empty and {"player_id", "position", "position_group", "player_name"}.issubset(labels.columns):
        label_pos = labels[["season", "week", "player_id", "player_name", "position", "position_group"]].copy()
        label_pos["source_priority"] = 0
        records.append(label_pos)

    if rosters_path.exists():
        try:
            if rosters_path.suffix == ".parquet":
                rosters = pd.read_parquet(
                    rosters_path,
                    columns=["season", "week", "gsis_id", "full_name", "football_name", "position"],
                    engine="fastparquet",
                )
            else:
                rosters = read_df(
                    rosters_path,
                    usecols=["season", "week", "gsis_id", "full_name", "football_name", "position"],
                )
        except Exception:
            rosters = read_df(rosters_path)
            keep = [col for col in ["season", "week", "gsis_id", "full_name", "football_name", "position"] if col in rosters.columns]
            rosters = rosters[keep]
        if not rosters.empty and {"gsis_id", "position"}.issubset(rosters.columns):
            roster_pos = rosters.copy()
            roster_pos = roster_pos.rename(columns={"gsis_id": "player_id", "football_name": "player_name"})
            if "player_name" not in roster_pos.columns:
                roster_pos["player_name"] = roster_pos.get("full_name", pd.NA)
            roster_pos["position_group"] = roster_pos["position"].apply(_position_group)
            roster_pos["source_priority"] = 1
            for col in ["season", "week"]:
                if col not in roster_pos.columns:
                    roster_pos[col] = -1
            records.append(roster_pos[["season", "week", "player_id", "player_name", "position", "position_group", "source_priority"]])

    if not records:
        return pd.DataFrame(columns=["player_id", "player_name_lookup", "position", "position_group"])

    lookup = pd.concat(records, ignore_index=True)
    lookup["season"] = pd.to_numeric(lookup["season"], errors="coerce").fillna(-1).astype(int)
    lookup["week"] = pd.to_numeric(lookup["week"], errors="coerce").fillna(-1).astype(int)
    lookup["player_id"] = _as_text(lookup["player_id"])
    lookup["position"] = _as_text(lookup["position"]).str.upper()
    lookup["position_group"] = lookup["position_group"].where(lookup["position_group"].notna(), lookup["position"].apply(_position_group))
    lookup = lookup.dropna(subset=["player_id", "position"])
    lookup = lookup.sort_values(["player_id", "season", "week", "source_priority"], kind="stable")
    lookup = lookup.drop_duplicates(subset=["player_id"], keep="last")
    return lookup.rename(columns={"player_name": "player_name_lookup"})[
        ["player_id", "player_name_lookup", "position", "position_group"]
    ]


def _prepare_projection_base(frame: pd.DataFrame, position_lookup: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna()]
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    for col in ["game_id", "player_id", "player_name", "team", "team_side"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["game_id"] = _as_text(out["game_id"])
    out["player_id"] = _as_text(out["player_id"])
    out["player_name"] = _as_text(out["player_name"])
    out["team"] = out["team"].apply(_clean_team)
    out["opponent"] = out.apply(lambda row: _infer_opponent(row["game_id"], row["team"]), axis=1)

    if not position_lookup.empty:
        out = out.merge(position_lookup, on="player_id", how="left")
        out["player_name"] = out["player_name"].where(out["player_name"].notna(), out["player_name_lookup"])
        out = out.drop(columns=["player_name_lookup"], errors="ignore")
    else:
        out["position"] = pd.NA
        out["position_group"] = pd.NA

    out["position"] = _as_text(out["position"]).str.upper()
    out["position_group"] = out["position_group"].where(out["position_group"].notna(), out["position"].apply(_position_group))
    return out.dropna(subset=["season", "week", "game_id", "team", "opponent", "player_id"])


def _candidate_rows_from_projection(
    frame: pd.DataFrame,
    *,
    market: str,
    baseline_col: str,
) -> pd.DataFrame:
    if frame.empty or baseline_col not in frame.columns:
        return pd.DataFrame()
    rows = frame.copy()
    rows["baseline_projection"] = pd.to_numeric(rows[baseline_col], errors="coerce")
    rows = rows[rows["baseline_projection"].notna()]
    rows["market"] = market
    rows["source_projection_col"] = baseline_col
    return rows


def build_prediction_candidates(
    *,
    labels: pd.DataFrame,
    qb_path: Path,
    offense_path: Path,
    defense_path: Path,
    rosters_path: Path = DEFAULT_ROSTERS_PATH,
) -> pd.DataFrame:
    position_lookup = _load_position_lookup(labels, rosters_path)
    qb = _prepare_projection_base(_read_csv_if_exists(qb_path), position_lookup)
    offense = _prepare_projection_base(_read_csv_if_exists(offense_path), position_lookup)
    defense = _prepare_projection_base(_read_csv_if_exists(defense_path), position_lookup)

    candidate_frames: list[pd.DataFrame] = []
    if not qb.empty:
        qb = qb.copy()
        qb["position"] = qb["position"].where(qb["position"].notna(), "QB")
        qb["position_group"] = "QB"
        candidate_frames.append(
            _candidate_rows_from_projection(qb, market="qb_passing_yards", baseline_col="projected_passing_yards")
        )
        candidate_frames.append(
            _candidate_rows_from_projection(qb, market="qb_passing_tds", baseline_col="projected_passing_tds")
        )
        candidate_frames.append(
            _candidate_rows_from_projection(qb, market="anytime_td", baseline_col="projected_rushing_tds")
        )

    if not offense.empty:
        rb = offense[offense["position_group"].eq("RB")].copy()
        wrte = offense[offense["position_group"].isin(["WR", "TE"])].copy()
        anytime = offense[offense["position_group"].isin(["RB", "WR", "TE"])].copy()
        candidate_frames.append(
            _candidate_rows_from_projection(rb, market="rb_rushing_yards", baseline_col="projected_rushing_yards")
        )
        candidate_frames.append(
            _candidate_rows_from_projection(rb, market="rb_rushing_tds", baseline_col="projected_rushing_tds")
        )
        candidate_frames.append(
            _candidate_rows_from_projection(
                wrte,
                market="wrte_receiving_yards",
                baseline_col="projected_receiving_yards",
            )
        )
        candidate_frames.append(
            _candidate_rows_from_projection(
                wrte,
                market="wrte_receiving_tds",
                baseline_col="projected_receiving_tds",
            )
        )
        candidate_frames.append(
            _candidate_rows_from_projection(anytime, market="anytime_td", baseline_col="projected_total_tds")
        )

    if not defense.empty:
        defense = defense[defense["position_group"].isin(["DL", "LB", "DB"]) | defense["position_group"].isna()].copy()
        candidate_frames.append(
            _candidate_rows_from_projection(defense, market="def_sacks", baseline_col="projected_sacks")
        )

    candidate_frames = [frame for frame in candidate_frames if not frame.empty]
    if not candidate_frames:
        return pd.DataFrame()
    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)
    candidates["market_family"] = candidates["market"].map(lambda market: "defense" if market == "def_sacks" else "offense")
    candidates = candidates.drop_duplicates(subset=["season", "week", "game_id", "team", "player_id", "market"], keep="last")
    return candidates


def _prior_player_market_summary(
    labels: pd.DataFrame,
    *,
    market: str,
    season: int,
    week: int,
) -> pd.DataFrame:
    required = {"season", "week", "player_id", "market", "actual_value"}
    if labels.empty or not required.issubset(labels.columns):
        return pd.DataFrame()

    out = labels.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["actual_value"] = pd.to_numeric(out["actual_value"], errors="coerce")
    out["market"] = _as_text(out["market"]).str.lower()
    out["player_id"] = _as_text(out["player_id"])
    out = out[
        out["season"].notna()
        & out["week"].notna()
        & out["actual_value"].notna()
        & out["market"].eq(str(market).lower())
    ].copy()
    if out.empty:
        return pd.DataFrame()

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out = out[(out["season"].lt(int(season))) | (out["season"].eq(int(season)) & out["week"].lt(int(week)))]
    if out.empty:
        return pd.DataFrame()

    for col in ["player_name", "position", "position_group"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = _as_text(out[col])

    latest = (
        out.sort_values(["player_id", "season", "week"], kind="stable")
        .drop_duplicates("player_id", keep="last")[
            ["player_id", "player_name", "position", "position_group"]
        ]
        .rename(
            columns={
                "player_name": "history_player_name",
                "position": "history_position",
                "position_group": "history_position_group",
            }
        )
    )
    summary = (
        out.groupby("player_id", as_index=False)
        .agg(
            baseline_projection=("actual_value", "mean"),
            games_sampled=("actual_value", "count"),
        )
        .merge(latest, on="player_id", how="left")
    )
    summary["baseline_projection"] = pd.to_numeric(summary["baseline_projection"], errors="coerce")
    summary["games_sampled"] = pd.to_numeric(summary["games_sampled"], errors="coerce").fillna(0).astype(int)
    return summary[summary["baseline_projection"].notna() & summary["games_sampled"].gt(0)].reset_index(drop=True)


def _active_availability_lookup(availability: pd.DataFrame | None, *, season: int, week: int) -> pd.DataFrame:
    if availability is None or availability.empty or "player_id" not in availability.columns:
        return pd.DataFrame()

    out = availability.copy()
    if {"season", "week"}.issubset(out.columns):
        season_values = pd.to_numeric(out["season"], errors="coerce")
        week_values = pd.to_numeric(out["week"], errors="coerce")
        exact_week = season_values.eq(int(season)) & week_values.eq(int(week))
        if exact_week.any():
            out = out[exact_week].copy()

    out["player_id"] = _as_text(out["player_id"])
    team_source = out["current_team"] if "current_team" in out.columns else out.get("team", pd.Series(pd.NA, index=out.index))
    out["availability_team"] = team_source.apply(_clean_team)
    for col in ["active_roster_flag", "practice_squad_flag", "injured_reserve_flag"]:
        if col not in out.columns:
            out[col] = False
    active = out["active_roster_flag"].astype("boolean").fillna(False)
    practice = out["practice_squad_flag"].astype("boolean").fillna(False)
    injured_reserve = out["injured_reserve_flag"].astype("boolean").fillna(False)
    out = out[active & ~practice & ~injured_reserve].copy()
    out = out.dropna(subset=["player_id", "availability_team"])
    if out.empty:
        return pd.DataFrame()

    for col in ["player_name", "position", "position_group", "availability_priority"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["availability_priority"] = pd.to_numeric(out["availability_priority"], errors="coerce").fillna(0)
    out["position"] = _as_text(out["position"]).str.upper()
    out["position_group"] = out["position_group"].where(out["position_group"].notna(), out["position"].apply(_position_group))
    out = out.sort_values(["player_id", "availability_priority"], ascending=[True, False], kind="stable")
    out = out.drop_duplicates("player_id", keep="first")
    return out[
        ["player_id", "availability_team", "player_name", "position", "position_group", "availability_priority"]
    ].rename(
        columns={
            "player_name": "availability_player_name",
            "position": "availability_position",
            "position_group": "availability_position_group",
        }
    )


def _line_offered_def_sack_candidates(
    *,
    labels: pd.DataFrame,
    odds: pd.DataFrame | None,
    player_identity: pd.DataFrame | None,
    availability: pd.DataFrame | None,
    season: int,
    week: int,
) -> pd.DataFrame:
    if odds is None or odds.empty:
        return pd.DataFrame()

    odds_required = {"game_id", "market", "player_name_key", "player_name_prefix_key"}
    if not odds_required.issubset(odds.columns):
        return pd.DataFrame()

    identity = _prepare_player_identity(player_identity)
    active_players = _active_availability_lookup(availability, season=season, week=week)
    history = _prior_player_market_summary(labels, market="def_sacks", season=season, week=week)
    if identity.empty or active_players.empty or history.empty:
        return pd.DataFrame()

    line_odds = odds[odds["market"].eq("def_sacks")].dropna(
        subset=["game_id", "player_name_key", "player_name_prefix_key"]
    ).copy()
    if line_odds.empty:
        return pd.DataFrame()

    parts = line_odds["game_id"].apply(_game_identity_parts)
    line_odds["_identity_season"] = [part[0] for part in parts]
    line_odds["_identity_week"] = [part[1] for part in parts]
    line_odds["_identity_candidate_team"] = [part[2] for part in parts]
    line_odds = line_odds.explode("_identity_candidate_team")
    line_odds = line_odds.dropna(
        subset=["_identity_season", "_identity_week", "_identity_candidate_team"]
    ).copy()
    if line_odds.empty:
        return pd.DataFrame()

    line_odds["_identity_season"] = line_odds["_identity_season"].astype(int)
    line_odds["_identity_week"] = line_odds["_identity_week"].astype(int)
    line_odds = line_odds[
        line_odds["_identity_season"].eq(int(season)) & line_odds["_identity_week"].eq(int(week))
    ].copy()
    if line_odds.empty:
        return pd.DataFrame()

    line_identity = line_odds.merge(
        identity,
        left_on=[
            "_identity_season",
            "_identity_week",
            "_identity_candidate_team",
            "player_name_key",
            "player_name_prefix_key",
        ],
        right_on=[
            "identity_season",
            "identity_week",
            "identity_team",
            "alias_name_key",
            "alias_prefix_key",
        ],
        how="inner",
    )
    if line_identity.empty:
        return pd.DataFrame()

    line_identity = line_identity.rename(columns={"identity_player_id": "player_id"})
    line_identity = line_identity.merge(active_players, on="player_id", how="inner")
    line_identity = line_identity[line_identity["availability_team"].eq(line_identity["identity_team"])].copy()
    if line_identity.empty:
        return pd.DataFrame()

    line_identity = line_identity.merge(history, on="player_id", how="inner")
    if line_identity.empty:
        return pd.DataFrame()

    line_identity["team"] = line_identity["identity_team"].apply(_clean_team)
    line_identity["opponent"] = line_identity.apply(lambda row: _infer_opponent(row["game_id"], row["team"]), axis=1)
    line_identity["team_side"] = line_identity.apply(lambda row: _infer_team_side(row["game_id"], row["team"]), axis=1)
    line_identity["player_name"] = line_identity["availability_player_name"].where(
        line_identity["availability_player_name"].notna(),
        line_identity["history_player_name"],
    )
    line_identity["position"] = line_identity["availability_position"].where(
        line_identity["availability_position"].notna(),
        line_identity["history_position"],
    )
    line_identity["position_group"] = line_identity["availability_position_group"].where(
        line_identity["availability_position_group"].notna(),
        line_identity["history_position_group"],
    )
    line_identity["position"] = _as_text(line_identity["position"]).str.upper()
    line_identity["position_group"] = line_identity["position_group"].where(
        line_identity["position_group"].notna(),
        line_identity["position"].apply(_position_group),
    )
    line_identity = line_identity[
        line_identity["team"].notna()
        & line_identity["opponent"].notna()
        & line_identity["team_side"].notna()
        & line_identity["player_name"].notna()
        & line_identity["position_group"].isin(["DL", "LB", "DB"])
    ].copy()
    if line_identity.empty:
        return pd.DataFrame()

    line_identity["season"] = int(season)
    line_identity["week"] = int(week)
    line_identity["market"] = "def_sacks"
    line_identity["market_family"] = "defense"
    line_identity["source_projection_col"] = "historical_def_sacks_avg"
    line_identity["player_rank"] = (
        line_identity.sort_values(
            ["game_id", "team", "baseline_projection", "player_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .groupby(["game_id", "team", "market"], sort=False)
        .cumcount()
        + 1
    )
    keep = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "team_side",
        "player_id",
        "player_name",
        "position",
        "position_group",
        "market",
        "market_family",
        "baseline_projection",
        "source_projection_col",
        "games_sampled",
        "player_rank",
    ]
    return line_identity[keep].drop_duplicates(
        subset=["season", "week", "game_id", "team", "player_id", "market"],
        keep="first",
    )


def _add_line_offered_def_sack_candidates(
    candidates: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    odds: pd.DataFrame | None,
    player_identity: pd.DataFrame | None,
    availability: pd.DataFrame | None,
    season: int,
    week: int,
) -> pd.DataFrame:
    additions = _line_offered_def_sack_candidates(
        labels=labels,
        odds=odds,
        player_identity=player_identity,
        availability=availability,
        season=season,
        week=week,
    )
    if additions.empty:
        return candidates
    if candidates.empty:
        return additions

    combined = pd.concat([candidates, additions], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=["season", "week", "game_id", "team", "player_id", "market"],
        keep="first",
    )
    if "player_rank" not in combined.columns:
        combined["player_rank"] = np.nan
    rank_missing = combined["player_rank"].isna() & combined["market"].eq("def_sacks")
    if rank_missing.any() and "baseline_projection" in combined.columns:
        computed = (
            combined[combined["market"].eq("def_sacks")]
            .sort_values(
                ["game_id", "team", "baseline_projection", "player_id"],
                ascending=[True, True, False, True],
                kind="stable",
            )
            .groupby(["game_id", "team", "market"], sort=False)
            .cumcount()
            + 1
        )
        combined.loc[computed.index.intersection(combined.index), "_computed_player_rank"] = computed
        combined["player_rank"] = combined["player_rank"].where(
            combined["player_rank"].notna(),
            combined.get("_computed_player_rank", pd.Series(np.nan, index=combined.index)),
        )
        combined = combined.drop(columns=["_computed_player_rank"], errors="ignore")
    return combined


def _add_total_td_projection(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "projected_total_tds" not in out.columns:
        out["projected_total_tds"] = np.nan
    total = pd.to_numeric(out["projected_total_tds"], errors="coerce")
    component_cols = [col for col in ["projected_rushing_tds", "projected_receiving_tds"] if col in out.columns]
    if component_cols:
        component_total = pd.Series(0.0, index=out.index)
        for col in component_cols:
            component_total = component_total.add(pd.to_numeric(out[col], errors="coerce").fillna(0.0), fill_value=0.0)
        total = total.where(total.notna(), component_total)
    out["projected_total_tds"] = total
    return out


def _novelty_td_candidate_rows(frame: pd.DataFrame, *, baseline_col: str) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for market in sorted(NOVELTY_OUTPUT_MARKETS):
        rows.append(_candidate_rows_from_projection(frame, market=market, baseline_col=baseline_col))
    return rows


def build_novelty_prediction_candidates(
    *,
    labels: pd.DataFrame,
    qb_path: Path,
    offense_path: Path,
    rosters_path: Path = DEFAULT_ROSTERS_PATH,
) -> pd.DataFrame:
    position_lookup = _load_position_lookup(labels, rosters_path)
    qb = _prepare_projection_base(_read_csv_if_exists(qb_path), position_lookup)
    offense = _prepare_projection_base(_read_csv_if_exists(offense_path), position_lookup)

    candidate_frames: list[pd.DataFrame] = []
    if not qb.empty:
        qb = qb.copy()
        qb["position"] = qb["position"].where(qb["position"].notna(), "QB")
        qb["position_group"] = "QB"
        candidate_frames.extend(_novelty_td_candidate_rows(qb, baseline_col="projected_rushing_tds"))

    if not offense.empty:
        offense = _add_total_td_projection(offense)
        td_players = offense[offense["position_group"].isin(["RB", "WR", "TE"])].copy()
        candidate_frames.extend(_novelty_td_candidate_rows(td_players, baseline_col="projected_total_tds"))

    candidate_frames = [frame for frame in candidate_frames if not frame.empty]
    if not candidate_frames:
        return pd.DataFrame()
    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)
    candidates["market_family"] = "novelty"
    candidates = candidates.drop_duplicates(subset=["season", "week", "game_id", "team", "player_id", "market"], keep="last")
    return candidates


def _confidence_tier(sample_size: object) -> str:
    try:
        sample = int(float(sample_size))
    except (TypeError, ValueError):
        sample = 0
    if sample >= 20:
        return "high"
    if sample >= 5:
        return "medium"
    return "low"


def _load_model_bundle(models_dir: Path, family: str, market: str) -> dict[str, object] | None:
    global _SKLEARN_VERSION_WARNING_EMITTED
    path = models_dir / family / f"{market}_model.pkl"
    if not path.exists():
        return None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = joblib.load(path)
    version_warnings = [warning for warning in caught if issubclass(warning.category, InconsistentVersionWarning)]
    if version_warnings and not _SKLEARN_VERSION_WARNING_EMITTED:
        print(
            "[player_props.predict] warning: player prop model artifacts were created with a different "
            "scikit-learn version than the active environment. Run `python -m src.player_props.train "
            "--start-season 2017 --exclude-from-season <season> --exclude-from-week <week>` in this "
            "environment to refresh models and remove this compatibility warning.",
            file=sys.stderr,
        )
        _SKLEARN_VERSION_WARNING_EMITTED = True
    for warning in caught:
        if not issubclass(warning.category, InconsistentVersionWarning):
            warnings.warn(warning.message, warning.category, stacklevel=2)
    return bundle


def _default_odds_path(season: int, week: int) -> Path:
    return RAW_DIR / f"propline_player_props_{int(season)}_wk{int(week):02d}.parquet"


def _legacy_bdl_odds_path(season: int, week: int) -> Path:
    return RAW_DIR / f"balldontlie_player_props_{int(season)}_wk{int(week):02d}.parquet"


def _read_frame_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.exists():
        return read_df(path)
    if path.suffix == ".parquet":
        csv_fallback = path.with_suffix(".csv")
        if csv_fallback.exists():
            return read_df(csv_fallback)
    return pd.DataFrame()


def _load_bdl_games_for_odds(season: int, week: int) -> pd.DataFrame:
    paths = [
        RAW_DIR / f"balldontlie_games_{int(season)}_wk{int(week):02d}.parquet",
        RAW_DIR / f"balldontlie_games_{int(season)}.parquet",
    ]
    frames = [_read_frame_if_exists(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True, sort=False)
    if "week" in games.columns:
        games = games[pd.to_numeric(games["week"], errors="coerce").eq(int(week))]
    return games


def _load_bdl_players_for_odds() -> pd.DataFrame:
    frames = [
        _read_frame_if_exists(RAW_DIR / "balldontlie_active_players.parquet"),
        _read_frame_if_exists(RAW_DIR / "balldontlie_players.parquet"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _prepare_bdl_game_lookup(games: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "season", "week", "visitor_team.abbreviation", "home_team.abbreviation"}
    if games.empty or not required.issubset(games.columns):
        return pd.DataFrame(columns=["bdl_game_id", "game_id"])
    out = games[list(required)].copy()
    out["bdl_game_id"] = pd.to_numeric(out["id"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["away"] = out["visitor_team.abbreviation"].apply(_clean_team)
    out["home"] = out["home_team.abbreviation"].apply(_clean_team)
    out = out.dropna(subset=["bdl_game_id", "season", "week", "away", "home"])
    out["bdl_game_id"] = out["bdl_game_id"].astype(int)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["game_id"] = out.apply(lambda row: f"{row['season']}_{row['week']:02d}_{row['away']}_{row['home']}", axis=1)
    return out[["bdl_game_id", "game_id"]].drop_duplicates()


def _prepare_bdl_player_lookup(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty or "id" not in players.columns:
        return pd.DataFrame(columns=["bdl_player_id", "player_name_key", "player_name_prefix_key", "bdl_team", "bdl_position"])
    out = players.copy()
    out["bdl_player_id"] = pd.to_numeric(out["id"], errors="coerce")
    first = out["first_name"] if "first_name" in out.columns else pd.Series(pd.NA, index=out.index)
    last = out["last_name"] if "last_name" in out.columns else pd.Series(pd.NA, index=out.index)
    out["bdl_player_name"] = [_full_name(f, l) for f, l in zip(first, last)]
    out["player_name_key"] = out["bdl_player_name"].apply(_player_name_key)
    out["player_name_prefix_key"] = out["bdl_player_name"].apply(_player_name_prefix_key)
    out["player_initial_last_key"] = out["bdl_player_name"].apply(_player_initial_last_key)
    team_col = "team.abbreviation" if "team.abbreviation" in out.columns else None
    out["bdl_team"] = out[team_col].apply(_clean_team) if team_col else pd.NA
    pos_col = "position_abbreviation" if "position_abbreviation" in out.columns else None
    out["bdl_position"] = out[pos_col].astype("string").str.upper() if pos_col else pd.NA
    out = out.dropna(subset=["bdl_player_id", "player_name_key"])
    out["bdl_player_id"] = out["bdl_player_id"].astype(int)
    out = out.sort_values(["bdl_player_id", "bdl_team"], kind="stable").drop_duplicates("bdl_player_id", keep="first")
    return out[
        [
            "bdl_player_id",
            "player_name_key",
            "player_name_prefix_key",
            "player_initial_last_key",
            "bdl_team",
            "bdl_position",
        ]
    ]


def _prepare_bdl_player_prop_odds(
    odds: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    *,
    vendor: str | Sequence[str] = DEFAULT_ODDS_VENDORS,
) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame()
    required = {"game_id", "player_id", "vendor", "prop_type", "line_value"}
    if not required.issubset(odds.columns):
        return pd.DataFrame()

    game_lookup = _prepare_bdl_game_lookup(games)
    player_lookup = _prepare_bdl_player_lookup(players)
    if game_lookup.empty or player_lookup.empty:
        return pd.DataFrame()

    out = odds.copy()
    out["vendor"] = out["vendor"].astype("string").str.lower()
    vendors = _normalize_vendors(vendor)
    if vendors:
        out = out[out["vendor"].isin(vendors)]
    out["market"] = out["prop_type"].map(BDL_PROP_TYPE_TO_MARKET)
    out = out[out["market"].notna()]
    out["bdl_game_id"] = pd.to_numeric(out["game_id"], errors="coerce")
    out["bdl_player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out["line"] = pd.to_numeric(out["line_value"], errors="coerce")
    out = out.dropna(subset=["bdl_game_id", "bdl_player_id", "line"])
    out["bdl_game_id"] = out["bdl_game_id"].astype(int)
    out["bdl_player_id"] = out["bdl_player_id"].astype(int)
    out = out.drop(columns=["game_id"], errors="ignore")
    out = out.merge(game_lookup, on="bdl_game_id", how="inner")
    out = out.merge(player_lookup, on="bdl_player_id", how="inner")

    over_odds_col = "market.over_odds" if "market.over_odds" in out.columns else None
    under_odds_col = "market.under_odds" if "market.under_odds" in out.columns else None
    out["over_odds"] = pd.to_numeric(out[over_odds_col], errors="coerce") if over_odds_col else np.nan
    out["under_odds"] = pd.to_numeric(out[under_odds_col], errors="coerce") if under_odds_col else np.nan
    over_prob = moneyline_to_prob(out["over_odds"])
    under_prob = moneyline_to_prob(out["under_odds"])
    total_prob = over_prob + under_prob
    out["implied_probability"] = np.where(total_prob > 0, over_prob / total_prob, over_prob)
    out["line_balance_distance"] = _line_balance_distance(out)
    out["line_is_comparable"] = out["line_balance_distance"].le(LINE_BALANCE_DISTANCE_LIMIT)
    out["line_type"] = "canonical"
    if "updated_at" not in out.columns:
        out["updated_at"] = pd.NA

    out = out.sort_values(["game_id", "market", "player_name_key", "vendor", "updated_at"], kind="stable")
    out = out.drop_duplicates(subset=["game_id", "market", "player_name_key", "vendor"], keep="last")
    return out[
        [
            "game_id",
            "market",
            "player_name_key",
            "player_name_prefix_key",
            "player_initial_last_key",
            "bdl_team",
            "line",
            "line_type",
            "implied_probability",
            "vendor",
            "over_odds",
            "under_odds",
            "line_balance_distance",
            "line_is_comparable",
            "updated_at",
        ]
    ].rename(columns={"vendor": "sportsbook", "updated_at": "line_updated_at"})


def _first_present(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for col in columns:
        if col in frame.columns:
            result = result.where(result.notna(), frame[col])
    return result


def _prepare_propline_player_prop_odds(
    odds: pd.DataFrame,
    *,
    season: int,
    week: int,
    vendor: str | Sequence[str] = DEFAULT_ODDS_VENDORS,
) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame()
    required = {
        "home_team",
        "away_team",
        "bookmaker_key",
        "market_key",
        "outcome_name",
        "player_name",
        "price",
        "point",
    }
    if not required.issubset(odds.columns):
        return pd.DataFrame()

    out = odds.copy()
    out["sportsbook"] = out["bookmaker_key"].astype("string").str.lower()
    vendors = _normalize_vendors(vendor)
    if vendors:
        out = out[out["sportsbook"].isin(vendors)]
    out["market"] = out["market_key"].map(PROPLINE_MARKET_TO_MODEL_MARKET)
    out = out[out["market"].notna()]
    out["home_abbr"] = out["home_team"].apply(_team_abbr_from_name)
    out["away_abbr"] = out["away_team"].apply(_team_abbr_from_name)
    out = out.dropna(subset=["home_abbr", "away_abbr", "player_name"])
    computed_game_id = out.apply(
        lambda row: f"{int(season)}_{int(week):02d}_{row['away_abbr']}_{row['home_abbr']}",
        axis=1,
    )
    if "game_id" in out.columns:
        source_game_id = out["game_id"].astype("string").str.strip()
        out["game_id"] = source_game_id.where(source_game_id.notna() & source_game_id.ne(""), computed_game_id)
    else:
        out["game_id"] = computed_game_id
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    out["player_name_prefix_key"] = out["player_name"].apply(_player_name_prefix_key)
    out["player_initial_last_key"] = out["player_name"].apply(_player_initial_last_key)
    out["line"] = pd.to_numeric(out["point"], errors="coerce")
    outcome_name = out["outcome_name"].astype("string")
    market_key = out["market_key"].astype("string").str.lower()
    touchdown_scorer_line = market_key.map(ONE_SIDED_TOUCHDOWN_MARKET_LINES)
    one_sided_touchdown_mask = touchdown_scorer_line.notna()
    milestone_mask = outcome_name.str.contains(r"(?i)\b\d+(?:\.\d+)?\s*\+", na=False)
    milestone_line = outcome_name.str.extract(r"(?i)\b(\d+(?:\.\d+)?)\s*\+")[0]
    out["line"] = out["line"].where(out["line"].notna(), pd.to_numeric(milestone_line, errors="coerce"))
    out["line"] = out["line"].where(
        out["line"].notna(),
        pd.to_numeric(touchdown_scorer_line, errors="coerce"),
    )
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["line_type"] = np.where(milestone_mask | one_sided_touchdown_mask, "milestone", "canonical")
    out["outcome_side"] = outcome_name.str.lower()
    out.loc[one_sided_touchdown_mask, "outcome_side"] = "over"
    out.loc[milestone_mask, "outcome_side"] = "over"
    out["line_updated_at"] = _first_present(out, ["last_change_at", "market_last_update", "book_updated_at"])
    out = out.dropna(subset=["game_id", "market", "player_name_key", "line", "price"])

    key_cols = [
        "game_id",
        "market",
        "player_name_key",
        "player_name_prefix_key",
        "player_initial_last_key",
        "line",
        "line_type",
        "sportsbook",
    ]
    over = out[out["outcome_side"].str.contains("over", na=False)].copy()
    under = out[out["outcome_side"].str.contains("under", na=False)].copy()
    over = over.sort_values(key_cols + ["line_updated_at"], kind="stable").drop_duplicates(key_cols, keep="last")
    under = under.sort_values(key_cols + ["line_updated_at"], kind="stable").drop_duplicates(key_cols, keep="last")

    base_cols = key_cols + ["line_updated_at"]
    over = over[base_cols + ["price"]].rename(columns={"price": "over_odds", "line_updated_at": "over_updated_at"})
    under = under[base_cols + ["price"]].rename(columns={"price": "under_odds", "line_updated_at": "under_updated_at"})
    merged = over.merge(under, on=key_cols, how="outer")
    if merged.empty:
        return pd.DataFrame()
    merged["line_updated_at"] = merged["over_updated_at"].where(merged["over_updated_at"].notna(), merged["under_updated_at"])
    over_prob = moneyline_to_prob(merged["over_odds"])
    under_prob = moneyline_to_prob(merged["under_odds"])
    total_prob = over_prob + under_prob
    merged["implied_probability"] = np.where(total_prob > 0, over_prob / total_prob, over_prob)
    merged["line_balance_distance"] = _line_balance_distance(merged)
    merged["line_is_comparable"] = merged["line_balance_distance"].le(LINE_BALANCE_DISTANCE_LIMIT)
    main_line_keys = ["game_id", "market", "player_name_key", "sportsbook"]
    two_way_line_mask = (
        merged["line_type"].eq("canonical")
        & pd.to_numeric(merged["over_odds"], errors="coerce").notna()
        & pd.to_numeric(merged["under_odds"], errors="coerce").notna()
    )
    two_way_line_counts = (
        merged.loc[two_way_line_mask]
        .groupby(main_line_keys, dropna=False)["line"]
        .nunique(dropna=True)
        .reset_index(name="_two_way_line_count")
    )
    if not two_way_line_counts.empty:
        merged = merged.merge(two_way_line_counts, on=main_line_keys, how="left")
        multi_point_two_way = two_way_line_mask & merged["_two_way_line_count"].fillna(0).gt(1)
        merged.loc[multi_point_two_way, "line_type"] = "alternate"
        merged = merged.drop(columns=["_two_way_line_count"], errors="ignore")

    two_way_mask = (
        merged["line_type"].eq("canonical")
        & pd.to_numeric(merged["over_odds"], errors="coerce").notna()
        & pd.to_numeric(merged["under_odds"], errors="coerce").notna()
    )
    main_candidates = merged[two_way_mask].copy()
    if not main_candidates.empty:
        main_lines = (
            main_candidates.sort_values(main_line_keys + ["line_balance_distance", "line"], kind="stable")
            .drop_duplicates(subset=main_line_keys, keep="first")[main_line_keys + ["line"]]
            .copy()
        )
        groups_with_main = main_lines[main_line_keys].drop_duplicates().assign(_has_main_line=True)
        main_lines["_is_main_line"] = True
        merged = merged.merge(groups_with_main, on=main_line_keys, how="left")
        merged = merged.merge(main_lines, on=main_line_keys + ["line"], how="left")
        has_main_line = merged["_has_main_line"].eq(True)
        is_main_line = merged["_is_main_line"].eq(True)
        alternate_mask = (
            merged["line_type"].eq("canonical")
            & has_main_line
            & ~is_main_line
        )
        merged.loc[alternate_mask, "line_type"] = "alternate"
        merged = merged.drop(columns=["_has_main_line", "_is_main_line"], errors="ignore")
    return merged[
        [
            "game_id",
            "market",
            "player_name_key",
            "player_name_prefix_key",
            "player_initial_last_key",
            "line",
            "line_type",
            "implied_probability",
            "sportsbook",
            "over_odds",
            "under_odds",
            "line_balance_distance",
            "line_is_comparable",
            "line_updated_at",
        ]
    ].sort_values(["game_id", "market", "player_name_key", "line"], kind="stable")


def prepare_player_prop_odds(
    odds: pd.DataFrame,
    games: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
    vendor: str | Sequence[str] = DEFAULT_ODDS_VENDORS,
) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame()
    if {"bookmaker_key", "market_key", "outcome_name", "player_name", "price", "point"}.issubset(odds.columns):
        if season is None or week is None:
            return pd.DataFrame()
        return _prepare_propline_player_prop_odds(odds, season=int(season), week=int(week), vendor=vendor)
    if games is None or players is None:
        return pd.DataFrame()
    return _prepare_bdl_player_prop_odds(odds, games, players, vendor=vendor)


def load_player_prop_odds(
    *,
    season: int,
    week: int,
    odds_path: Path | None = None,
    vendor: str | Sequence[str] = DEFAULT_ODDS_VENDORS,
) -> pd.DataFrame:
    odds_file = odds_path or _default_odds_path(season, week)
    if not odds_file.exists() and odds_path is None:
        odds_file = _legacy_bdl_odds_path(season, week)
    odds = _read_frame_if_exists(odds_file)
    if odds.empty:
        return pd.DataFrame()
    if {"bookmaker_key", "market_key", "outcome_name", "player_name", "price", "point"}.issubset(odds.columns):
        return prepare_player_prop_odds(odds, season=season, week=week, vendor=vendor)
    return prepare_player_prop_odds(
        odds,
        _load_bdl_games_for_odds(season, week),
        _load_bdl_players_for_odds(),
        season=season,
        week=week,
        vendor=vendor,
    )


def _line_stats_by_prediction(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty or "line" not in joined.columns:
        return pd.DataFrame(columns=["_prediction_row_id"] + LINE_STAT_COLUMNS)

    valid = joined[pd.to_numeric(joined["line"], errors="coerce").notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=["_prediction_row_id"] + LINE_STAT_COLUMNS)

    if "line_type" in valid.columns:
        line_type = valid["line_type"].astype("string").str.lower()
        valid["_line_type_rank"] = np.where(line_type.eq("canonical").fillna(False), 0, 1)
        canonical_ids = set(valid.loc[valid["_line_type_rank"].eq(0), "_prediction_row_id"].dropna().tolist())
        if canonical_ids:
            valid = valid[
                valid["_line_type_rank"].eq(0) | ~valid["_prediction_row_id"].isin(canonical_ids)
            ].copy()
    else:
        valid["_line_type_rank"] = 0

    if "line_is_comparable" in valid.columns:
        comparable_mask = valid["line_is_comparable"].astype("boolean").fillna(False).astype(bool)
    else:
        comparable_mask = pd.Series(False, index=valid.index)
    comparable = valid[comparable_mask].copy()
    comparable_ids = set(comparable["_prediction_row_id"].dropna().tolist())
    fallback = valid[~valid["_prediction_row_id"].isin(comparable_ids)].copy()
    if not fallback.empty:
        if "_line_balance_distance" not in fallback.columns:
            if "line_balance_distance" in fallback.columns:
                fallback["_line_balance_distance"] = pd.to_numeric(
                    fallback["line_balance_distance"], errors="coerce"
                ).fillna(np.inf)
            else:
                fallback["_line_balance_distance"] = pd.Series(np.inf, index=fallback.index)
        if "_line_distance" not in fallback.columns:
            projection = (
                pd.to_numeric(fallback["projection"], errors="coerce")
                if "projection" in fallback.columns
                else pd.Series(np.nan, index=fallback.index)
            )
            fallback["_line_distance"] = (
                projection - pd.to_numeric(fallback["line"], errors="coerce")
            ).abs()
            fallback["_line_distance"] = fallback["_line_distance"].fillna(np.inf)
        fallback = fallback.sort_values(
            ["_prediction_row_id", "sportsbook", "_line_type_rank", "_line_balance_distance", "_line_distance", "line"],
            kind="stable",
        )
        fallback = fallback.drop_duplicates(subset=["_prediction_row_id", "sportsbook"], keep="first")
    stats_source = pd.concat([comparable, fallback], ignore_index=True, sort=False)
    if stats_source.empty:
        return pd.DataFrame(columns=["_prediction_row_id"] + LINE_STAT_COLUMNS)

    stats_source["line"] = pd.to_numeric(stats_source["line"], errors="coerce")
    stats = (
        stats_source.groupby("_prediction_row_id", as_index=False)
        .agg(
            line_consensus=("line", "mean"),
            line_min=("line", "min"),
            line_max=("line", "max"),
            line_std=("line", "std"),
            line_book_count=("sportsbook", "nunique"),
            line_options_count=("line", "count"),
            market=("market", "first"),
        )
        .reset_index(drop=True)
    )
    stats["line_range"] = stats["line_max"] - stats["line_min"]
    threshold = np.where(stats["market"].isin(COUNT_MARKETS), COUNT_LINE_VARIANCE_THRESHOLD, YARD_LINE_VARIANCE_THRESHOLD)
    stats["line_variance_flag"] = stats["line_range"].fillna(0).ge(threshold)
    stats["line_std"] = stats["line_std"].fillna(0)
    return stats.drop(columns=["market"], errors="ignore")


def _identity_matched_fallback_source(
    source: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_market: str,
    identity_table: pd.DataFrame,
) -> pd.DataFrame:
    prediction_required = {"season", "week", "game_id", "team", "player_id"}
    if identity_table.empty or not prediction_required.issubset(predictions.columns):
        return pd.DataFrame()

    identity_source = source.dropna(
        subset=["game_id", "player_name_key", "player_name_prefix_key"]
    ).copy()
    if identity_source.empty:
        return pd.DataFrame()
    parts = identity_source["game_id"].apply(_game_identity_parts)
    identity_source["_identity_season"] = [part[0] for part in parts]
    identity_source["_identity_week"] = [part[1] for part in parts]
    identity_source["_identity_candidate_team"] = [part[2] for part in parts]
    identity_source = identity_source.explode("_identity_candidate_team")
    identity_source = identity_source.dropna(
        subset=["_identity_season", "_identity_week", "_identity_candidate_team"]
    ).copy()
    if identity_source.empty:
        return pd.DataFrame()
    identity_source["_identity_season"] = identity_source["_identity_season"].astype(int)
    identity_source["_identity_week"] = identity_source["_identity_week"].astype(int)
    line_identity = identity_source.merge(
        identity_table,
        left_on=[
            "_identity_season",
            "_identity_week",
            "_identity_candidate_team",
            "player_name_key",
            "player_name_prefix_key",
        ],
        right_on=[
            "identity_season",
            "identity_week",
            "identity_team",
            "alias_name_key",
            "alias_prefix_key",
        ],
        how="inner",
    )
    if line_identity.empty:
        return pd.DataFrame()

    target_predictions = predictions[predictions["market"].eq(target_market)].copy()
    target_predictions = target_predictions.dropna(subset=["season", "week", "game_id", "team", "player_id"])
    if target_predictions.empty:
        return pd.DataFrame()
    target_predictions["season"] = pd.to_numeric(target_predictions["season"], errors="coerce")
    target_predictions["week"] = pd.to_numeric(target_predictions["week"], errors="coerce")
    target_predictions["team"] = target_predictions["team"].apply(_clean_team)
    target_predictions["player_id"] = _as_text(target_predictions["player_id"])
    target_predictions = target_predictions.dropna(subset=["season", "week", "team", "player_id"])
    target_predictions["season"] = target_predictions["season"].astype(int)
    target_predictions["week"] = target_predictions["week"].astype(int)

    matched = line_identity.merge(
        target_predictions[["season", "week", "game_id", "team", "player_id"]].drop_duplicates(),
        left_on=["identity_season", "identity_week", "game_id", "identity_team", "identity_player_id"],
        right_on=["season", "week", "game_id", "team", "player_id"],
        how="inner",
    )
    if matched.empty:
        return pd.DataFrame()
    return matched[list(source.columns)].copy()


def _add_anytime_td_fallback_odds(
    odds: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    identity_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if odds.empty or predictions.empty:
        return odds
    required_odds = {
        "game_id",
        "market",
        "player_name_key",
        "player_name_prefix_key",
        "sportsbook",
    }
    required_predictions = {"game_id", "market", "player_name_key", "player_name_prefix_key"}
    if not required_odds.issubset(odds.columns) or not required_predictions.issubset(predictions.columns):
        return odds

    source = odds[odds["market"].eq("anytime_td")].copy()
    if source.empty:
        return odds

    target_predictions = predictions[predictions["market"].isin(ANYTIME_TD_FALLBACK_TARGET_MARKETS)].copy()
    target_predictions = target_predictions.dropna(
        subset=["game_id", "market", "player_name_key", "player_name_prefix_key"]
    )
    if target_predictions.empty:
        return odds

    identity_table = identity_table if identity_table is not None else pd.DataFrame()
    exact_key_cols = ["game_id", "market", "player_name_key", "player_name_prefix_key", "sportsbook"]
    existing_exact = set(
        map(
            tuple,
            odds[odds["market"].isin(ANYTIME_TD_FALLBACK_TARGET_MARKETS)]
            .dropna(subset=exact_key_cols)[exact_key_cols]
            .astype(str)
            .to_numpy(),
        )
    )
    source_match_cols = ["game_id", "player_name_key", "player_name_prefix_key"]
    fallback_frames: list[pd.DataFrame] = []
    for target_market in sorted(ANYTIME_TD_FALLBACK_TARGET_MARKETS):
        target_keys = (
            target_predictions[target_predictions["market"].eq(target_market)][source_match_cols]
            .drop_duplicates()
            .copy()
        )
        fallback_sources: list[pd.DataFrame] = []
        if not target_keys.empty:
            direct = source.merge(target_keys, on=source_match_cols, how="inner")
            if not direct.empty:
                fallback_sources.append(direct)
        identity_matched = _identity_matched_fallback_source(
            source,
            target_predictions,
            target_market=target_market,
            identity_table=identity_table,
        )
        if not identity_matched.empty:
            fallback_sources.append(identity_matched)
        if not fallback_sources:
            continue
        fallback = pd.concat(fallback_sources, ignore_index=True, sort=False)
        fallback = fallback.drop_duplicates(
            subset=["game_id", "player_name_key", "player_name_prefix_key", "sportsbook", "line", "line_type"],
            keep="first",
        )
        fallback["market"] = target_market
        fallback["line_type"] = ANYTIME_TD_FALLBACK_LINE_TYPE
        fallback_keys = list(map(tuple, fallback[exact_key_cols].astype(str).to_numpy()))
        fallback = fallback[[key not in existing_exact for key in fallback_keys]].copy()
        if not fallback.empty:
            fallback_frames.append(fallback)

    if not fallback_frames:
        return odds
    expanded = pd.concat([odds, *fallback_frames], ignore_index=True, sort=False)
    return expanded.drop_duplicates(subset=exact_key_cols + ["line", "line_type"], keep="first")


def _attach_prop_odds(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    identity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if predictions.empty or odds.empty:
        return predictions
    out = predictions.copy()
    out["_prediction_row_id"] = np.arange(len(out))
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    out["player_name_prefix_key"] = out["player_name"].apply(_player_name_prefix_key)
    out["player_initial_last_key"] = out["player_name"].apply(_player_initial_last_key)
    odds_value_cols = [
        "line",
        "implied_probability",
        "sportsbook",
        "over_odds",
        "under_odds",
        "line_updated_at",
        "line_type",
        "line_balance_distance",
        "line_is_comparable",
    ]
    out = out.drop(columns=odds_value_cols, errors="ignore")
    odds = odds.copy()
    if "player_name_prefix_key" not in odds.columns:
        odds["player_name_prefix_key"] = pd.NA
    if "player_initial_last_key" not in odds.columns:
        odds["player_initial_last_key"] = pd.NA
    identity_table = _prepare_player_identity(identity)
    odds = _add_anytime_td_fallback_odds(odds, out, identity_table=identity_table)
    odds["_odds_row_id"] = np.arange(len(odds))

    strict_cols = ["game_id", "market", "player_name_key", "player_name_prefix_key"]
    legacy_cols = ["game_id", "market", "player_name_key"]
    initial_cols = ["game_id", "market", "player_initial_last_key"]
    matches: list[pd.DataFrame] = []
    matched_ids: set[int] = set()
    matched_odds_ids: set[int] = set()

    if not identity_table.empty and {"season", "week", "player_id"}.issubset(out.columns):
        identity_odds = odds.dropna(subset=strict_cols).copy()
        if not identity_odds.empty:
            parts = identity_odds["game_id"].apply(_game_identity_parts)
            identity_odds["_identity_season"] = [part[0] for part in parts]
            identity_odds["_identity_week"] = [part[1] for part in parts]
            identity_odds["_identity_candidate_team"] = [part[2] for part in parts]
            identity_odds = identity_odds.explode("_identity_candidate_team")
            identity_odds = identity_odds.dropna(
                subset=["_identity_season", "_identity_week", "_identity_candidate_team"]
            ).copy()
        identity_predictions = out.dropna(subset=["season", "week", "game_id", "market", "player_id"]).copy()
        if not identity_odds.empty and not identity_predictions.empty:
            line_identity = identity_odds.merge(
                identity_table,
                left_on=[
                    "_identity_season",
                    "_identity_week",
                    "_identity_candidate_team",
                    "player_name_key",
                    "player_name_prefix_key",
                ],
                right_on=[
                    "identity_season",
                    "identity_week",
                    "identity_team",
                    "alias_name_key",
                    "alias_prefix_key",
                ],
                how="inner",
            )
            if not line_identity.empty:
                identity_matches = identity_predictions.merge(
                    line_identity,
                    left_on=["season", "week", "game_id", "market", "player_id"],
                    right_on=["identity_season", "identity_week", "game_id", "market", "identity_player_id"],
                    how="inner",
                    suffixes=("", "_odds"),
                )
                if not identity_matches.empty:
                    matches.append(identity_matches)
                    matched_ids.update(identity_matches["_prediction_row_id"].dropna().astype(int).tolist())
                    matched_odds_ids.update(identity_matches["_odds_row_id"].dropna().astype(int).tolist())

    strict_odds = odds[~odds["_odds_row_id"].isin(matched_odds_ids)].dropna(subset=strict_cols)
    strict_predictions = out.dropna(subset=strict_cols)
    if not strict_odds.empty and not strict_predictions.empty:
        strict_matches = strict_predictions.merge(strict_odds, on=strict_cols, how="inner")
        if not strict_matches.empty:
            matches.append(strict_matches)
            matched_ids.update(strict_matches["_prediction_row_id"].dropna().astype(int).tolist())
            matched_odds_ids.update(strict_matches["_odds_row_id"].dropna().astype(int).tolist())

    legacy_odds = odds[~odds["_odds_row_id"].isin(matched_odds_ids)].dropna(subset=legacy_cols).copy()
    if not legacy_odds.empty:
        prefix_counts = (
            legacy_odds.groupby(legacy_cols, dropna=False)["player_name_prefix_key"]
            .nunique(dropna=True)
            .reset_index(name="_prefix_key_count")
        )
        legacy_odds = legacy_odds.merge(prefix_counts, on=legacy_cols, how="left")
        legacy_odds = legacy_odds[legacy_odds["_prefix_key_count"].fillna(0).le(1)]
        legacy_odds = legacy_odds.drop(columns=["player_name_prefix_key", "_prefix_key_count"], errors="ignore")
        fallback_predictions = out[~out["_prediction_row_id"].isin(matched_ids)]
        fallback_matches = fallback_predictions.merge(legacy_odds, on=legacy_cols, how="inner")
        if not fallback_matches.empty:
            matches.append(fallback_matches)
            matched_ids.update(fallback_matches["_prediction_row_id"].dropna().astype(int).tolist())
            matched_odds_ids.update(fallback_matches["_odds_row_id"].dropna().astype(int).tolist())

    initial_odds = odds[~odds["_odds_row_id"].isin(matched_odds_ids)].dropna(subset=initial_cols).copy()
    initial_predictions = out.dropna(subset=initial_cols).copy()
    if not initial_odds.empty and not initial_predictions.empty:
        prediction_key_counts = (
            initial_predictions.groupby(initial_cols, dropna=False)["_prediction_row_id"]
            .nunique(dropna=True)
            .reset_index(name="_prediction_key_count")
        )
        initial_predictions = initial_predictions.merge(prediction_key_counts, on=initial_cols, how="left")
        initial_predictions = initial_predictions[initial_predictions["_prediction_key_count"].fillna(0).eq(1)]
        initial_predictions = initial_predictions.drop(columns=["_prediction_key_count"], errors="ignore")
        initial_matches = initial_predictions.merge(initial_odds, on=initial_cols, how="inner")
        if not initial_matches.empty:
            matches.append(initial_matches)
            matched_ids.update(initial_matches["_prediction_row_id"].dropna().astype(int).tolist())
            matched_odds_ids.update(initial_matches["_odds_row_id"].dropna().astype(int).tolist())

    if matches:
        matched = pd.concat(matches, ignore_index=True, sort=False)
        dedupe_cols = [
            col
            for col in [
                "_prediction_row_id",
                "line",
                "implied_probability",
                "sportsbook",
                "over_odds",
                "under_odds",
                "line_updated_at",
                "line_type",
            ]
            if col in matched.columns
        ]
        if dedupe_cols:
            matched = matched.drop_duplicates(subset=dedupe_cols, keep="first")
        unmatched = out[~out["_prediction_row_id"].isin(matched_ids)].copy()
        for col in odds_value_cols:
            if col not in unmatched.columns:
                unmatched[col] = np.nan
        if unmatched.empty:
            out = matched
        else:
            for col in matched.columns:
                if col not in unmatched.columns:
                    unmatched[col] = np.nan
            for col in unmatched.columns:
                if col not in matched.columns:
                    matched[col] = np.nan
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
                    category=FutureWarning,
                )
                out = pd.concat([matched, unmatched[matched.columns]], ignore_index=True, sort=False)
    else:
        for col in odds_value_cols:
            if col not in out.columns:
                out[col] = np.nan
    if "bdl_team" in out.columns:
        same_team = out["bdl_team"].isna() | out["team"].eq(out["bdl_team"])
        for col in odds_value_cols:
            if col in out.columns:
                out.loc[~same_team, col] = np.nan
    line_stats = _line_stats_by_prediction(out)
    out = out.drop(columns=LINE_STAT_COLUMNS, errors="ignore")
    if not line_stats.empty:
        out = out.merge(line_stats, on="_prediction_row_id", how="left")
    if "line" in out.columns:
        if "line_type" in out.columns:
            out["_line_type_rank"] = np.where(
                out["line_type"].astype("string").str.lower().eq("canonical").fillna(False),
                0,
                1,
            )
        else:
            out["_line_type_rank"] = 0
        if "line_balance_distance" in out.columns:
            out["_line_balance_distance"] = pd.to_numeric(out["line_balance_distance"], errors="coerce").fillna(np.inf)
        else:
            out["_line_balance_distance"] = pd.Series(np.inf, index=out.index)
        out["_line_distance"] = (
            pd.to_numeric(out["projection"], errors="coerce") - pd.to_numeric(out["line"], errors="coerce")
        ).abs()
        out["_line_distance"] = out["_line_distance"].fillna(np.inf)
        out["_line_price_rank"] = _line_selection_price_rank(out)
        out["_sportsbook_rank"] = _sportsbook_selection_rank(out["sportsbook"])
        out = out.sort_values(
            [
                "_prediction_row_id",
                "_line_type_rank",
                "_line_balance_distance",
                "_line_distance",
                "_line_price_rank",
                "_sportsbook_rank",
                "sportsbook",
            ],
            kind="stable",
        )
        out = out.drop_duplicates("_prediction_row_id", keep="first")
    out["edge"] = pd.to_numeric(out["projection"], errors="coerce") - pd.to_numeric(out["line"], errors="coerce")
    return out.drop(
        columns=[
            "player_name_key",
            "player_name_prefix_key",
            "player_initial_last_key",
            "bdl_team",
            "_prediction_row_id",
            "_odds_row_id",
            "_identity_season",
            "_identity_week",
            "_identity_candidate_team",
            "identity_season",
            "identity_week",
            "identity_team",
            "identity_player_id",
            "alias_name_key",
            "alias_prefix_key",
            "_line_distance",
            "_line_price_rank",
            "_line_balance_distance",
            "_line_type_rank",
            "_sportsbook_rank",
            "line_balance_distance",
            "line_is_comparable",
        ],
        errors="ignore",
    )


def _prepare_nflverse_injuries(injuries: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    required = {"season", "week", "gsis_id"}
    if injuries.empty or not required.issubset(injuries.columns):
        return pd.DataFrame(columns=["player_id"] + INJURY_COLUMNS)

    out = injuries.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].eq(int(season)) & out["week"].le(int(week))]
    if out.empty:
        return pd.DataFrame(columns=["player_id"] + INJURY_COLUMNS)

    out["player_id"] = _as_text(out["gsis_id"])
    out["team"] = out["team"].apply(_clean_team) if "team" in out.columns else pd.NA
    out["player_name_key"] = out["full_name"].apply(_player_name_key) if "full_name" in out.columns else pd.NA
    out["injury_status"] = out.get("report_status", pd.Series(pd.NA, index=out.index)).apply(_clean_injury_text)
    out["injury_practice_status"] = out.get("practice_status", pd.Series(pd.NA, index=out.index)).apply(_clean_injury_text)
    primary = out.get("report_primary_injury", pd.Series(pd.NA, index=out.index)).apply(_clean_injury_text)
    practice = out.get("practice_primary_injury", pd.Series(pd.NA, index=out.index)).apply(_clean_injury_text)
    out["injury_body_part"] = primary.where(primary.notna(), practice)
    out["injury_report_week"] = out["week"].astype("Int64")
    out["injury_source"] = "nflverse"
    out["injury_last_updated"] = out.get("date_modified", pd.Series(pd.NA, index=out.index))
    out["_severity"] = out["injury_status"].apply(_injury_severity)
    out = out.sort_values(["player_id", "injury_report_week", "_severity"], kind="stable")
    out = out.drop_duplicates("player_id", keep="last")
    out["injury_reported"] = out.apply(_injury_reported, axis=1)
    out["injury_exclusion_flag"] = out["injury_status"].apply(_injury_exclusion_flag)
    out["prop_quality_flag"] = out.apply(_injury_quality_flag, axis=1)
    return out[
        [
            "player_id",
            "player_name_key",
            "team",
            *INJURY_COLUMNS,
        ]
    ]


def _prepare_bdl_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries.empty or not {"player.first_name", "player.last_name"}.issubset(injuries.columns):
        return pd.DataFrame(columns=["player_name_key", "team"] + INJURY_COLUMNS)

    out = injuries.copy()
    out["player_name"] = [_full_name(first, last) for first, last in zip(out["player.first_name"], out["player.last_name"])]
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    team_col = "player.team.abbreviation" if "player.team.abbreviation" in out.columns else None
    out["team"] = out[team_col].apply(_clean_team) if team_col else pd.NA
    out["injury_status"] = out.get("status", pd.Series(pd.NA, index=out.index)).apply(_clean_injury_text)
    out["injury_practice_status"] = pd.NA
    out["injury_body_part"] = pd.NA
    out["injury_report_week"] = pd.NA
    out["injury_source"] = "balldontlie"
    out["injury_last_updated"] = out.get("date", pd.Series(pd.NA, index=out.index))
    out["_severity"] = out["injury_status"].apply(_injury_severity)
    out["_date"] = pd.to_datetime(out["injury_last_updated"], errors="coerce", utc=True)
    out = out.dropna(subset=["player_name_key", "team"])
    if out.empty:
        return pd.DataFrame(columns=["player_name_key", "team"] + INJURY_COLUMNS)
    out = out.sort_values(["player_name_key", "team", "_date", "_severity"], kind="stable")
    out = out.drop_duplicates(["player_name_key", "team"], keep="last")
    out["injury_reported"] = out.apply(_injury_reported, axis=1)
    out["injury_exclusion_flag"] = out["injury_status"].apply(_injury_exclusion_flag)
    out["prop_quality_flag"] = out.apply(_injury_quality_flag, axis=1)
    return out[["player_name_key", "team", *INJURY_COLUMNS]]


def load_player_injury_status(
    *,
    season: int,
    week: int,
    injuries_path: Path = DEFAULT_INJURIES_PATH,
    bdl_injuries_path: Path = DEFAULT_BDL_INJURIES_PATH,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    nfl_injuries = _read_frame_if_exists(injuries_path)
    if not nfl_injuries.empty:
        prepared = _prepare_nflverse_injuries(nfl_injuries, season=season, week=week)
        if not prepared.empty:
            frames.append(prepared)

    bdl_injuries = _read_frame_if_exists(bdl_injuries_path)
    if not bdl_injuries.empty:
        prepared = _prepare_bdl_injuries(bdl_injuries)
        if not prepared.empty:
            frames.append(prepared)

    if not frames:
        return pd.DataFrame(columns=["player_id", "player_name_key", "team", *INJURY_COLUMNS])
    combined = pd.concat([frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True, sort=False)
    return combined.reindex(columns=["player_id", "player_name_key", "team", *INJURY_COLUMNS])


def _default_injury_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    defaults: dict[str, object] = {
        "injury_status": pd.NA,
        "injury_practice_status": pd.NA,
        "injury_body_part": pd.NA,
        "injury_report_week": pd.NA,
        "injury_source": pd.NA,
        "injury_last_updated": pd.NA,
        "injury_reported": False,
        "injury_exclusion_flag": False,
        "prop_quality_flag": "ok",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        else:
            if value is not pd.NA:
                out[col] = out[col].where(out[col].notna(), value)
    return out


def _attach_injury_status(predictions: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return _default_injury_values(predictions)
    if injuries.empty:
        return _default_injury_values(predictions)

    out = predictions.copy()
    injuries = injuries.copy()
    out = out.drop(columns=INJURY_COLUMNS, errors="ignore")
    for col in ["player_id", "player_name_key", "team"]:
        if col not in injuries.columns:
            injuries[col] = pd.NA
    out["_player_name_key"] = out["player_name"].apply(_player_name_key)
    out["_team_key"] = out["team"].apply(_clean_team)

    id_lookup = injuries.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="last")
    if not id_lookup.empty:
        merge_cols = ["player_id", *INJURY_COLUMNS]
        out = out.merge(id_lookup[merge_cols], on="player_id", how="left")

    name_lookup = injuries.dropna(subset=["player_name_key", "team"]).drop_duplicates(
        ["player_name_key", "team"],
        keep="last",
    )
    if not name_lookup.empty:
        fallback = out[["_player_name_key", "_team_key"]].merge(
            name_lookup.rename(columns={"player_name_key": "_player_name_key", "team": "_team_key"})[
                ["_player_name_key", "_team_key", *INJURY_COLUMNS]
            ],
            on=["_player_name_key", "_team_key"],
            how="left",
            suffixes=("", "_fallback"),
        )
        for col in INJURY_COLUMNS:
            fallback_col = f"{col}_fallback" if f"{col}_fallback" in fallback.columns else col
            if col not in out.columns:
                out[col] = fallback[fallback_col]
            else:
                out[col] = out[col].where(out[col].notna(), fallback[fallback_col])

    out = _default_injury_values(out)
    out["injury_reported"] = out["injury_reported"].astype("boolean").fillna(False).astype(bool)
    out["injury_exclusion_flag"] = out["injury_exclusion_flag"].astype("boolean").fillna(False).astype(bool)
    out["prop_quality_flag"] = out.apply(_injury_quality_flag, axis=1)
    return out.drop(columns=["_player_name_key", "_team_key"], errors="ignore")


def _score_market(
    features: pd.DataFrame,
    *,
    bundle: dict[str, object],
    generated_at: str,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    feature_cols = list(bundle.get("feature_cols") or [])
    if not feature_cols:
        return pd.DataFrame()
    frame = features.copy()
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = np.nan
    regressor = bundle["regressor"]
    projection = np.clip(regressor.predict(frame[feature_cols]), 0, None)
    classifier = bundle.get("classifier")
    if classifier is not None:
        prob_over = classifier.predict_proba(frame[feature_cols])[:, 1]
    elif bundle.get("target_type") == "count_event":
        prob_over = 1.0 - np.exp(-projection)
    else:
        prob_over = np.full(len(frame), np.nan)

    out = frame.copy()
    out["model_projection"] = projection
    out["projection"] = projection
    out["prob_over"] = prob_over
    out["prob_over_raw"] = np.nan
    out["prob_edge"] = np.nan
    out["prob_over_method"] = np.where(pd.notna(out["prob_over"]), "over_zero_model", pd.NA)
    out["prob_over_sample_size"] = np.nan
    out["over_break_even_probability"] = np.nan
    out["under_break_even_probability"] = np.nan
    out["over_ev"] = np.nan
    out["under_ev"] = np.nan
    out["edge_side"] = pd.NA
    out["edge_odds"] = np.nan
    out["edge_break_even_probability"] = np.nan
    out["edge_ev"] = np.nan
    out["edge_probability_edge"] = np.nan
    out["edge_recommendation_flag"] = False
    out["edge_recommendation_tier"] = "research_only"
    out["edge_policy_scope"] = pd.NA
    out["edge_policy_strategy"] = pd.NA
    out["edge_policy_threshold"] = np.nan
    out["edge_policy_roi"] = np.nan
    out["edge_policy_n_bets"] = np.nan
    out["feature_count"] = len(feature_cols)
    out["model_name"] = f"{bundle.get('model_type', 'model')}:{bundle.get('market', out['market'].iloc[0])}"
    out["sample_size"] = out["player_games_prior"].fillna(0).astype(int)
    out["confidence_tier"] = out["sample_size"].apply(_confidence_tier)
    out["line"] = np.nan
    out["line_type"] = pd.NA
    out["implied_probability"] = np.nan
    out["edge"] = np.nan
    out["sportsbook"] = pd.NA
    out["over_odds"] = np.nan
    out["under_odds"] = np.nan
    out["line_updated_at"] = pd.NA
    for col in LINE_STAT_COLUMNS:
        out[col] = False if col == "line_variance_flag" else np.nan
    out["injury_status"] = pd.NA
    out["injury_practice_status"] = pd.NA
    out["injury_body_part"] = pd.NA
    out["injury_report_week"] = pd.NA
    out["injury_source"] = pd.NA
    out["injury_last_updated"] = pd.NA
    out["injury_reported"] = False
    out["injury_exclusion_flag"] = False
    out["prop_quality_flag"] = "ok"
    for col in AVAILABILITY_COLUMNS:
        if col == "roster_validation_flag":
            out[col] = "unknown"
        elif col == "active_current_roster":
            out[col] = pd.NA
        elif col == "depth_chart_rank":
            out[col] = np.nan
        else:
            out[col] = pd.NA
    out["bettable_flag"] = False
    out["generated_at"] = generated_at
    return out


def predict_player_props(
    *,
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    novelty_labels: pd.DataFrame | None = None,
    novelty_candidates: pd.DataFrame | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
    odds: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    player_identity: pd.DataFrame | None = None,
    over_probability_history: pd.DataFrame | None = None,
    edge_policy: pd.DataFrame | None = None,
    season: int | None = None,
    week: int | None = None,
    generated_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    novelty_labels = novelty_labels if novelty_labels is not None else pd.DataFrame()
    novelty_candidates = novelty_candidates if novelty_candidates is not None else pd.DataFrame()
    if candidates.empty and novelty_candidates.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()

    if season is None:
        season_source = candidates if not candidates.empty else novelty_candidates
        season = int(pd.to_numeric(season_source["season"], errors="coerce").dropna().max())
    if week is None:
        week_source = candidates if not candidates.empty else novelty_candidates
        week = int(
            pd.to_numeric(week_source[week_source["season"].eq(season)]["week"], errors="coerce").dropna().max()
        )

    if not candidates.empty:
        candidates = candidates[(candidates["season"].eq(int(season))) & (candidates["week"].eq(int(week)))].copy()
    if not novelty_candidates.empty:
        novelty_candidates = novelty_candidates[
            (novelty_candidates["season"].eq(int(season))) & (novelty_candidates["week"].eq(int(week)))
        ].copy()
    candidates = _add_line_offered_def_sack_candidates(
        candidates,
        labels=labels,
        odds=odds,
        player_identity=player_identity,
        availability=availability,
        season=int(season),
        week=int(week),
    )

    scored: list[pd.DataFrame] = []
    feature_sets: list[tuple[str, pd.DataFrame]] = []
    if not candidates.empty:
        offense_features, defense_features = build_player_prop_prediction_features(
            labels,
            candidates,
            exclude_from_season=int(season),
            exclude_from_week=int(week),
        )
        feature_sets.extend([("offense", offense_features), ("defense", defense_features)])
    if not novelty_candidates.empty and not novelty_labels.empty:
        novelty_features = build_player_prop_novelty_prediction_features(
            novelty_labels,
            novelty_candidates,
            exclude_from_season=int(season),
            exclude_from_week=int(week),
        )
        feature_sets.append(("novelty", novelty_features))

    for family, feature_frame in feature_sets:
        for market, market_frame in feature_frame.groupby("market", sort=True):
            bundle = _load_model_bundle(models_dir, family, str(market))
            if bundle is None:
                continue
            scored_market = _score_market(market_frame, bundle=bundle, generated_at=generated_at)
            if not scored_market.empty:
                scored.append(scored_market)

    if not scored:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()

    predictions = pd.concat(scored, ignore_index=True, sort=False)
    predictions = _attach_prop_odds(
        predictions,
        odds if odds is not None else pd.DataFrame(),
        identity=player_identity,
    )
    if over_probability_history is not None and not over_probability_history.empty:
        predictions = attach_current_over_probabilities(
            predictions,
            over_probability_history,
            season=int(season),
            week=int(week),
        )
    predictions = _add_prop_betting_edges(predictions)
    predictions = _attach_injury_status(predictions, injuries if injuries is not None else pd.DataFrame())
    predictions = attach_player_availability(predictions, availability if availability is not None else pd.DataFrame())
    roster_flag = predictions.get("roster_validation_flag", pd.Series("unknown", index=predictions.index))
    roster_exclusion = roster_flag.isin({"team_mismatch", "inactive_roster", "practice_squad"})
    predictions.loc[roster_flag.eq("team_mismatch"), "prop_quality_flag"] = "roster_mismatch"
    predictions.loc[roster_flag.isin({"inactive_roster", "practice_squad"}), "prop_quality_flag"] = "roster_exclusion"
    line_type = predictions.get("line_type", pd.Series(pd.NA, index=predictions.index)).astype("string").str.lower()
    fallback_line = line_type.isin(NON_BETTABLE_LINE_TYPES).fillna(False)
    predictions.loc[fallback_line & predictions["prop_quality_flag"].eq("ok"), "prop_quality_flag"] = "fallback_line"
    predictions["bettable_flag"] = (
        predictions["line"].notna()
        & ~fallback_line
        & ~predictions["injury_exclusion_flag"].fillna(False).astype(bool)
        & ~roster_exclusion
    )
    predictions = _attach_edge_recommendations(predictions, edge_policy)
    novelty_mask = predictions["market"].isin(NOVELTY_OUTPUT_MARKETS)
    predictions.loc[novelty_mask, "bettable_flag"] = False
    predictions.loc[novelty_mask, "edge_recommendation_flag"] = False
    predictions.loc[novelty_mask, "edge_recommendation_tier"] = "research_only"
    for col in ["games_sampled", "player_rank", "team_side", "baseline_projection"]:
        if col not in predictions.columns:
            predictions[col] = np.nan
    predictions = predictions.sort_values(["game_id", "team", "market", "projection"], ascending=[True, True, True, False])
    predictions = predictions[[col for col in OUTPUT_COLUMNS if col in predictions.columns]]

    qb = predictions[predictions["market"].isin(QB_MARKETS)].copy()
    offense = predictions[predictions["market"].isin(OFFENSE_OUTPUT_MARKETS)].copy()
    defense = predictions[predictions["market"].eq("def_sacks")].copy()
    novelty = predictions[predictions["market"].isin(NOVELTY_OUTPUT_MARKETS)].copy()
    return qb, offense, defense, novelty


def _write_output(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for col in OUTPUT_COLUMNS:
        if col not in output.columns:
            output[col] = np.nan
    output[OUTPUT_COLUMNS].to_csv(path, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate weekly trained-model player prop prediction CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--novelty-labels", type=Path, default=DEFAULT_NOVELTY_LABELS_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--qb-source", type=Path, default=None)
    parser.add_argument("--offense-source", type=Path, default=None)
    parser.add_argument("--defense-source", type=Path, default=None)
    parser.add_argument("--odds-source", type=Path, default=None)
    parser.add_argument("--odds-vendor", nargs="+", default=list(DEFAULT_ODDS_VENDORS))
    parser.add_argument("--over-probability-history", type=Path, default=DEFAULT_OVER_PROBABILITY_HISTORY_PATH)
    parser.add_argument("--edge-policy", type=Path, default=DEFAULT_EDGE_POLICY_PATH)
    parser.add_argument("--availability", type=Path, default=DEFAULT_AVAILABILITY_PATH)
    parser.add_argument("--player-identity", type=Path, default=DEFAULT_PLAYER_IDENTITY_PATH)
    parser.add_argument("--injuries", type=Path, default=DEFAULT_INJURIES_PATH)
    parser.add_argument("--bdl-injuries", type=Path, default=DEFAULT_BDL_INJURIES_PATH)
    parser.add_argument("--rosters", type=Path, default=DEFAULT_ROSTERS_PATH)
    parser.add_argument("--qb-output", type=Path, default=DEFAULT_QB_OUTPUT)
    parser.add_argument("--offense-output", type=Path, default=DEFAULT_OFFENSE_OUTPUT)
    parser.add_argument("--defense-output", type=Path, default=DEFAULT_DEFENSE_OUTPUT)
    parser.add_argument("--novelty-output", type=Path, default=DEFAULT_NOVELTY_OUTPUT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    labels = read_df(args.labels) if args.labels.exists() else pd.DataFrame()
    novelty_labels = read_df(args.novelty_labels) if args.novelty_labels.exists() else pd.DataFrame()
    prediction_dir = args.prediction_dir
    qb_source = args.qb_source or prediction_dir / "predictions_players_qb.csv"
    offense_source = args.offense_source or prediction_dir / "predictions_players_offense.csv"
    defense_source = args.defense_source or prediction_dir / "predictions_players_defense.csv"
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_source,
        offense_path=offense_source,
        defense_path=defense_source,
        rosters_path=args.rosters,
    )
    identity_labels = novelty_labels if not novelty_labels.empty else labels
    novelty_candidates = build_novelty_prediction_candidates(
        labels=identity_labels,
        qb_path=qb_source,
        offense_path=offense_source,
        rosters_path=args.rosters,
    )
    odds = load_player_prop_odds(season=args.season, week=args.week, odds_path=args.odds_source, vendor=args.odds_vendor)
    over_probability_history = _read_frame_if_exists(args.over_probability_history)
    injuries = load_player_injury_status(
        season=args.season,
        week=args.week,
        injuries_path=args.injuries,
        bdl_injuries_path=args.bdl_injuries,
    )
    availability = _read_frame_if_exists(args.availability)
    player_identity = _read_frame_if_exists(args.player_identity)
    edge_policy = _read_frame_if_exists(args.edge_policy)
    qb, offense, defense, novelty = predict_player_props(
        labels=labels,
        candidates=candidates,
        novelty_labels=novelty_labels,
        novelty_candidates=novelty_candidates,
        models_dir=args.models_dir,
        odds=odds,
        injuries=injuries,
        availability=availability,
        player_identity=player_identity,
        over_probability_history=over_probability_history,
        edge_policy=edge_policy,
        season=args.season,
        week=args.week,
    )
    _write_output(qb, args.qb_output)
    _write_output(offense, args.offense_output)
    _write_output(defense, args.defense_output)
    _write_output(novelty, args.novelty_output)

    if args.debug:
        print("[player_props.predict] row counts:")
        print(f"  qb: {len(qb)}")
        print(f"  offense: {len(offense)}")
        print(f"  defense: {len(defense)}")
        print(f"  novelty: {len(novelty)}")
    print(f"[player_props.predict] wrote QB rows={len(qb)} -> {args.qb_output.resolve()}")
    print(f"[player_props.predict] wrote offense rows={len(offense)} -> {args.offense_output.resolve()}")
    print(f"[player_props.predict] wrote defense rows={len(defense)} -> {args.defense_output.resolve()}")
    print(f"[player_props.predict] wrote novelty rows={len(novelty)} -> {args.novelty_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
