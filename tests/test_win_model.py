from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import train, win_model


def _synthetic_matchups(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength_home = rng.normal(size=n)
    strength_away = rng.normal(size=n)
    margin = 6 * (strength_home - strength_away) + rng.normal(scale=6, size=n)
    margin = np.where(np.round(margin) == 0, 1.0, np.round(margin))
    return pd.DataFrame(
        {
            "season": np.repeat([2021, 2022, 2023], n // 3),
            "week": np.tile(np.arange(1, n // 3 + 1) % 18 + 1, 3),
            "game_id": [f"g{i}" for i in range(n)],
            "home_team": "AAA",
            "away_team": "BBB",
            "pass_epa_per_db_rolling3_home": strength_home,
            "pass_epa_per_db_rolling3_away": strength_away,
            "sched_rest_days_home": rng.integers(4, 14, size=n).astype(float),
            "sched_rest_days_away": rng.integers(4, 14, size=n).astype(float),
            "home_margin": margin,
            "home_win": (margin > 0).astype(int),
        }
    )


_FAST_PARAMS = {"model_type": "hist", "hgb_max_iter": 30}


def test_production_win_model_uses_all_rows_and_matches_walk_forward_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "EVAL_DIR", tmp_path)
    df = _synthetic_matchups()

    train.train_win_prob(df, skip_logit=True, cv_folds=1, param_config=_FAST_PARAMS)
    shipped = joblib.load(tmp_path / "winprob_gb.pkl")

    assert shipped["training_rows"] == len(df)
    assert shipped["logistic_model"] is None

    feat_df = train.make_feature_diffs(df)
    walk_forward = win_model.fit_win_bundle(feat_df, shipped["features"], param_config=_FAST_PARAMS, skip_logit=True)
    shipped_proba = shipped["model"].predict_proba(
        win_model.prepare_matrix(feat_df, shipped["features"]).values
    )[:, 1]
    assert np.allclose(win_model.predict_win_bundle(walk_forward, feat_df), shipped_proba)


def test_calibrated_win_model_keeps_holdout_split(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "EVAL_DIR", tmp_path)
    df = _synthetic_matchups()

    train.train_win_prob(df, calibrate=True, skip_logit=True, cv_folds=1, param_config=_FAST_PARAMS)
    shipped = joblib.load(tmp_path / "winprob_gb.pkl")

    assert shipped["calibrated"] is True
    assert shipped["training_rows"] < len(df)


def test_production_spread_model_uses_all_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "EVAL_DIR", tmp_path)
    df = _synthetic_matchups()

    train.train_spread(df, cv_folds=2, param_config=_FAST_PARAMS)
    shipped = joblib.load(tmp_path / "spread_gb.pkl")

    assert shipped["training_rows"] == len(df)
    assert shipped["model_type"] == "hgb"
    assert set(shipped["quantile_models"]) == {0.2, 0.8}


def test_model_kind_resolution_is_shared():
    assert win_model.resolve_win_model_kind({"model_type": "hist"}) == "hgb"
    assert win_model.resolve_spread_model_kind({"model_type": "huber"}) == "hgb"
    assert win_model.resolve_spread_model_kind({"model_type": "gbr"}) == "gbr"


def _anchored_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.normal(size=n)
    edge = rng.normal(size=n)  # information the market does not price
    market = 1.0 / (1.0 + np.exp(-0.8 * strength))
    p_true = 1.0 / (1.0 + np.exp(-(0.8 * strength + 0.6 * edge)))
    y = (rng.uniform(size=n) < p_true).astype(int)
    return pd.DataFrame(
        {
            "edge_diff": edge,
            "sched_implied_prob_diff": 2 * market - 1,
            "sched_implied_prob_home": market * 1.04,
            "sched_implied_prob_away": (1 - market) * 1.04,
            "home_win": y,
        }
    )


_ANCHOR_PARAMS = {"model_type": "xgb", "market_anchor": True, "xgb_n_estimators": 60, "xgb_learning_rate": 0.05}


def test_current_market_columns_are_identified():
    assert win_model.is_current_market_column("sched_implied_prob_home")
    assert win_model.is_current_market_column("sched_spread_close_diff")
    assert win_model.is_current_market_column("sched_moneyline_diff")
    assert not win_model.is_current_market_column("sched_implied_prob_rolling3_diff")
    assert not win_model.is_current_market_column("sched_spread_close_prev_diff")


def test_market_anchor_appends_anchor_columns():
    cols = win_model.win_feature_columns(["edge_diff"], _ANCHOR_PARAMS, _anchored_frame().columns)
    assert cols == ["edge_diff", "sched_implied_prob_home", "sched_implied_prob_away"]
    assert win_model.win_feature_columns(["edge_diff"], {"model_type": "xgb"}, ["edge_diff"]) == ["edge_diff"]
    with pytest.raises(ValueError):
        win_model.win_feature_columns(["edge_diff"], _ANCHOR_PARAMS, ["edge_diff"])


def test_anchored_model_starts_from_market_and_ignores_current_lines():
    df = _anchored_frame()
    cols = ["edge_diff", "sched_implied_prob_diff", "sched_implied_prob_home", "sched_implied_prob_away"]
    zero_trees = {**_ANCHOR_PARAMS, "xgb_n_estimators": 1, "xgb_learning_rate": 1e-9}
    model, label = win_model.fit_win_classifier(
        df[cols].to_numpy(), df["home_win"].to_numpy(), zero_trees, feature_names=cols
    )
    assert label.startswith("xgb_anchored")
    assert model.residual_features_ == ["edge_diff"]
    market = df["sched_implied_prob_home"] / (df["sched_implied_prob_home"] + df["sched_implied_prob_away"])
    np.testing.assert_allclose(model.predict_proba(df[cols].to_numpy())[:, 1], market, atol=1e-3)


def test_anchored_model_learns_residual_and_handles_missing_lines():
    df = _anchored_frame()
    bundle = win_model.fit_win_bundle(df.iloc[:500], ["edge_diff", "sched_implied_prob_diff"], param_config=_ANCHOR_PARAMS)
    assert bundle["features"][-2:] == list(win_model.MARKET_ANCHOR_COLUMNS)
    holdout = df.iloc[500:].copy()
    p = win_model.predict_win_bundle(bundle, holdout)
    market = holdout["sched_implied_prob_home"] / (holdout["sched_implied_prob_home"] + holdout["sched_implied_prob_away"])
    corr_with_edge = np.corrcoef(np.log(p / (1 - p)) - np.log(market / (1 - market)), holdout["edge_diff"])[0, 1]
    assert corr_with_edge > 0.5

    holdout.loc[holdout.index[:5], ["sched_implied_prob_home", "sched_implied_prob_away"]] = np.nan
    p_missing = win_model.predict_win_bundle(bundle, holdout)
    assert np.isfinite(p_missing).all()


def test_anchored_requires_xgb_and_feature_names():
    df = _anchored_frame(n=100)
    X = df[["edge_diff", "sched_implied_prob_home", "sched_implied_prob_away"]].to_numpy()
    with pytest.raises(ValueError):
        win_model.fit_win_classifier(X, df["home_win"].to_numpy(), _ANCHOR_PARAMS)
    with pytest.raises(ValueError):
        win_model.fit_win_classifier(
            X, df["home_win"].to_numpy(), {"model_type": "hist", "market_anchor": True},
            feature_names=["edge_diff", "sched_implied_prob_home", "sched_implied_prob_away"],
        )


def test_anchored_spread_predicts_line_plus_residual():
    rng = np.random.default_rng(5)
    n = 500
    line = rng.normal(scale=6, size=n)
    edge = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "edge_diff": edge,
            "sched_spread_close_diff": 2 * line,
            "sched_spread_close_home": line,
            "home_margin": line + 3 * edge + rng.normal(scale=5, size=n),
        }
    )
    params = {"model_type": "hist", "market_anchor": True, "hgb_max_iter": 50}
    bundle = win_model.fit_spread_bundle(df.iloc[:400], ["edge_diff", "sched_spread_close_diff"], param_config=params)
    assert bundle["features"][-1] == "sched_spread_close_home"
    assert bundle["model"].residual_features_ == ["edge_diff"]
    pred = win_model.predict_spread_bundle(bundle, df.iloc[400:])
    resid = pred - df["sched_spread_close_home"].iloc[400:].to_numpy()
    assert np.corrcoef(resid, df["edge_diff"].iloc[400:])[0, 1] > 0.8
