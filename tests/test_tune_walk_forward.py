from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models import tune_walk_forward as wf
from tools.run_pipeline import is_walk_forward_tuning_config


def _frame(seasons=(2018, 2019, 2020, 2021, 2022), per_season: int = 60, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = per_season * len(seasons)
    home = rng.normal(size=n)
    away = rng.normal(size=n)
    margin = 6 * (home - away) + rng.normal(scale=6, size=n)
    margin = np.where(np.round(margin) == 0, 1.0, np.round(margin))
    return pd.DataFrame(
        {
            "season": np.repeat(seasons, per_season),
            "week": np.tile(np.arange(1, per_season + 1) % 18 + 1, len(seasons)),
            "pass_epa_per_db_rolling3_home": home,
            "pass_epa_per_db_rolling3_away": away,
            "sched_implied_prob_home": 1 / (1 + np.exp(-(home - away))) + 0.02,
            "sched_implied_prob_away": 1 - 1 / (1 + np.exp(-(home - away))) + 0.02,
            "home_margin": margin,
            "home_win": (margin > 0).astype(int),
        }
    )


def test_default_validation_seasons_skip_first_and_latest():
    assert wf.default_validation_seasons([2017, 2018, 2019, 2020, 2021, 2022], 3) == [2019, 2020, 2021]
    assert wf.default_validation_seasons([2017, 2018, 2019], 4) == [2018]
    with pytest.raises(ValueError):
        wf.default_validation_seasons([2020, 2021], 2)


def test_walk_forward_folds_never_train_on_validation_or_later_seasons():
    data = wf.prepare_walk_forward_data(_frame(), target="win_prob", validation_seasons=[2020, 2021])

    assert data.validation_seasons == [2020, 2021]
    for (train_idx, val_idx), season in zip(data.folds, data.validation_seasons):
        assert (data.seasons[train_idx] < season).all()
        assert (data.seasons[val_idx] == season).all()
    assert data.market_prob is not None


def test_score_win_params_reports_model_and_market_metrics():
    data = wf.prepare_walk_forward_data(_frame(), target="win_prob", validation_seasons=[2021, 2022])
    params = {"model_type": "hist", "hgb_max_iter": 30, "hgb_max_depth": 2}

    scores = wf.score_win_params(params, data)

    assert scores["n_games"] == 120
    assert 0.0 < scores["brier"] < 0.3
    assert {"market_brier", "market_auc", "model_brier_same_rows"}.issubset(scores)


def test_score_spread_params_runs_on_walk_forward_folds():
    data = wf.prepare_walk_forward_data(_frame(), target="spread", validation_seasons=[2022])
    scores = wf.score_spread_params({"model_type": "hist", "hgb_max_iter": 30}, data)

    assert scores["n_games"] == 60
    assert scores["mae"] > 0


def test_feature_fingerprint_is_order_insensitive():
    assert wf.feature_fingerprint(["b", "a"]) == wf.feature_fingerprint(["a", "b"])
    assert wf.feature_fingerprint(["a"]) != wf.feature_fingerprint(["a", "b"])


def test_pipeline_only_applies_walk_forward_configs(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"best_value": 0.35, "best_params": {"model_type": "xgb"}}), encoding="utf-8")
    current = tmp_path / "current.json"
    current.write_text(
        json.dumps({"best_params": {"model_type": "xgb"}, "tuning_scheme": wf.TUNING_SCHEME}), encoding="utf-8"
    )

    assert not is_walk_forward_tuning_config(legacy)
    assert is_walk_forward_tuning_config(current)
    assert not is_walk_forward_tuning_config(tmp_path / "missing.json")


def test_market_anchor_tuning_scores_residual_model():
    frame = _frame()
    frame["sched_implied_prob_diff"] = frame["sched_implied_prob_home"] - frame["sched_implied_prob_away"]
    data = wf.prepare_walk_forward_data(frame, target="win_prob", validation_seasons=[2022], market_anchor=True)

    assert data.market_anchor
    assert data.feature_names[-2:] == ["sched_implied_prob_home", "sched_implied_prob_away"]
    scores = wf.score_win_params({"model_type": "xgb", "xgb_n_estimators": 30}, data)
    assert 0.0 < scores["brier"] < 0.3


def test_market_anchor_spread_tuning():
    frame = _frame()
    frame["sched_spread_close_home"] = 6 * (frame["pass_epa_per_db_rolling3_home"] - frame["pass_epa_per_db_rolling3_away"])
    data = wf.prepare_walk_forward_data(frame, target="spread", validation_seasons=[2022], market_anchor=True)

    assert data.feature_names[-1] == "sched_spread_close_home"
    scores = wf.score_spread_params({"model_type": "hist", "hgb_max_iter": 30}, data)
    assert 0 < scores["mae"] < 10
