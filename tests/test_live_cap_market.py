"""Live predictions relax the 0.85/0.15 cap from the feature rows' market, as walk-forward does."""
import numpy as np
import pandas as pd

from src.predict.predict_upcoming import _market_cap_input
from src.predict.utils import apply_probability_caps


def _frames():
    preds = pd.DataFrame({"season": [2026] * 3, "week": [11] * 3, "home_team": ["BUF", "KC", "CHI"],
                          "away_team": ["MIA", "ARI", "NO"], "temp": [30.0, 40.0, 35.0]})
    feats = preds.assign(sched_implied_prob_home=[0.948, 0.936, 0.55], home_moneyline=[-1800.0, -1450.0, -120.0])
    return preds, feats


def test_market_comes_from_feature_rows_not_preds():
    preds, feats = _frames()
    caps = _market_cap_input(preds, feats)
    assert caps.tolist() == [0.948, 0.936, 0.55]
    assert _market_cap_input(preds, preds) is None  # preds alone has no market (the old bug)


def test_heavy_favourites_follow_the_market_like_walk_forward():
    preds, feats = _frames()
    calibrated = np.array([0.903, 0.906, 0.60])  # Week 11 MIA-BUF, ARI-KC, and an ordinary game
    live = apply_probability_caps(calibrated, _market_cap_input(preds, feats))
    walk_forward = apply_probability_caps(calibrated, pd.to_numeric(feats["sched_implied_prob_home"]))
    assert np.allclose(live, walk_forward)
    assert np.allclose(live, [0.903, 0.906, 0.60])  # was [0.85, 0.85, 0.60]
    assert np.allclose(apply_probability_caps(calibrated, None), [0.85, 0.85, 0.60])


def test_moneyline_fallback_and_index_alignment():
    preds, feats = _frames()
    preds.index = feats.index = [10, 11, 12]
    caps = _market_cap_input(preds, feats.drop(columns="sched_implied_prob_home"))
    assert list(caps.index) == [10, 11, 12]
    assert caps.iloc[0] > 0.94 and caps.iloc[2] < 0.6
