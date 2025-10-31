from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def moneyline_to_prob(values: Sequence[float] | pd.Series | np.ndarray | None) -> np.ndarray:
    """
    Convert American moneyline odds to implied win probability.
    Returns NaN for missing or malformed inputs.
    """
    if values is None:
        return np.array([], dtype=float)
    series = pd.to_numeric(values, errors="coerce")
    if isinstance(series, pd.Series):
        odds = series.to_numpy(dtype=float, copy=False)
    else:
        odds = np.asarray(series, dtype=float)
    probs = np.full_like(odds, np.nan, dtype=float)
    if probs.size == 0:
        return probs
    pos_mask = odds > 0
    neg_mask = odds < 0
    probs[pos_mask] = 100.0 / (odds[pos_mask] + 100.0)
    probs[neg_mask] = (-odds[neg_mask]) / ((-odds[neg_mask]) + 100.0)
    return probs


def apply_probability_caps(
    probs: Iterable[float] | np.ndarray | pd.Series,
    market_probs: Iterable[float] | np.ndarray | pd.Series | None = None,
    base_high: float = 0.85,
    margin: float = 0.05,
    min_floor: float = 0.02,
    max_cap: float = 0.98,
) -> np.ndarray:
    """
    Cap raw win probabilities to avoid extreme confidence unless supported by market odds.

    Parameters
    ----------
    probs : iterable of float
        Raw model probabilities (home win perspective).
    market_probs : iterable of float, optional
        Market-implied probabilities (e.g., from moneyline). Used to relax caps when Vegas agrees.
    base_high : float
        Default upper cap applied when market data is unavailable or less extreme.
    margin : float
        Buffer applied to market probabilities when adjusting caps.
    min_floor : float
        Minimum lower bound allowed after adjustments.
    max_cap : float
        Maximum upper bound allowed after adjustments.
    """
    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.size == 0:
        return probs_arr

    base_low = 1.0 - base_high
    upper = np.full_like(probs_arr, base_high, dtype=float)
    lower = np.full_like(probs_arr, base_low, dtype=float)

    if market_probs is not None:
        market_arr = (
            moneyline_to_prob(market_probs)
            if not isinstance(market_probs, (np.ndarray, pd.Series))
            else np.asarray(pd.to_numeric(market_probs, errors="coerce"), dtype=float)
        )
        if market_arr.size != probs_arr.size:
            market_arr = np.full_like(probs_arr, np.nan, dtype=float)
        valid = np.isfinite(market_arr)
        high_mask = valid & (market_arr > base_high)
        low_mask = valid & (market_arr < base_low)
        upper[high_mask] = np.minimum(max_cap, market_arr[high_mask] + margin)
        lower[low_mask] = np.maximum(min_floor, market_arr[low_mask] - margin)

    upper = np.clip(upper, base_high, max_cap)
    complementary = np.clip(1.0 - upper, min_floor, 1.0)
    lower = np.clip(lower, min_floor, complementary)
    conflict = lower >= upper
    if np.any(conflict):
        adjust = np.maximum(min_floor, upper[conflict] - 0.05)
        lower[conflict] = np.minimum(adjust, upper[conflict] - 1e-3)

    clipped = np.clip(probs_arr, lower, upper)
    return clipped
