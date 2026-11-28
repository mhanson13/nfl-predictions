"""Market moneyline median across books is taken in decimal odds (no impossible prices)."""

import numpy as np
import pandas as pd

from src.predict.predict_upcoming import (
    _first_existing_moneyline_median,
    _first_existing_numeric_median,
)

PATTERN = r"^odds_.*_home_moneyline$"


def _frame(*rows):
    books = ["dk", "fd", "mgm", "czr"]
    return pd.DataFrame([dict(zip([f"odds_{b}_home_moneyline" for b in books], r)) for r in rows])


def test_books_straddling_even_money_give_a_real_price():
    df = _frame([-105, 100, np.nan, np.nan], [-102, -101, 100, 102])
    old = _first_existing_numeric_median(df, PATTERN)
    new = _first_existing_moneyline_median(df, PATTERN)
    assert old.tolist() == [-2.5, -0.5]  # the bug: prices between -100 and +100
    assert new.tolist() == [-102.44, -100.5]  # decimal midpoint of -101 (1.990) and +100 (2.000)
    assert ((new.abs() >= 100) | new.isna()).all()


def test_odd_number_of_books_returns_the_middle_book_price():
    df = _frame([-120, -110, -115, np.nan], [150, 135, 140, np.nan], [-105, 100, 110, np.nan])
    assert _first_existing_moneyline_median(df, PATTERN).tolist() == [-115.0, 140.0, 100.0]


def test_same_sign_even_count_matches_the_old_median_to_the_cent():
    df = _frame([-110, -120, np.nan, np.nan], [130, 140, np.nan, np.nan], [-116, -116, np.nan, np.nan])
    new = _first_existing_moneyline_median(df, PATTERN)
    old = _first_existing_numeric_median(df, PATTERN)
    assert (new - old).abs().max() <= 0.25
    assert new.tolist()[2] == -116.0


def test_bucket_edges_have_no_float_noise():
    df = _frame([120, 120, np.nan, np.nan], [-120, np.nan, np.nan, np.nan])
    out = _first_existing_moneyline_median(df, PATTERN)
    assert out.tolist() == [120.0, -120.0]
    assert out.iloc[0] <= 120  # would be 120.00000000000001 without rounding


def test_missing_and_invalid_prices():
    df = _frame([np.nan, np.nan, np.nan, np.nan], [0, -110, np.nan, np.nan], ["x", 150, np.nan, np.nan])
    out = _first_existing_moneyline_median(df, PATTERN)
    assert np.isnan(out.iloc[0])
    assert out.tolist()[1:] == [-110.0, 150.0]
    assert _first_existing_moneyline_median(pd.DataFrame({"a": [1]}), PATTERN).isna().all()
