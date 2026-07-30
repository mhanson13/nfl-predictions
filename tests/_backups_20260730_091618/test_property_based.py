"""
Property-based tests for data transformations.

Uses hypothesis for property-based testing to ensure data transformations
maintain invariants and handle edge cases correctly.
"""

import pytest  # type: ignore
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes

from src.utils.odds import (
    american_to_decimal,
    decimal_to_american,
    implied_probability,
    calculate_ev
)


class TestOddsConversions:
    """Property-based tests for odds conversions."""
    
    @given(st.integers(min_value=100, max_value=10000))
    def test_american_positive_to_decimal_roundtrip(self, odds):
        """Test that converting positive American odds to decimal and back preserves value."""
        decimal = american_to_decimal(odds)
        back_to_american = decimal_to_american(decimal)
        
        # Should be approximately equal (within rounding)
        assert abs(back_to_american - odds) <= 1
    
    @given(st.integers(min_value=-10000, max_value=-100))
    def test_american_negative_to_decimal_roundtrip(self, odds):
        """Test that converting negative American odds to decimal and back preserves value."""
        decimal = american_to_decimal(odds)
        back_to_american = decimal_to_american(decimal)
        
        # Should be approximately equal (within rounding)
        assert abs(back_to_american - odds) <= 1
    
    @given(st.floats(min_value=1.01, max_value=100.0))
    def test_decimal_odds_always_positive(self, decimal_odds):
        """Test that decimal odds are always positive."""
        assume(not np.isnan(decimal_odds))
        assume(not np.isinf(decimal_odds))
        
        american = decimal_to_american(decimal_odds)
        
        # American odds can be positive or negative, but never zero
        assert american != 0
    
    @given(st.floats(min_value=1.01, max_value=100.0))
    def test_implied_probability_bounds(self, decimal_odds):
        """Test that implied probability is always between 0 and 1."""
        assume(not np.isnan(decimal_odds))
        assume(not np.isinf(decimal_odds))
        
        prob = implied_probability(decimal_odds)
        
        assert 0.0 <= prob <= 1.0
    
    @given(
        st.floats(min_value=0.01, max_value=0.99),
        st.floats(min_value=1.01, max_value=100.0)
    )
    def test_expected_value_properties(self, win_prob, decimal_odds):
        """Test expected value calculation properties."""
        assume(not np.isnan(win_prob))
        assume(not np.isnan(decimal_odds))
        assume(not np.isinf(decimal_odds))
        
        stake = 100.0
        ev = calculate_ev(win_prob, decimal_odds, stake)
        
        # EV should be finite
        assert not np.isnan(ev)
        assert not np.isinf(ev)
        
        # If win_prob is 0, EV should be -stake
        if win_prob == 0:
            assert abs(ev - (-stake)) < 0.01
        
        # If win_prob is 1, EV should be positive
        if win_prob == 1:
            assert ev > 0


class TestDataFrameTransformations:
    """Property-based tests for DataFrame transformations."""
    
    @given(data_frames([
        column('team', dtype=str),
        column('score', dtype=int),
        column('opponent_score', dtype=int)
    ], index=range_indexes(min_size=1, max_size=100)))
    def test_win_loss_calculation_properties(self, df):
        """Test that win/loss calculations maintain invariants."""
        # Calculate wins
        df['win'] = (df['score'] > df['opponent_score']).astype(int)
        df['loss'] = (df['score'] < df['opponent_score']).astype(int)
        df['tie'] = (df['score'] == df['opponent_score']).astype(int)
        
        # Property: win + loss + tie should always equal 1 for each row
        assert (df['win'] + df['loss'] + df['tie'] == 1).all()
        
        # Property: wins and losses should be binary
        assert df['win'].isin([0, 1]).all()
        assert df['loss'].isin([0, 1]).all()
    
    @given(data_frames([
        column('value', dtype=float),
    ], index=range_indexes(min_size=1, max_size=100)))
    def test_rolling_average_properties(self, df):
        """Test that rolling averages maintain properties."""
        assume(not df['value'].isna().all())
        assume(not df['value'].isinf().any())
        
        window = 3
        df['rolling_avg'] = df['value'].rolling(window=window, min_periods=1).mean()
        
        # Property: rolling average should be between min and max of window
        for i in range(len(df)):
            start = max(0, i - window + 1)
            window_values = df['value'].iloc[start:i+1]
            
            if not window_values.isna().all():
                rolling_val = df['rolling_avg'].iloc[i]
                if not np.isnan(rolling_val):
                    assert window_values.min() <= rolling_val <= window_values.max()
    
    @given(data_frames([
        column('team_a', dtype=str),
        column('team_b', dtype=str),
    ], index=range_indexes(min_size=1, max_size=50)))
    def test_team_pairing_symmetry(self, df):
        """Test that team pairings are symmetric."""
        # Create pairs
        df['pair'] = df.apply(
            lambda row: tuple(sorted([row['team_a'], row['team_b']])),
            axis=1
        )
        
        # Property: sorted pairs should be deterministic
        for _, row in df.iterrows():
            pair = tuple(sorted([row['team_a'], row['team_b']]))
            assert row['pair'] == pair


class TestTeamAbbreviationNormalization:
    """Property-based tests for team abbreviation normalization."""
    
    @given(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
    def test_normalization_idempotent(self, abbr):
        """Test that normalizing twice gives same result as normalizing once."""
        from src.utils.teams import normalize_team_abbr
        
        # First normalization
        normalized_once = normalize_team_abbr(abbr)
        
        # Second normalization
        normalized_twice = normalize_team_abbr(normalized_once)
        
        # Should be the same
        assert normalized_once == normalized_twice
    
    @given(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
    def test_normalization_uppercase(self, abbr):
        """Test that normalized abbreviations are uppercase."""
        from src.utils.teams import normalize_team_abbr
        
        normalized = normalize_team_abbr(abbr)
        
        # Should be uppercase
        assert normalized == normalized.upper()
    
    @given(st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=10))
    def test_normalization_consistency(self, abbrs):
        """Test that same input always gives same output."""
        from src.utils.teams import normalize_team_abbr
        
        # Normalize all
        normalized = [normalize_team_abbr(abbr) for abbr in abbrs]
        
        # Normalize again
        normalized_again = [normalize_team_abbr(abbr) for abbr in abbrs]
        
        # Should be identical
        assert normalized == normalized_again


class TestStatisticalAggregations:
    """Property-based tests for statistical aggregations."""
    
    @given(st.lists(st.floats(min_value=-1000, max_value=1000), min_size=1, max_size=100))
    def test_mean_bounds(self, values):
        """Test that mean is between min and max."""
        assume(all(not np.isnan(v) and not np.isinf(v) for v in values))
        
        mean = np.mean(values)
        
        assert min(values) <= mean <= max(values)
    
    @given(st.lists(st.floats(min_value=0, max_value=1000), min_size=2, max_size=100))
    def test_variance_non_negative(self, values):
        """Test that variance is always non-negative."""
        assume(all(not np.isnan(v) and not np.isinf(v) for v in values))
        assume(len(set(values)) > 1)  # Need at least 2 distinct values
        
        variance = np.var(values)
        
        assert variance >= 0
    
    @given(st.lists(st.floats(min_value=-1000, max_value=1000), min_size=1, max_size=100))
    def test_sum_commutative(self, values):
        """Test that sum is commutative."""
        assume(all(not np.isnan(v) and not np.isinf(v) for v in values))
        
        sum1 = sum(values)
        sum2 = sum(reversed(values))
        
        assert abs(sum1 - sum2) < 1e-10


class TestDateTimeOperations:
    """Property-based tests for date/time operations."""
    
    @given(st.datetimes(min_value=pd.Timestamp('2000-01-01'), max_value=pd.Timestamp('2030-12-31')))
    def test_date_parsing_roundtrip(self, dt):
        """Test that date parsing and formatting roundtrips correctly."""
        # Convert to string and back
        date_str = dt.strftime('%Y-%m-%d')
        parsed = pd.to_datetime(date_str)
        
        # Should match original date (ignoring time)
        assert parsed.date() == dt.date()
    
    @given(
        st.datetimes(min_value=pd.Timestamp('2000-01-01'), max_value=pd.Timestamp('2030-12-31')),
        st.datetimes(min_value=pd.Timestamp('2000-01-01'), max_value=pd.Timestamp('2030-12-31'))
    )
    def test_date_difference_properties(self, dt1, dt2):
        """Test properties of date differences."""
        diff = (dt2 - dt1).days
        reverse_diff = (dt1 - dt2).days
        
        # Property: difference should be opposite when reversed
        assert diff == -reverse_diff
        
        # Property: difference with self should be zero
        assert (dt1 - dt1).days == 0


class TestCachingInvariants:
    """Property-based tests for caching behavior."""
    
    @given(st.text(min_size=1, max_size=100), st.integers(min_value=1, max_value=1000))
    def test_cache_key_deterministic(self, key, value):
        """Test that cache keys are deterministic."""
        from src.utils.cache import get_cache
        
        cache = get_cache()
        
        # Set value
        cache.set(key, value)
        
        # Get value multiple times
        val1 = cache.get(key)
        val2 = cache.get(key)
        
        # Should be the same
        assert val1 == val2 == value


# Configure hypothesis settings for faster tests
settings.register_profile("ci", max_examples=50, deadline=1000)
settings.register_profile("dev", max_examples=10, deadline=500)
settings.register_profile("thorough", max_examples=200, deadline=2000)

# Use dev profile by default
settings.load_profile("dev")

# Made with Bob
