# Phase 4: Paper Trading Simulator - Implementation Guide

## Overview

The paper trading simulator (`analysis/paper_trading.py`) enables risk-free testing of betting strategies using historical predictions and closing odds. It implements Kelly criterion bankroll management and tracks performance metrics across multiple strategies.

## Key Features

### 1. Kelly Criterion Bet Sizing
Optimal bet sizing formula: `f = (bp - q) / b`
- `b` = decimal odds - 1 (net odds)
- `p` = win probability
- `q` = 1 - p (loss probability)
- `f` = fraction of bankroll to bet

**Fractional Kelly:** Most professional bettors use 25-50% of full Kelly to reduce variance while maintaining positive expected growth.

### 2. Four Predefined Strategies

| Strategy | Kelly Fraction | Min CLV | Max Bet % | Risk Profile |
|----------|---------------|---------|-----------|--------------|
| Conservative | 10% | 3.0% | 2.0% | Lowest risk, steady growth |
| Moderate | 25% | 2.0% | 5.0% | Balanced (recommended) |
| Kelly Optimal | 50% | 1.0% | 10.0% | Higher variance, optimal growth |
| Aggressive | 100% | 0.0% | 15.0% | Maximum risk, fastest growth/drawdown |

### 3. Performance Metrics

**Return Metrics:**
- **ROI (Return on Investment):** Total profit / total wagered
- **Total Profit:** Net gain/loss in dollars
- **Win Rate:** Percentage of winning bets

**Risk Metrics:**
- **Sharpe Ratio:** Risk-adjusted return (higher is better, >1.0 is good)
- **Max Drawdown:** Largest peak-to-trough decline in bankroll
- **Final Bankroll:** Ending capital after all bets

**Efficiency Metrics:**
- **Average Bet Size:** Mean stake per bet
- **Total Wagered:** Sum of all stakes
- **Number of Bets:** Total bets placed

## Installation

No additional dependencies required beyond the main project requirements.

## Usage

### Basic Simulation

```bash
# Simulate moderate strategy for 2025 season
python -m analysis.paper_trading --season 2025 --strategy moderate
```

### Compare All Strategies

```bash
# Compare conservative, moderate, Kelly optimal, and aggressive
python -m analysis.paper_trading --season 2025 --compare-strategies
```

### Generate Full Report

```bash
# Create comprehensive analysis report
python -m analysis.paper_trading --season 2025 --generate-report
```

### Custom Strategy

```python
from analysis.paper_trading import PaperTradingStrategy, PaperTradingSimulator

# Define custom strategy
custom = PaperTradingStrategy(
    name="custom",
    kelly_fraction=0.30,      # 30% Kelly
    min_clv=0.025,            # 2.5% minimum CLV
    max_bet_pct=0.06,         # 6% max bet size
    min_edge=0.01,            # 1% minimum edge
    min_prob=0.45,            # 45% minimum win probability
    max_prob=0.95             # 95% maximum win probability
)

# Run simulation
sim = PaperTradingSimulator(initial_bankroll=10000.0, strategy=custom)
# ... place bets ...
metrics = sim.get_metrics()
```

## Data Requirements

The simulator requires CLV data from `clv_tracker.py`:

```bash
# Generate CLV data first
python -m analysis.clv_tracker --season 2025 --generate-report

# Then run paper trading
python -m analysis.paper_trading --season 2025 --strategy moderate
```

**Required CLV columns:**
- `game_id`, `season`, `week`
- `bet_type` (moneyline or spread)
- `side` (home or away)
- `team`
- `model_prob` - Model's win probability
- `closing_odds` - Market closing odds (American format)
- `closing_prob` - Implied probability from closing odds
- `clv` - Closing Line Value (model_prob - closing_prob)
- `won` - Actual outcome (True/False)

## Output Files

### Strategy Results JSON
`analysis/paper_trading/YYYY_strategy_results.json`

```json
{
  "strategy": "moderate",
  "initial_bankroll": 10000.0,
  "final_bankroll": 11250.0,
  "total_profit": 1250.0,
  "roi": 0.0625,
  "n_bets": 45,
  "win_rate": 0.5778,
  "sharpe_ratio": 1.23,
  "max_drawdown": 0.0850,
  "avg_bet_size": 425.50,
  "total_wagered": 19147.50
}
```

### Bet History CSV
`analysis/paper_trading/YYYY_strategy_bets.csv`

Contains every bet placed with:
- Game details (game_id, season, week, team)
- Bet parameters (type, side, stake, odds)
- Model metrics (model_prob, clv)
- Outcome (won, profit, bankroll_after)

### Strategy Comparison JSON
`analysis/paper_trading/YYYY_comparison.json`

Side-by-side metrics for all strategies tested.

### Analysis Report
`analysis/paper_trading/YYYY_report.md`

Comprehensive markdown report with:
- Strategy performance summary
- Risk-adjusted returns
- Bet distribution analysis
- Recommendations

## Interpretation Guide

### ROI (Return on Investment)

**Formula:** `(Total Profit / Total Wagered) × 100%`

- **Positive ROI:** Strategy is profitable
- **>5% ROI:** Excellent performance (rare in sports betting)
- **2-5% ROI:** Good performance (sustainable edge)
- **0-2% ROI:** Marginal edge (may not overcome variance)
- **Negative ROI:** Losing strategy

**Break-even threshold:** 52.4% win rate needed to overcome typical -110 vig.

### Sharpe Ratio

**Formula:** `(Mean Return - Risk-Free Rate) / Std Dev of Returns`

- **>2.0:** Exceptional risk-adjusted returns
- **1.0-2.0:** Good risk-adjusted returns
- **0.5-1.0:** Acceptable risk-adjusted returns
- **<0.5:** Poor risk-adjusted returns

Higher Sharpe = better return per unit of risk taken.

### Max Drawdown

**Definition:** Largest peak-to-trough decline in bankroll.

- **<10%:** Conservative, low volatility
- **10-20%:** Moderate volatility (acceptable)
- **20-30%:** High volatility (requires discipline)
- **>30%:** Very high volatility (risky)

**Risk of Ruin:** With 20% max drawdown, you need 25% gain to recover. Manage carefully.

### Win Rate

**Break-even rates:**
- **-110 odds (typical):** 52.4% needed
- **Even money (+100):** 50.0% needed
- **-150 favorites:** 60.0% needed
- **+150 underdogs:** 40.0% needed

**Reality check:** Professional sports bettors typically achieve 53-55% win rates. Claims of 60%+ are extremely rare and often unsustainable.

## Strategy Selection Guide

### Conservative Strategy
**Best for:**
- Risk-averse bettors
- Small bankrolls (<$5,000)
- Learning and testing
- Long-term steady growth

**Characteristics:**
- Lowest volatility
- Smallest drawdowns
- Slowest growth
- Highest CLV threshold (only best opportunities)

### Moderate Strategy (Recommended)
**Best for:**
- Most bettors
- Medium bankrolls ($5,000-$25,000)
- Balanced risk/reward
- Sustainable long-term betting

**Characteristics:**
- Balanced volatility
- Acceptable drawdowns (10-15%)
- Good growth potential
- Reasonable CLV threshold

### Kelly Optimal Strategy
**Best for:**
- Experienced bettors
- Larger bankrolls (>$25,000)
- Higher risk tolerance
- Maximizing long-term growth

**Characteristics:**
- Higher volatility
- Larger drawdowns (15-25%)
- Faster growth potential
- Lower CLV threshold (more bets)

### Aggressive Strategy
**Best for:**
- High risk tolerance
- Very large bankrolls (>$50,000)
- Short-term testing only
- Understanding maximum variance

**Characteristics:**
- Highest volatility
- Largest drawdowns (>25%)
- Fastest growth OR fastest ruin
- No CLV threshold (all edges)

**Warning:** Full Kelly is mathematically optimal but psychologically difficult. Most professionals use 25-50% Kelly.

## Risk Management Best Practices

### 1. Bankroll Management
- Never bet more than 5% of bankroll on single game (even with Kelly)
- Keep 50% of bankroll in reserve for drawdowns
- Recalculate bet sizes after every bet (Kelly adjusts to current bankroll)

### 2. Variance Expectations
- **Short-term (10-20 bets):** Expect high variance, results may not reflect true edge
- **Medium-term (50-100 bets):** Variance decreases, edge becomes clearer
- **Long-term (200+ bets):** True edge emerges, variance smooths out

### 3. Psychological Factors
- **Losing streaks:** 5-10 losses in a row are normal even with 55% win rate
- **Drawdowns:** Expect 15-20% drawdowns even with positive edge
- **Discipline:** Stick to strategy during variance swings

### 4. When to Stop
- **Max drawdown exceeded:** Pause and reassess
- **Win rate <50% over 100+ bets:** Model may have lost edge
- **Emotional betting:** Taking bets outside strategy parameters

## Validation Workflow

### Step 1: Generate CLV Data
```bash
python -m analysis.clv_tracker --season 2025 --generate-report
```

### Step 2: Run Paper Trading
```bash
python -m analysis.paper_trading --season 2025 --compare-strategies
```

### Step 3: Analyze Results
- Review ROI and Sharpe ratio for each strategy
- Check max drawdown vs risk tolerance
- Examine bet distribution (are you getting enough opportunities?)
- Compare win rate to break-even threshold

### Step 4: Select Strategy
- Choose strategy matching your risk profile
- Verify positive ROI over 50+ bets
- Ensure max drawdown is acceptable
- Confirm Sharpe ratio >0.5

### Step 5: Forward Test
- Run selected strategy on new season
- Track performance weekly
- Adjust if edge deteriorates

## Testing

Run unit tests to verify functionality:

```bash
python test_paper_trading.py
```

Tests cover:
- American to decimal odds conversion
- Kelly criterion calculations
- Strategy definitions
- Simulator bet placement
- Metrics calculation

## Troubleshooting

### No CLV data found
**Error:** `FileNotFoundError: CLV data not found`

**Solution:** Run `clv_tracker.py` first to generate CLV data:
```bash
python -m analysis.clv_tracker --season 2025 --generate-report
```

### No bets placed
**Issue:** Simulator places 0 bets

**Causes:**
1. CLV threshold too high (no opportunities meet criteria)
2. Min edge threshold too high
3. Probability range too narrow

**Solution:** Use less restrictive strategy (e.g., "aggressive" instead of "conservative")

### Negative ROI
**Issue:** Strategy shows negative returns

**Causes:**
1. Model lacks true edge vs market
2. Sample size too small (variance)
3. Strategy parameters misaligned with model strengths

**Solution:**
1. Verify model has positive CLV over 100+ games
2. Increase sample size (test multiple seasons)
3. Adjust strategy thresholds based on CLV distribution

### High drawdown
**Issue:** Max drawdown exceeds comfort level

**Causes:**
1. Kelly fraction too high (full Kelly is volatile)
2. Insufficient diversification (too few bets)
3. Unlucky variance (short-term)

**Solution:**
1. Reduce Kelly fraction (use 25% instead of 50%)
2. Lower CLV threshold to increase bet frequency
3. Test over longer period to smooth variance

## Advanced Topics

### Multi-Season Analysis
```python
from analysis.paper_trading import simulate_season, compare_strategies

# Test across multiple seasons
seasons = [2023, 2024, 2025]
all_results = []

for season in seasons:
    results = compare_strategies(season)
    all_results.append(results)

# Aggregate metrics
total_roi = sum(r['roi'] for r in all_results) / len(all_results)
total_sharpe = sum(r['sharpe_ratio'] for r in all_results) / len(all_results)
```

### Custom Bet Filters
```python
# Only bet on home favorites with high CLV
def custom_filter(bet_data):
    return (
        bet_data['side'] == 'home' and
        bet_data['model_prob'] > 0.55 and
        bet_data['clv'] > 0.03
    )

# Apply filter in simulation
# (requires modifying simulator to accept filter function)
```

### Bankroll Growth Simulation
```python
# Monte Carlo simulation of bankroll growth
import numpy as np

def simulate_growth(initial, roi, volatility, n_bets):
    bankroll = initial
    for _ in range(n_bets):
        bet_size = bankroll * 0.02  # 2% per bet
        outcome = np.random.normal(roi, volatility)
        bankroll += bet_size * outcome
    return bankroll

# Run 1000 simulations
final_bankrolls = [
    simulate_growth(10000, 0.05, 0.15, 100)
    for _ in range(1000)
]

print(f"Median final: ${np.median(final_bankrolls):.2f}")
print(f"5th percentile: ${np.percentile(final_bankrolls, 5):.2f}")
```

## References

- **Kelly Criterion:** J. L. Kelly Jr., "A New Interpretation of Information Rate" (1956)
- **Fractional Kelly:** Edward O. Thorp, "The Kelly Criterion in Blackjack Sports Betting, and the Stock Market" (2008)
- **Sports Betting Math:** Joseph Buchdahl, "Fixed Odds Sports Betting: Statistical Forecasting and Risk Management" (2003)

## Next Steps

After completing paper trading analysis:

1. **Phase 5:** Implement calibration monitoring for prediction quality
2. **Phase 6:** Compare against public benchmarks (nfelo, Vegas consensus)
3. **Phase 7:** Integrate results into Streamlit dashboard
4. **Phase 8:** Automate weekly validation pipeline

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test file (`test_paper_trading.py`) for examples
3. Examine CLV data to verify model has edge
4. Consult `analysis/paper_trading/README.md` for strategy guide

---

**Remember:** Paper trading is simulation only. Real betting involves additional factors (line shopping, bet limits, account management, taxes, emotional discipline). Always bet responsibly and within your means.