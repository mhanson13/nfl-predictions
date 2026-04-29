# Paper Trading Simulator

This directory contains paper trading simulation results - testing betting strategies without risking real money.

## What is Paper Trading?

**Paper trading** simulates placing bets with proper bankroll management to test strategies and track performance over time.

### Key Features

- **Kelly Criterion** - Optimal bet sizing based on edge
- **Multiple Strategies** - Conservative, Moderate, Kelly Optimal, Aggressive
- **Risk Metrics** - ROI, Sharpe ratio, max drawdown
- **Bankroll Tracking** - Complete history of bankroll changes
- **Performance Analysis** - Compare strategies head-to-head

## Directory Structure

```
analysis/paper_trading/
├── 2025_moneyline_results.json    # Detailed simulation results
├── 2025_moneyline_report.md       # Human-readable report
├── 2025_spread_results.json       # Spread betting results
├── 2025_spread_report.md          # Spread betting report
└── README.md (this file)
```

## Usage

### Simulate Single Strategy

```bash
python -m analysis.paper_trading --season 2025 --strategy moderate
```

### Compare All Strategies

```bash
python -m analysis.paper_trading --season 2025 --compare-strategies
```

### Generate Full Report

```bash
python -m analysis.paper_trading --season 2025 --generate-report
```

### Spread Betting

```bash
python -m analysis.paper_trading --season 2025 --bet-type spread --generate-report
```

## Strategies

### Conservative
- **Kelly Fraction:** 10% (very cautious)
- **Min CLV:** 3% (only strong edges)
- **Max Bet:** 2% of bankroll
- **Best For:** Risk-averse, capital preservation

### Moderate (Recommended)
- **Kelly Fraction:** 25% (quarter Kelly)
- **Min CLV:** 2% (moderate edges)
- **Max Bet:** 5% of bankroll
- **Best For:** Balanced risk/reward

### Kelly Optimal
- **Kelly Fraction:** 50% (half Kelly)
- **Min CLV:** 1% (most edges)
- **Max Bet:** 10% of bankroll
- **Best For:** Aggressive growth, can handle volatility

### Aggressive
- **Kelly Fraction:** 100% (full Kelly)
- **Min CLV:** 0% (all positive CLV)
- **Max Bet:** 15% of bankroll
- **Best For:** Maximum growth, high risk tolerance

## Kelly Criterion

The Kelly Criterion calculates optimal bet size:

```
f = (bp - q) / b

where:
  f = fraction of bankroll to bet
  b = decimal odds - 1
  p = win probability
  q = 1 - p
```

### Why Fractional Kelly?

- **Full Kelly** - Optimal growth but high volatility
- **Half Kelly** - 75% of growth, 50% of volatility
- **Quarter Kelly** - More stable, slower growth

Most professionals use 1/4 to 1/2 Kelly.

## Performance Metrics

### ROI (Return on Investment)
```
ROI = Total Profit / Total Staked
```
- **Good:** > 5%
- **Excellent:** > 10%
- **Elite:** > 15%

### Sharpe Ratio
Measures risk-adjusted returns:
- **Good:** > 1.0
- **Excellent:** > 2.0
- **Elite:** > 3.0

### Max Drawdown
Maximum peak-to-trough decline:
- **Acceptable:** < 20%
- **Good:** < 15%
- **Excellent:** < 10%

### Win Rate
Percentage of winning bets:
- **Break-even:** ~52.4% (accounting for vig)
- **Good:** > 55%
- **Excellent:** > 60%

## Example Report

```markdown
# Paper Trading Report

**Season:** 2025  
**Bet Type:** Moneyline  
**Initial Bankroll:** $10,000  

## Strategy Comparison

| Strategy | Bets | Win Rate | ROI | Final Bankroll | Max DD | Sharpe |
|----------|------|----------|-----|----------------|--------|--------|
| Conservative | 87 | 58.6% | +8.2% | $11,234 | 5.3% | 2.14 |
| Moderate | 189 | 56.1% | +12.4% | $13,567 | 12.1% | 1.87 |
| Kelly Optimal | 317 | 54.2% | +15.8% | $16,234 | 18.7% | 1.52 |
| Aggressive | 544 | 52.8% | +18.3% | $18,901 | 28.4% | 1.21 |
```

## Interpreting Results

### Good Performance

✅ **Positive ROI** - Making money over time
✅ **Win Rate > 52.4%** - Beating break-even
✅ **Sharpe > 1.0** - Good risk-adjusted returns
✅ **Max DD < 20%** - Manageable volatility
✅ **Consistent across strategies** - Not just luck

### Warning Signs

⚠️ **Negative ROI** - Losing money
⚠️ **Win Rate < 50%** - Below break-even
⚠️ **Sharpe < 0.5** - Poor risk-adjusted returns
⚠️ **Max DD > 30%** - Excessive volatility
⚠️ **Only one strategy profitable** - May be overfitting

## Integration with CLV Tracking

Paper trading uses CLV data:

```bash
# 1. Calculate CLV
python -m analysis.clv_tracker --season 2025 --generate-report

# 2. Simulate paper trading
python -m analysis.paper_trading --season 2025 --generate-report
```

## Risk Management

### Bankroll Management
- Never bet more than max_bet_pct of bankroll
- Use fractional Kelly to reduce volatility
- Maintain emergency reserve (don't bet 100%)

### Strategy Selection
- **New bettors:** Start with Conservative
- **Experienced:** Use Moderate or Kelly Optimal
- **Professionals:** Can consider Aggressive with proper bankroll

### Sample Size
- **< 50 bets:** Too small, high variance
- **50-100 bets:** Starting to see patterns
- **100-500 bets:** Reliable signal
- **500+ bets:** High confidence

## Success Criteria

✅ Positive ROI across multiple strategies
✅ Win rate > 52.4% (break-even threshold)
✅ Sharpe ratio > 1.0
✅ Max drawdown < 20%
✅ Consistent performance over 100+ bets
✅ ROI correlates with CLV

## Limitations

⚠️ **Past performance ≠ future results**
⚠️ **Simulated ≠ real betting** (no slippage, limits, etc.)
⚠️ **Assumes closing odds available** (may not be realistic)
⚠️ **No psychological factors** (tilt, discipline, etc.)
⚠️ **No bankroll constraints** (assumes infinite liquidity)

## Next Steps

After paper trading shows consistent profitability:

1. **Start small** - Use minimum bet sizes
2. **Track real results** - Compare to paper trading
3. **Adjust strategy** - Based on real-world performance
4. **Scale gradually** - Increase stakes as bankroll grows
5. **Stay disciplined** - Stick to strategy, avoid tilt

## References

- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Bankroll Management](https://www.pinnacle.com/en/betting-articles/Betting-Strategy/bankroll-management-in-betting)
- [Risk of Ruin](https://www.pinnacle.com/en/betting-articles/Betting-Strategy/risk-of-ruin-in-betting)