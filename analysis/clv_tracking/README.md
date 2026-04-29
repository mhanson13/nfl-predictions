# Closing Line Value (CLV) Tracking

This directory contains CLV analysis results - the gold standard metric for measuring true edge against the betting market.

## What is CLV?

**Closing Line Value (CLV)** measures how often the model's implied probability beats the closing odds.

```
CLV = model_probability - closing_odds_implied_probability
```

### Why CLV Matters

- **Positive CLV** = Model found value vs market
- **Consistent positive CLV** = True edge (even if win rate < 100%)
- **CLV correlates with long-term profitability**
- **Break-even threshold** = 52.4% (accounting for vig)

### Example

```
Model says: 65% chance home team wins
Closing odds: -150 (60% implied probability)
CLV = 0.65 - 0.60 = +0.05 (5% edge)
```

Even if this bet loses, finding consistent +5% CLV across many bets = long-term profit.

## Directory Structure

```
analysis/clv_tracking/
├── 2025_moneyline_clv.csv          # Full season moneyline CLV
├── 2025_spread_clv.csv             # Full season spread CLV
├── 2025_clv_report.md              # Full season report
├── 2025_week_10_moneyline_clv.csv  # Weekly moneyline CLV
├── 2025_week_10_spread_clv.csv     # Weekly spread CLV
├── 2025_week_10_clv_report.md      # Weekly report
└── README.md (this file)
```

## Usage

### Calculate CLV for Specific Week

```bash
python -m analysis.clv_tracker --season 2025 --week 10 --generate-report
```

### Calculate CLV for Full Season

```bash
python -m analysis.clv_tracker --season 2025 --generate-report
```

### Quick Analysis

```bash
# Just calculate, don't generate full report
python -m analysis.clv_tracker --season 2025 --week 10
```

## File Formats

### Moneyline CLV (YYYY_moneyline_clv.csv)

```csv
game_id,season,week,bet_type,side,team,model_prob,closing_odds,closing_prob,clv,actual_margin,actual_win
2025_10_BUF_KC,2025,10,moneyline,home,KC,0.65,-150,0.60,0.05,3,1
2025_10_BUF_KC,2025,10,moneyline,away,BUF,0.35,+130,0.435,-0.085,-3,0
```

**Key Fields:**
- `model_prob` - Model's win probability
- `closing_odds` - Closing moneyline odds (American format)
- `closing_prob` - Implied probability from closing odds
- `clv` - Closing Line Value (positive = model found value)
- `actual_win` - Did the bet win? (1=yes, 0=no)

### Spread CLV (YYYY_spread_clv.csv)

```csv
game_id,season,week,bet_type,side,team,spread_line,model_prob,closing_odds,closing_prob,clv,actual_cover
2025_10_BUF_KC,2025,10,spread,home,KC,-3.5,0.58,-110,0.524,0.056,1
2025_10_BUF_KC,2025,10,spread,away,BUF,+3.5,0.42,-110,0.524,-0.104,0
```

**Key Fields:**
- `spread_line` - Point spread
- `model_prob` - Model's cover probability
- `closing_odds` - Closing spread odds (American format)
- `clv` - Closing Line Value
- `actual_cover` - Did the bet cover? (1=yes, 0=no)

### CLV Report (YYYY_clv_report.md)

Human-readable markdown report with:
- Overall CLV statistics (mean, median, std dev)
- Positive CLV rate
- Analysis by CLV threshold (0%, 2%, 5%)
- Hit rates for each threshold

## Interpreting Results

### Good CLV Performance

✅ **Mean CLV > 0** - Model consistently finds value
✅ **Positive CLV Rate > 50%** - More than half of opportunities have value
✅ **Hit Rate > 52.4%** when CLV > 0 - Profitable after vig
✅ **Consistent across weeks** - Not just lucky streaks

### Warning Signs

⚠️ **Mean CLV < 0** - Model consistently on wrong side of market
⚠️ **Positive CLV Rate < 40%** - Rarely finding value
⚠️ **Hit Rate < 50%** even with positive CLV - Model miscalibrated
⚠️ **High variance** - Inconsistent CLV across weeks

## Example Report

```markdown
# Closing Line Value (CLV) Report

**Season:** 2025  
**Period:** Full Season  

## Moneyline CLV

| Metric | Value |
|--------|-------|
| Total Opportunities | 544 |
| Mean CLV | +0.0234 |
| Median CLV | +0.0156 |
| Std Dev | 0.0892 |
| Positive CLV Rate | 58.3% |

### By CLV Threshold

| Threshold | Opportunities | Mean CLV | Hit Rate |
|-----------|---------------|----------|----------|
| >0.00 | 317 | +0.0456 | 54.2% |
| >0.02 | 189 | +0.0623 | 56.1% |
| >0.05 | 87 | +0.0891 | 58.6% |
```

## Integration with Live Tracking

CLV analysis works with the live tracking system:

```bash
# Week workflow
# 1. Lock predictions (Thursday)
python -m analysis.live_tracking --lock-predictions --season 2025 --week 10

# 2. After games (Tuesday)
python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10

# 3. Calculate CLV
python -m analysis.clv_tracker --season 2025 --week 10 --generate-report
```

## Key Insights

### CLV vs Win Rate

**Important:** CLV is more predictive of long-term success than raw win rate.

- **High win rate, negative CLV** = Lucky streak, will regress
- **Moderate win rate, positive CLV** = True edge, sustainable
- **Low win rate, positive CLV** = Unlucky variance, keep betting

### Sample Size

- **< 50 bets** - Too small, high variance
- **50-100 bets** - Starting to see patterns
- **100-500 bets** - Reliable signal
- **500+ bets** - High confidence in edge

### Threshold Selection

- **CLV > 0%** - All positive CLV opportunities
- **CLV > 2%** - Moderate edge filter
- **CLV > 5%** - Strong edge filter (fewer opportunities)

## Success Criteria

✅ Mean CLV > +0.02 (2% edge)
✅ Positive CLV rate > 55%
✅ Hit rate > 52.4% when CLV > 0
✅ Consistent across 100+ opportunities
✅ Stable across multiple weeks/seasons

## References

- [Pinnacle: Closing Line Value](https://www.pinnacle.com/en/betting-articles/Betting-Strategy/closing-line-value-betting)
- [The Logic of Sports Betting](https://www.amazon.com/Logic-Sports-Betting-Ed-Miller/dp/1909457868)
- Academic research on CLV and long-term profitability