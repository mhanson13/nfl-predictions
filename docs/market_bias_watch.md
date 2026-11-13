# Watch item: home win probability vs the market

**Opened:** 2026-11-13, during the week 10 review.
**Monitor:** `src/evaluation/market_bias_monitor.py`. It runs after every
pipeline run, prints `[market_bias]` lines into the debug log, and writes
`predictions/evaluation/market_bias_monitor.csv`.

## What we saw

The win model is anchored on the no-vig market: XGBoost starts from the market
logit and learns a residual. In the week 10 live slate the published home
probability was **below the market in 12 of 13 games**. The mean gap was −1.6
points (t = −4.3), compared with a typical gap of about −0.4 points across past
seasons.

| Scope | Games | Mean gap (model − market) | Home win rate | Market mean | Brier vs market | Brier cost of the shift |
|---|---|---|---|---|---|---|
| Week 10 slate | 13 | −0.016 | — | — | — | — |
| 2026 season (walk-forward) | 136 | −0.005 | 0.493 | 0.554 | +0.0021 | −0.0006 |
| All history 2018–2026 | 2,355 | −0.004 | 0.541 | 0.553 | +0.0000 | −0.0001 |

How to read it:

- Across history the model marks home teams down slightly, and that has been
  right: home teams won 54.1% against a market-implied 55.3%.
- In 2026, home teams are winning only 49% against a market-implied 55%. The
  model leans the same way, and the lean has *helped*: removing it would make
  2026 Brier 0.0006 worse.
- The model still trails the market on 2026 Brier by 0.0021. That loss comes
  from game-by-game residual noise, not from the home shift.
- The risk is that the model *chases* a short-run home slump. Week 10's −1.6
  points is about four times the historical lean. If 2026 home results revert,
  a bigger markdown will cost Brier.

## What the monitor reports

| Scope | Meaning |
|---|---|
| `current_slate` | Mean gap over the games being predicted. `watch` when the absolute gap is at least 0.015 |
| `season`, `last_4_weeks`, `last_8_weeks` | Walk-forward games this season, with the home win rate, Brier vs market, and `brier_cost_of_shift` (Brier of the model minus Brier with its mean gap removed) |
| `all_history` | Context only (`info`) |
| `season_<YYYY>` | Per-season history rows (`history`) |

Status for the history windows:
- `watch`: the gap is statistically clear (|t| ≥ 2).
- `alert`: the gap is clear *and* removing it would have improved Brier by more than 0.0005. The lean is then actively costing accuracy.

## What to do

| Monitor shows | Action |
|---|---|
| `watch` only | Note it in the weekly review; no change |
| `alert` on `last_8_weeks` or `season` | Investigate which features drive the residual (SHAP on the residual model, `analysis/shap`). Consider stronger regularization of the residual, or a cap on the per-slate mean residual |
| `current_slate` gap at least 0.03 | Treat as a data problem until proven otherwise: check injuries, QB status and the market line for that slate |

## Log

| Date | Week | Slate gap | Season status | Note |
|---|---|---|---|---|
| 2026-11-13 | 10 | −0.016 (`watch`) | `watch`, shift cost −0.0006 | Opened. The lean matches 2026 home results; keep watching |
