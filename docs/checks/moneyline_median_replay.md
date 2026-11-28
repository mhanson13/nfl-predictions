# Market moneyline median: replay of saved 2026 runs (2026-11-25)

**Change:** `predict_upcoming` computed `market_home_moneyline` / `market_away_moneyline` as a plain
median of every book's American price. Books straddling even money (e.g. -105 and +100) gave
impossible prices between -100 and +100. The fix takes the median in decimal odds and rounds to
cents (`_first_existing_moneyline_median`). An odd number of books gives the same book price as
before; an even number gives the decimal midpoint of the two middle prices.

**Replay:** every saved 2026 run (`predictions/predictions.csv.backup_*` plus `predictions.csv`):
153 runs, of which 91 carry per-book moneylines (the other 62 have none, so the fix cannot change
them). For each run, `_attach_market_policy_fields` ran twice against the same policy files, once
with the old median and once with the new one, and every output cell was compared. Script:
scratchpad `mlfix/replay.py` (read-only).

## Result

- **No change** in any run to `home_pick`, `home_confidence_pct`, `market_model_pick`,
  `market_vegas_pick`, the model/calibrated confidence bins, any policy tier or status, the
  recommendation flags or `market_public_recommendation`.
- **6 games differ**, all in columns copied to MTB (`predictions.csv`):

| Week | Game | Change |
|---|---|---|
| 5 | 2026_05_CHI_GB | home ML **-1.5 -> -101.48**; market home prob 0.027 -> 0.482; ML edge -0.354 -> +0.101; edge bin `<0%` -> `10-15%` (1 run) |
| 8 | 2026_08_LV_NYJ | away ML **-1.0 -> -100.99** and **-1.5 -> -101.48**; market away prob 0.018/0.026 -> 0.480/0.481; ML edge 0.51 -> 0.05; edge bin `15%+` -> `5-8%`; vegas-side prob bin `<35%` -> `45-50%` (16 runs; most only by cents) |
| 6 | 2026_06_CHI_ATL, 2026_06_DAL_GB, 2026_06_NO_NYG | moneyline by 0.01-0.02 (e.g. -116.0 -> -115.99); probabilities/edges by < 0.0001 |
| 8 | 2026_08_KC_DEN | moneyline by 0.01-0.02; probabilities/edges by < 0.0001 |

**First affected week:** week 5 (run saved 2026-10-11). Only weeks 5 and 8 had impossible prices.

Past picks, the MTB archives and MTB data are **not** rewritten: published picks stand as
published (pick-of-record transparency rule). The fix applies from its merge (Tue 12-01) on.

**MTB (2026-11-25):** MTB does not read or display the market moneyline, no-vig probability or moneyline-edge columns, so no published page showed the impossible prices and nothing needs suppressing.
