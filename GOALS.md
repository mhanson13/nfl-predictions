## Model Performance Checklist

### 1. Basic Metadata
- [x] `n` = 2581 predictions - large, stable full-sample set (2002-2025)
- [x] Forward holdout (2024-2025) `n` = 399 games tracked separately
- [ ] Track `n` per season/run to ensure consistent evaluation size

---

### 2. Winner Prediction (Binary Outcome)

| Metric | Published Benchmark* | Current Value | Goal / Objective |
|--------|----------------------|---------------|------------------|
| Accuracy | ~0.65-0.70 (pregame models) | **0.579** | >= 0.70 full-season |
| AUC | >= 0.70 strong; >= 0.75 excellent | **0.761** | >= 0.78 next tier |

---

### 3. Probability Calibration

| Metric | Published Benchmark* | Current Value | Goal / Objective |
|--------|----------------------|---------------|------------------|
| Brier Score | ~0.208 (e.g., FiveThirtyEight) | **0.195** | <= 0.20 |
| LogLoss | Lower = better (no fixed public number) | **0.654** | <= 0.60 |

---

### 4. Margin/Score Prediction

| Metric | Context / Comparable | Current Value | Goal / Objective |
|--------|----------------------|---------------|------------------|
| MAE | ~10 pts typical NFL margin error | **10.629** | <= 9.5 |
| RMSE | Highlights larger prediction errors | **13.675** | <= 12.5 |

---

### 5. Forward Validation Snapshot (2024-2025 holdout)

| Metric | Value |
|--------|-------|
| Accuracy | 0.714 |
| AUC | 0.774 |
| Brier Score | 0.215 |
| LogLoss | 0.621 |
| MAE | 10.04 |
| RMSE | 13.21 |

---

### 6. Evaluation Methodology

- [x] Out-of-time (walk-forward) validation
- [x] Compare predictions vs **market closing odds** (ROI analysis integrated)
- [x] Calibration plots per probability bin
- [x] Feature importance tracking after each new data feed (feature lift analysis)
- [ ] Season-by-season metric table in README
- [ ] Document major pipeline/data changes in changelog

---

### 7. Improvement Actions

- [x] Added new data feed - **AUC improved +0.012** and **Brier -0.005**
- [x] Add volatility features (weather, injuries, travel, short rest)
- [x] Apply probability calibration technique (weekly isotonic scaling)
- [x] Build **market edge dashboard** (model vs implied odds)
- [x] Promote Visual Crossing historical weather feed (NOAA + Tomorrow.io now act as fallbacks)
- [ ] Re-evaluate metrics after each feature addition and document change

---

### 8. Feature & Market Insights

- **Feature lift (`analysis/feature_lift.py`)**
  - Win probability enriched vs baseline: AUC +0.018, Brier -0.006 (2016-2025 sample).
  - Spread enriched vs baseline: MAE -0.28 pts, RMSE -0.41 pts.
- **Forward validation (`analysis/forward_validation.py`)**
  - Recent holdout (2024-2025) retains MAE 9.53; highlights need for additional contemporary features to regain AUC >=0.72 in future seasons.
- **Market ROI (`analysis/market_roi.py`)**
  - 2020-2023: Home moneyline EV>5% filter ROI **+8.0%** across 310 wagers; Away spread edge>2 pts ROI **+0.60** per unit over 319 bets.
  - 2016-2025: Home EV>5% moneyline ROI **+2.8%**; Away edge>2 pts spread ROI **+0.61** per unit across 688 bets.
- **Volatility diagnostics (`analysis/volatility_slices.py`)**
  - High wind (>= 15 mph, 188 games) drives MAE **11.06** vs. 10.18 in calmer conditions.
  - Long travel (>= 1500 miles, 454 games) lifts MAE to **10.44**; calibrated shrinkage now tempers those predictions.
- **Weather coverage**
  - Visual Crossing hourly history now supplies primary weather features back to 2002, with NOAA (recent obs) and Tomorrow.io archives acting as lower-priority fallbacks.
- **Volatility classifier (`analysis/volatility_classifier.py`)**
  - Calibrated logistic (80% decision threshold) AUC **0.98**, precision **1.00**, recall **0.85** on 2024-2025 holdout; travel/timezone and wind signals remain top drivers for shrinkage coverage (~15% of games).

---

\*Benchmarks based on publicly available research and models (e.g., FiveThirtyEight pre-game win probabilities, Kaggle ML comparisons, sports analytics calibration literature).

---

### Instructions for Use
- Update "Current Value" and checkboxes after each major run.
- Record snapshot tables per season/year for historical progress tracking.
- Treat "Goal / Objective" column as next-step performance targets.
