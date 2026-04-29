# Live Prediction Tracking

This directory contains locked predictions, actual results, and validation metrics for live tracking.

## Directory Structure

```
predictions_log/
├── 2025/
│   ├── week_01_predictions.csv    # Locked before kickoff
│   ├── week_01_actuals.csv        # Fetched after games
│   ├── week_01_metrics.json       # Performance metrics
│   ├── week_01_report.md          # Weekly validation report
│   ├── week_02_predictions.csv
│   ├── week_02_actuals.csv
│   └── ...
├── 2026/
│   └── ...
└── README.md (this file)
```

## Workflow

### Thursday (Before Games)
Lock predictions with timestamps to prevent retroactive changes:
```bash
python -m analysis.live_tracking --lock-predictions --season 2025 --week 10
```

### Tuesday (After Games)
Fetch actual results and calculate metrics:
```bash
python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10
python -m analysis.live_tracking --calculate-metrics --season 2025 --week 10
python -m analysis.live_tracking --generate-report --season 2025 --week 10
```

### Or Run Complete Cycle
```bash
python -m analysis.live_tracking --weekly-cycle --season 2025 --week 10
```

## File Formats

### Predictions (week_NN_predictions.csv)
Locked before games with timestamp:
- `game_id`, `home_team`, `away_team`
- `home_win_prob`, `pred_home_margin`
- `locked_at` - ISO 8601 timestamp
- `lock_season`, `lock_week`

### Actuals (week_NN_actuals.csv)
Fetched from matchup_features.parquet:
- `game_id`, `home_team`, `away_team`
- `home_score`, `away_score`, `home_margin`
- `actual_home_win`
- `fetched_at` - ISO 8601 timestamp

### Metrics (week_NN_metrics.json)
Performance metrics:
```json
{
  "season": 2025,
  "week": 10,
  "n_games": 14,
  "calculated_at": "2025-11-12T10:30:00Z",
  "win_probability": {
    "accuracy": 0.714,
    "auc": 0.774,
    "brier": 0.215,
    "logloss": 0.621
  },
  "spread": {
    "mae": 10.04,
    "rmse": 13.21
  }
}
```

### Report (week_NN_report.md)
Human-readable markdown report with all metrics.

## Tracking Status

List all tracked weeks:
```bash
python -m analysis.live_tracking --list-tracked
```

Filter by season:
```bash
python -m analysis.live_tracking --list-tracked --season 2025
```

## Audit Trail

Every file includes timestamps:
- **Predictions**: `locked_at` field shows when predictions were locked
- **Actuals**: `fetched_at` field shows when results were retrieved
- **Metrics**: `calculated_at` field shows when metrics were computed

This provides full transparency and prevents retroactive changes to predictions.

## Integration with Pipeline

The live tracking system integrates with the main pipeline:

```bash
# Generate predictions
python -m src.predict.predict_upcoming --season 2025 --week 10

# Lock them immediately
python -m analysis.live_tracking --lock-predictions --season 2025 --week 10

# After games complete
python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10
python -m analysis.live_tracking --calculate-metrics --season 2025 --week 10
```

## Automation

Set up weekly cron jobs:

**Thursday (before games):**
```bash
# Generate and lock predictions
python -m src.predict.predict_upcoming --season 2025 --week auto
python -m analysis.live_tracking --lock-predictions --season 2025 --week auto
```

**Tuesday (after games):**
```bash
# Fetch actuals and generate report
python -m analysis.live_tracking --fetch-actuals --season 2025 --week auto
python -m analysis.live_tracking --calculate-metrics --season 2025 --week auto
python -m analysis.live_tracking --generate-report --season 2025 --week auto
```

## Success Criteria

✅ 50+ weeks of locked predictions vs actuals
✅ Full audit trail with timestamps
✅ Weekly performance tracking
✅ Automated validation reports
✅ No retroactive prediction changes