# Automated Validation Guide - Phase 8

## Overview

The `run_validation.bat` script provides intelligent, automated validation for the NFL prediction model on Windows. It automatically determines which validation tasks to run based on the current day of the week and month, eliminating manual intervention while maintaining full transparency.

## Quick Start

### Basic Usage

Simply run the batch file - it will automatically determine what to do:

```cmd
run_validation.bat
```

### Force All Tasks

To run all validation tasks regardless of the day:

```cmd
run_validation.bat --force-all
```

### Show Help

```cmd
run_validation.bat --help
```

## Automatic Schedule

The script follows the NFL weekly cycle and validation best practices:

### Thursday - Lock Predictions
**What happens:** Predictions are locked with timestamps before games kick off
**Why:** Prevents retroactive changes and establishes audit trail
**Tasks run:**
- Lock predictions for current week
- Save with ISO 8601 timestamps

### Tuesday - Fetch Results & Weekly Reports
**What happens:** Actual game results are fetched and compared to predictions
**Why:** Validates model performance against real outcomes
**Tasks run:**
- Fetch actual game results
- Calculate weekly metrics (accuracy, AUC, Brier, MAE, RMSE)
- Generate weekly validation report

### 1st of Month - Monthly Reports
**What happens:** Comprehensive monthly analysis across all validation dimensions
**Why:** Tracks longer-term trends and model health
**Tasks run:**
- Calibration monitoring with drift detection
- CLV (Closing Line Value) analysis
- Paper trading simulation with Kelly criterion
- Benchmark comparison vs nfelo and Vegas

### February - End of Season Validation
**What happens:** Full walk-forward validation across multiple seasons
**Why:** Comprehensive multi-year performance assessment
**Tasks run:**
- Walk-forward validation with rolling windows
- Multi-season consistency analysis
- Complete validation report

### Other Days
**What happens:** Script exits gracefully with no tasks scheduled
**Why:** Validation only runs when needed to avoid redundant processing

## Configuration

### Season and Week

By default, the script uses:
- `SEASON=2025` (current season)
- `WEEK=auto` (automatically detects current week)

To change these, edit the configuration section in `run_validation.bat`:

```batch
REM Configuration
set SEASON=2025
set WEEK=auto
```

Or set to specific week:
```batch
set WEEK=10
```

### Python Executable

If your Python installation is not in PATH, update:

```batch
set PYTHON=C:\Python311\python.exe
```

## Output and Logging

### Console Output

The script provides color-coded console output:
- 🟦 **Blue:** Task headers and section dividers
- 🟩 **Green:** Successful operations
- 🟨 **Yellow:** Informational messages
- 🟥 **Red:** Errors and failures

### Log Files

Every run creates a timestamped log file:
```
validation_run_YYYYMMDD_HHMMSS.log
```

Example: `validation_run_20250101_143022.log`

The log contains:
- Execution timestamp
- Day of week and month
- All task outputs
- Success/failure status
- Error messages (if any)

### Generated Reports

Validation tasks create reports in their respective directories:

**Weekly Reports:**
```
predictions_log/2025/week_10_report.md
predictions_log/2025/week_10_metrics.json
```

**Monthly Reports:**
```
analysis/calibration_monitoring/2025_calibration_report.md
analysis/clv_tracking/2025_clv_report.md
analysis/paper_trading/2025_trading_report.md
analysis/benchmark_comparison/2025_benchmark_report.md
```

**Seasonal Reports:**
```
analysis/walk_forward_validation/validation_summary.json
analysis/walk_forward_validation/validation_report.md
```

## Task Scheduling

### Windows Task Scheduler

To run automatically, create scheduled tasks:

#### Thursday Task (Lock Predictions)

1. Open Task Scheduler
2. Create Basic Task
3. Name: "NFL Validation - Lock Predictions"
4. Trigger: Weekly, Thursday, 10:00 AM
5. Action: Start a program
   - Program: `cmd.exe`
   - Arguments: `/c "cd /d C:\Code\nfl-predictions && run_validation.bat"`
6. Finish

#### Tuesday Task (Fetch Results)

1. Open Task Scheduler
2. Create Basic Task
3. Name: "NFL Validation - Fetch Results"
4. Trigger: Weekly, Tuesday, 10:00 AM
5. Action: Start a program
   - Program: `cmd.exe`
   - Arguments: `/c "cd /d C:\Code\nfl-predictions && run_validation.bat"`
6. Finish

#### Monthly Task (Reports)

1. Open Task Scheduler
2. Create Basic Task
3. Name: "NFL Validation - Monthly Reports"
4. Trigger: Monthly, 1st day, 10:00 AM
5. Action: Start a program
   - Program: `cmd.exe`
   - Arguments: `/c "cd /d C:\Code\nfl-predictions && run_validation.bat"`
6. Finish

### PowerShell Scheduled Job

Alternative using PowerShell:

```powershell
# Thursday - Lock predictions
$trigger = New-JobTrigger -Weekly -DaysOfWeek Thursday -At "10:00AM"
Register-ScheduledJob -Name "NFL-LockPredictions" -Trigger $trigger -ScriptBlock {
    Set-Location "C:\Code\nfl-predictions"
    .\run_validation.bat
}

# Tuesday - Fetch results
$trigger = New-JobTrigger -Weekly -DaysOfWeek Tuesday -At "10:00AM"
Register-ScheduledJob -Name "NFL-FetchResults" -Trigger $trigger -ScriptBlock {
    Set-Location "C:\Code\nfl-predictions"
    .\run_validation.bat
}

# Monthly - Reports
$trigger = New-JobTrigger -Monthly -DaysOfMonth 1 -At "10:00AM"
Register-ScheduledJob -Name "NFL-MonthlyReports" -Trigger $trigger -ScriptBlock {
    Set-Location "C:\Code\nfl-predictions"
    .\run_validation.bat
}
```

## Manual Validation Commands

If you need to run specific validation tasks manually:

### Lock Predictions
```cmd
python -m analysis.live_tracking --lock-predictions --season 2025 --week 10
```

### Fetch Results
```cmd
python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10
```

### Weekly Report
```cmd
python -m analysis.live_tracking --generate-report --season 2025 --week 10
```

### Calibration Monitoring
```cmd
python -m analysis.calibration_monitor --season 2025 --generate-report
```

### CLV Analysis
```cmd
python -m analysis.clv_tracker --season 2025 --generate-report
```

### Paper Trading
```cmd
python -m analysis.paper_trading --season 2025 --compare-strategies
```

### Benchmark Comparison
```cmd
python -m analysis.benchmark_comparison --season 2025 --generate-report
```

### Walk-Forward Validation
```cmd
python -m analysis.walk_forward_validation
```

## Troubleshooting

### Script doesn't run

**Issue:** Double-clicking the batch file opens and closes immediately

**Solution:** Run from command prompt to see errors:
```cmd
cd C:\Code\nfl-predictions
run_validation.bat
```

### Python not found

**Issue:** `'python' is not recognized as an internal or external command`

**Solution:** Update the `PYTHON` variable in the script:
```batch
set PYTHON=C:\Python311\python.exe
```

Or add Python to PATH.

### Module not found errors

**Issue:** `ModuleNotFoundError: No module named 'analysis'`

**Solution:** Ensure you're running from the project root directory:
```cmd
cd C:\Code\nfl-predictions
run_validation.bat
```

### No tasks scheduled

**Issue:** Script says "No tasks scheduled for today"

**Solution:** This is normal on days other than Thursday, Tuesday, 1st of month, or February. Use `--force-all` to run all tasks:
```cmd
run_validation.bat --force-all
```

### Permission errors

**Issue:** Cannot write to log files or output directories

**Solution:** Run as administrator or check directory permissions.

### Task Scheduler not running

**Issue:** Scheduled tasks don't execute

**Solution:**
1. Check Task Scheduler is running
2. Verify task is enabled
3. Check task history for errors
4. Ensure "Run whether user is logged on or not" is selected
5. Verify working directory is set correctly

## Integration with Existing Workflow

The automated validation runner integrates seamlessly with the existing pipeline:

### Before Automation
```cmd
# Manual weekly workflow
python -m src.predict.predict_upcoming --season 2025 --week 10
python -m analysis.live_tracking --lock-predictions --season 2025 --week 10
# Wait for games...
python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10
python -m analysis.live_tracking --calculate-metrics --season 2025 --week 10
python -m analysis.live_tracking --generate-report --season 2025 --week 10
```

### After Automation
```cmd
# Automatic - just run on Thursday and Tuesday
run_validation.bat
```

### With Task Scheduler
```
# Completely automatic - zero manual intervention
# Tasks run automatically on schedule
```

## Best Practices

### 1. Review Logs Regularly
Check log files weekly to ensure tasks are running successfully:
```cmd
dir validation_run_*.log /o-d
type validation_run_20250101_143022.log
```

### 2. Monitor Dashboard
Use the Streamlit dashboard to visualize validation metrics:
```cmd
streamlit run streamlit_app.py
```
Navigate to "Live Validation" tab.

### 3. Investigate Failures
If tasks fail:
1. Check the log file for error messages
2. Run the failed command manually to debug
3. Verify data files exist and are accessible
4. Check Python environment and dependencies

### 4. Backup Validation Data
Regularly backup validation directories:
```cmd
xcopy predictions_log predictions_log_backup /E /I /Y
xcopy analysis analysis_backup /E /I /Y
```

### 5. Update Season Configuration
At the start of each season, update the `SEASON` variable:
```batch
set SEASON=2026
```

## Success Criteria

The automated validation system is working correctly when:

- ✅ Thursday tasks lock predictions before games
- ✅ Tuesday tasks fetch results and generate reports
- ✅ Monthly tasks run on the 1st of each month
- ✅ Seasonal validation runs in February
- ✅ All log files show successful task completion
- ✅ Dashboard displays current validation metrics
- ✅ No manual intervention required

## Advanced Usage

### Custom Scheduling

To run validation on different days, modify the day-of-week checks:

```batch
REM Run on Wednesday instead of Thursday
if %DOW%==3 (
    set RUN_LOCK=1
)
```

### Email Notifications

Add email notifications for failures (requires PowerShell):

```batch
if %ERROR_COUNT% gtr 0 (
    powershell -Command "Send-MailMessage -To 'you@example.com' -From 'nfl-validation@example.com' -Subject 'Validation Failed' -Body 'Check log: %LOGFILE%' -SmtpServer 'smtp.example.com'"
)
```

### Slack Notifications

Add Slack webhook notifications:

```batch
if %ERROR_COUNT% gtr 0 (
    curl -X POST -H "Content-Type: application/json" -d "{\"text\":\"Validation failed. Check log: %LOGFILE%\"}" https://hooks.slack.com/services/YOUR/WEBHOOK/URL
)
```

## Summary

The automated validation runner (`run_validation.bat`) provides:

- **Zero-touch operation** - Runs automatically based on day/month
- **Intelligent scheduling** - Knows what to run and when
- **Complete logging** - Full audit trail of all operations
- **Error handling** - Graceful failure with detailed error messages
- **Flexibility** - Manual override with `--force-all`
- **Integration** - Works with existing validation modules
- **Transparency** - Color-coded output and comprehensive logs

**Phase 8 is now complete. The validation infrastructure is fully automated.**

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-29  
**Status:** Complete - Phase 8 automated validation implemented