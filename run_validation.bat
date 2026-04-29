@echo off
REM ============================================================================
REM NFL Prediction Model - Automated Validation Runner
REM Phase 8: Smart validation automation for Windows
REM ============================================================================
REM
REM This script automatically runs the appropriate validation tasks based on:
REM - Day of week (Thursday = lock predictions, Tuesday = fetch results)
REM - Day of month (1st = monthly reports, end of season = full validation)
REM
REM Usage:
REM   run_validation.bat              Run appropriate tasks for today
REM   run_validation.bat --force-all  Run all validation tasks
REM   run_validation.bat --help       Show this help
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Color codes for output
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

REM ============================================================================
REM Configuration
REM ============================================================================

REM Current season and week (auto-detected or set manually)
set SEASON=2025
set WEEK=auto

REM Python executable (adjust if needed)
set PYTHON=python

REM Log file
set LOGFILE=validation_run_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

REM ============================================================================
REM Parse command line arguments
REM ============================================================================

set FORCE_ALL=0
set SHOW_HELP=0

:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="--force-all" set FORCE_ALL=1
if /i "%~1"=="--help" set SHOW_HELP=1
if /i "%~1"=="-h" set SHOW_HELP=1
shift
goto parse_args
:end_parse_args

if %SHOW_HELP%==1 (
    echo.
    echo %BLUE%NFL Prediction Model - Automated Validation Runner%RESET%
    echo.
    echo Usage:
    echo   run_validation.bat              Run appropriate tasks for today
    echo   run_validation.bat --force-all  Run all validation tasks
    echo   run_validation.bat --help       Show this help
    echo.
    echo Automatic Schedule:
    echo   Thursday:  Lock predictions before games
    echo   Tuesday:   Fetch results and generate reports
    echo   1st:       Monthly calibration and CLV reports
    echo   End of season: Full walk-forward validation
    echo.
    echo Manual Commands:
    echo   Lock predictions:     python -m analysis.live_tracking --lock-predictions --season %SEASON% --week %WEEK%
    echo   Fetch results:        python -m analysis.live_tracking --fetch-actuals --season %SEASON% --week %WEEK%
    echo   Weekly report:        python -m analysis.live_tracking --generate-report --season %SEASON% --week %WEEK%
    echo   Calibration:          python -m analysis.calibration_monitor --season %SEASON% --generate-report
    echo   CLV analysis:         python -m analysis.clv_tracker --season %SEASON% --generate-report
    echo   Paper trading:        python -m analysis.paper_trading --season %SEASON% --compare-strategies
    echo   Benchmarks:           python -m analysis.benchmark_comparison --season %SEASON% --generate-report
    echo   Walk-forward:         python -m analysis.walk_forward_validation
    echo.
    exit /b 0
)

REM ============================================================================
REM Detect current date/time
REM ============================================================================

echo %BLUE%========================================%RESET%
echo %BLUE%NFL Validation Runner%RESET%
echo %BLUE%========================================%RESET%
echo.

REM Get day of week (1=Monday, 7=Sunday)
for /f "tokens=1" %%a in ('powershell -command "(Get-Date).DayOfWeek.value__"') do set DOW=%%a

REM Get day of month
for /f "tokens=1" %%a in ('powershell -command "(Get-Date).Day"') do set DOM=%%a

REM Get month
for /f "tokens=1" %%a in ('powershell -command "(Get-Date).Month"') do set MONTH=%%a

REM Get day name
for /f "tokens=1" %%a in ('powershell -command "(Get-Date).DayOfWeek"') do set DAY_NAME=%%a

echo %YELLOW%Current Date:%RESET% %date%
echo %YELLOW%Day of Week:%RESET% %DAY_NAME% (DOW=%DOW%)
echo %YELLOW%Day of Month:%RESET% %DOM%
echo %YELLOW%Month:%RESET% %MONTH%
echo %YELLOW%Season:%RESET% %SEASON%
echo %YELLOW%Week:%RESET% %WEEK%
echo %YELLOW%Log File:%RESET% %LOGFILE%
echo.

REM Start logging
echo ============================================================================ > %LOGFILE%
echo NFL Validation Run - %date% %time% >> %LOGFILE%
echo ============================================================================ >> %LOGFILE%
echo Day of Week: %DAY_NAME% (DOW=%DOW%) >> %LOGFILE%
echo Day of Month: %DOM% >> %LOGFILE%
echo Month: %MONTH% >> %LOGFILE%
echo Season: %SEASON% >> %LOGFILE%
echo Week: %WEEK% >> %LOGFILE%
echo. >> %LOGFILE%

REM ============================================================================
REM Determine what to run
REM ============================================================================

set RUN_LOCK=0
set RUN_FETCH=0
set RUN_WEEKLY=0
set RUN_MONTHLY=0
set RUN_SEASONAL=0

if %FORCE_ALL%==1 (
    echo %YELLOW%Force mode: Running ALL validation tasks%RESET%
    echo Force mode: Running ALL validation tasks >> %LOGFILE%
    set RUN_LOCK=1
    set RUN_FETCH=1
    set RUN_WEEKLY=1
    set RUN_MONTHLY=1
    set RUN_SEASONAL=1
) else (
    REM Thursday (DOW=4) - Lock predictions before games
    if %DOW%==4 (
        echo %GREEN%Thursday detected: Locking predictions before games%RESET%
        echo Thursday detected: Locking predictions before games >> %LOGFILE%
        set RUN_LOCK=1
    )
    
    REM Tuesday (DOW=2) - Fetch results and generate reports
    if %DOW%==2 (
        echo %GREEN%Tuesday detected: Fetching results and generating reports%RESET%
        echo Tuesday detected: Fetching results and generating reports >> %LOGFILE%
        set RUN_FETCH=1
        set RUN_WEEKLY=1
    )
    
    REM First of month - Monthly reports
    if %DOM%==1 (
        echo %GREEN%First of month: Running monthly reports%RESET%
        echo First of month: Running monthly reports >> %LOGFILE%
        set RUN_MONTHLY=1
    )
    
    REM End of season (February) - Full validation
    if %MONTH%==2 (
        echo %GREEN%February detected: Running end-of-season validation%RESET%
        echo February detected: Running end-of-season validation >> %LOGFILE%
        set RUN_SEASONAL=1
    )
)

echo.

REM ============================================================================
REM Execute validation tasks
REM ============================================================================

set ERROR_COUNT=0
set SUCCESS_COUNT=0

REM ----------------------------------------------------------------------------
REM Task 1: Lock Predictions (Thursday)
REM ----------------------------------------------------------------------------

if %RUN_LOCK%==1 (
    echo %BLUE%[1/5] Locking predictions...%RESET%
    echo [1/5] Locking predictions... >> %LOGFILE%
    
    %PYTHON% -m analysis.live_tracking --lock-predictions --season %SEASON% --week %WEEK% >> %LOGFILE% 2>&1
    
    if !errorlevel! equ 0 (
        echo %GREEN%[PASS] Predictions locked successfully%RESET%
        echo [PASS] Predictions locked successfully >> %LOGFILE%
        set /a SUCCESS_COUNT+=1
    ) else (
        echo %RED%[FAIL] Failed to lock predictions%RESET%
        echo [FAIL] Failed to lock predictions >> %LOGFILE%
        set /a ERROR_COUNT+=1
    )
    echo.
)

REM ----------------------------------------------------------------------------
REM Task 2: Fetch Actuals (Tuesday)
REM ----------------------------------------------------------------------------

if %RUN_FETCH%==1 (
    echo %BLUE%[2/5] Fetching actual results...%RESET%
    echo [2/5] Fetching actual results... >> %LOGFILE%
    
    %PYTHON% -m analysis.live_tracking --fetch-actuals --season %SEASON% --week %WEEK% >> %LOGFILE% 2>&1
    
    if !errorlevel! equ 0 (
        echo %GREEN%[PASS] Actuals fetched successfully%RESET%
        echo [PASS] Actuals fetched successfully >> %LOGFILE%
        set /a SUCCESS_COUNT+=1
    ) else (
        echo %RED%[FAIL] Failed to fetch actuals%RESET%
        echo [FAIL] Failed to fetch actuals >> %LOGFILE%
        set /a ERROR_COUNT+=1
    )
    echo.
)

REM ----------------------------------------------------------------------------
REM Task 3: Weekly Reports (Tuesday)
REM ----------------------------------------------------------------------------

if %RUN_WEEKLY%==1 (
    echo %BLUE%[3/5] Generating weekly reports...%RESET%
    echo [3/5] Generating weekly reports... >> %LOGFILE%
    
    REM Calculate metrics
    %PYTHON% -m analysis.live_tracking --calculate-metrics --season %SEASON% --week %WEEK% >> %LOGFILE% 2>&1
    
    REM Generate report
    %PYTHON% -m analysis.live_tracking --generate-report --season %SEASON% --week %WEEK% >> %LOGFILE% 2>&1
    
    if !errorlevel! equ 0 (
        echo %GREEN%[PASS] Weekly reports generated%RESET%
        echo [PASS] Weekly reports generated >> %LOGFILE%
        set /a SUCCESS_COUNT+=1
    ) else (
        echo %RED%[FAIL] Failed to generate weekly reports%RESET%
        echo [FAIL] Failed to generate weekly reports >> %LOGFILE%
        set /a ERROR_COUNT+=1
    )
    echo.
)

REM ----------------------------------------------------------------------------
REM Task 4: Monthly Reports (1st of month)
REM ----------------------------------------------------------------------------

if %RUN_MONTHLY%==1 (
    echo %BLUE%[4/5] Generating monthly reports...%RESET%
    echo [4/5] Generating monthly reports... >> %LOGFILE%
    
    REM Calibration monitoring
    echo   - Calibration monitoring... >> %LOGFILE%
    %PYTHON% -m analysis.calibration_monitor --season %SEASON% --generate-report >> %LOGFILE% 2>&1
    
    REM CLV analysis
    echo   - CLV analysis... >> %LOGFILE%
    %PYTHON% -m analysis.clv_tracker --season %SEASON% --generate-report >> %LOGFILE% 2>&1
    
    REM Paper trading
    echo   - Paper trading simulation... >> %LOGFILE%
    %PYTHON% -m analysis.paper_trading --season %SEASON% --compare-strategies >> %LOGFILE% 2>&1
    
    REM Benchmark comparison
    echo   - Benchmark comparison... >> %LOGFILE%
    %PYTHON% -m analysis.benchmark_comparison --season %SEASON% --generate-report >> %LOGFILE% 2>&1
    
    if !errorlevel! equ 0 (
        echo %GREEN%[PASS] Monthly reports generated%RESET%
        echo [PASS] Monthly reports generated >> %LOGFILE%
        set /a SUCCESS_COUNT+=1
    ) else (
        echo %RED%[FAIL] Some monthly reports failed%RESET%
        echo [FAIL] Some monthly reports failed >> %LOGFILE%
        set /a ERROR_COUNT+=1
    )
    echo.
)

REM ----------------------------------------------------------------------------
REM Task 5: Seasonal Validation (End of season)
REM ----------------------------------------------------------------------------

if %RUN_SEASONAL%==1 (
    echo %BLUE%[5/5] Running full walk-forward validation...%RESET%
    echo [5/5] Running full walk-forward validation... >> %LOGFILE%
    
    %PYTHON% -m analysis.walk_forward_validation >> %LOGFILE% 2>&1
    
    if !errorlevel! equ 0 (
        echo %GREEN%[PASS] Walk-forward validation complete%RESET%
        echo [PASS] Walk-forward validation complete >> %LOGFILE%
        set /a SUCCESS_COUNT+=1
    ) else (
        echo %RED%[FAIL] Walk-forward validation failed%RESET%
        echo [FAIL] Walk-forward validation failed >> %LOGFILE%
        set /a ERROR_COUNT+=1
    )
    echo.
)

REM ============================================================================
REM Summary
REM ============================================================================

echo %BLUE%========================================%RESET%
echo %BLUE%Validation Summary%RESET%
echo %BLUE%========================================%RESET%
echo.
echo %GREEN%Successful tasks: %SUCCESS_COUNT%%RESET%
echo %RED%Failed tasks: %ERROR_COUNT%%RESET%
echo.
echo %YELLOW%Log file: %LOGFILE%%RESET%
echo.

echo. >> %LOGFILE%
echo ============================================================================ >> %LOGFILE%
echo Summary >> %LOGFILE%
echo ============================================================================ >> %LOGFILE%
echo Successful tasks: %SUCCESS_COUNT% >> %LOGFILE%
echo Failed tasks: %ERROR_COUNT% >> %LOGFILE%
echo. >> %LOGFILE%

if %ERROR_COUNT% gtr 0 (
    echo %RED%Some tasks failed. Check log file for details.%RESET%
    echo Some tasks failed. Check log file for details. >> %LOGFILE%
    exit /b 1
) else (
    if %SUCCESS_COUNT% equ 0 (
        echo %YELLOW%No tasks scheduled for today.%RESET%
        echo %YELLOW%Run with --force-all to execute all validation tasks.%RESET%
        echo No tasks scheduled for today. >> %LOGFILE%
    ) else (
        echo %GREEN%All tasks completed successfully!%RESET%
        echo All tasks completed successfully! >> %LOGFILE%
    )
    exit /b 0
)

@REM Made with Bob
