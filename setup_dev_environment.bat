@echo off
REM NFL Predictions - Development Environment Setup Script
REM Run this script to install all development dependencies and configure tools

echo ========================================
echo NFL Predictions - Dev Environment Setup
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo ERROR: pyproject.toml not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo [1/5] Installing development dependencies...
pip install pytest pytest-cov pytest-xdist hypothesis black ruff mypy pre-commit pandera ipython jupyter bandit
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/5] Installing pre-commit hooks...
pre-commit install
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to install pre-commit hooks
    echo You can install them later with: pre-commit install
)

echo.
echo [3/5] Verifying installation...
python -c "import pandas; import xgboost; import pandera; import pytest; print('✓ All core imports successful')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Import verification failed
    pause
    exit /b 1
)

echo.
echo [4/5] Checking XGBoost GPU support...
python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"

echo.
echo [5/5] Running initial code formatting...
echo This will format all Python files to match Black style...
choice /C YN /M "Do you want to format all code now"
if %ERRORLEVEL% EQU 1 (
    black src/ tests/ tools/
    echo Code formatting complete!
) else (
    echo Skipping code formatting. You can run it later with: black src/ tests/ tools/
)

echo.
echo ========================================
echo Setup Complete! 🎉
echo ========================================
echo.
echo Next steps:
echo 1. Run tests: pytest tests/ -v
echo 2. Check code quality: pre-commit run --all-files
echo 3. Format code: black src/ tests/ tools/
echo 4. Lint code: ruff check src/ tests/ --fix
echo 5. Type check: mypy src/
echo.
echo For GPU training, use: python -m src.models.train --use-gpu
echo.
pause

@REM Made with Bob
