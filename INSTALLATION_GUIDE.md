# Installation Guide - NFL Predictions Platform

## Prerequisites
- Python 3.9 or higher
- Git
- pip (Python package manager)

## Step 1: Install Core Dependencies

### Option A: Using pip (Recommended)
```bash
# Install the project in editable mode with all dependencies
pip install -e .

# This will install:
# - Core dependencies (pandas, numpy, xgboost, etc.)
# - Development tools (pytest, black, ruff, mypy)
# - All packages defined in pyproject.toml
```

### Option B: Using requirements.txt (Legacy)
```bash
# Install from existing requirements.txt
pip install -r requirements.txt

# Then install dev dependencies separately
pip install pytest pytest-cov black ruff mypy bandit pre-commit pandera
```

## Step 2: Install Pre-commit Hooks

Pre-commit hooks automatically check code quality before each commit.

```bash
# Install pre-commit hooks
pre-commit install

# Test the hooks (optional)
pre-commit run --all-files
```

This will run on every `git commit`:
- **Black**: Auto-format Python code
- **Ruff**: Fast Python linter
- **Mypy**: Type checking
- **Bandit**: Security vulnerability scanner
- **Trailing whitespace**: Remove trailing spaces
- **YAML validation**: Check YAML syntax

## Step 3: Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v

# Check code formatting
black --check src/ tests/

# Run linter
ruff check src/ tests/

# Type checking
mypy src/
```

## Step 4: Set Up Environment Variables

Create a `.env` file in the project root (if not exists):

```bash
# API Keys
SPORTSDATAIO_API_KEY=your_key_here
SPORTRADAR_API_KEY=your_key_here
VISUALCROSSING_API_KEY=your_key_here

# Environment
ENVIRONMENT=development  # or 'production'

# Optional: Override default paths
# DATA_DIR=/custom/path/to/data
# RESULTS_DIR=/custom/path/to/results
```

## Step 5: Initialize Cache Directory

```bash
# The cache directory will be created automatically
# But you can pre-create it if needed
mkdir -p data/processed/.cache
```

## What Gets Installed

### Core Dependencies (from pyproject.toml)
- **pandas**: DataFrame operations
- **numpy**: Numerical computing
- **xgboost**: Gradient boosting models
- **scikit-learn**: ML utilities
- **requests**: HTTP requests
- **python-dotenv**: Environment variable management
- **pandera**: DataFrame schema validation
- **pyarrow**: Parquet file support

### Development Tools
- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **black**: Code formatter (PEP 8)
- **ruff**: Fast linter (replaces flake8, isort, etc.)
- **mypy**: Static type checker
- **bandit**: Security linter
- **pre-commit**: Git hook framework

### Optional Tools (for advanced features)
- **streamlit**: Web dashboard (if using streamlit_app.py)
- **plotly**: Interactive visualizations
- **shap**: Model interpretability

## Troubleshooting

### Issue: Pre-commit hooks fail
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Clear cache and reinstall
pre-commit clean
pre-commit install
```

### Issue: Import errors after installation
```bash
# Reinstall in editable mode
pip install -e . --force-reinstall

# Or add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Mypy type checking errors
```bash
# Install type stubs for libraries
pip install types-requests types-python-dateutil

# Or disable strict mode temporarily in pyproject.toml
```

### Issue: Black and existing code style conflicts
```bash
# Format all code to match Black style
black src/ tests/ tools/

# This is a one-time operation
```

## Recommended IDE Setup

### VS Code
Install these extensions:
- Python (Microsoft)
- Pylance (Microsoft)
- Black Formatter (Microsoft)
- Ruff (Astral Software)
- Even Better TOML (tamasfe)

Settings are already configured in `.editorconfig`.

### PyCharm
- Enable Black formatter: Settings → Tools → Black
- Enable Ruff: Settings → Tools → External Tools
- Configure pytest: Settings → Tools → Python Integrated Tools

## Next Steps After Installation

1. **Run the test suite**: `pytest tests/ -v`
2. **Format existing code**: `black src/ tests/ tools/`
3. **Check for issues**: `ruff check src/ tests/ --fix`
4. **Run type checking**: `mypy src/`
5. **Test pre-commit**: `pre-commit run --all-files`

## Continuous Integration (Future)

When ready, add GitHub Actions workflow:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -e .
      - run: pytest tests/
      - run: black --check src/ tests/
      - run: ruff check src/ tests/
      - run: mypy src/
```

## Questions?

- Check `pyproject.toml` for all dependencies
- Check `.pre-commit-config.yaml` for hook configurations
- Check `src/config.py` for configuration options
- Review `README.md` for project overview