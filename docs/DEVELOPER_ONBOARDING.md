# Developer Onboarding Guide

Welcome to the NFL Predictions Platform! This guide will help you get up to speed quickly.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Development Setup](#development-setup)
4. [Code Organization](#code-organization)
5. [Key Patterns](#key-patterns)
6. [Testing](#testing)
7. [Common Tasks](#common-tasks)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Project Overview

The NFL Predictions Platform is a machine learning system that predicts NFL game outcomes using:
- Historical game data
- Player statistics and injuries
- Weather conditions
- Team performance metrics
- Betting market data

### Tech Stack
- **Language**: Python 3.9+
- **ML Framework**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy, pyarrow
- **APIs**: ESPN, Sportradar, NFLverse, weather services
- **Storage**: Parquet files, SQLite (tuning), Milvus (vector DB)
- **Testing**: pytest, hypothesis
- **Documentation**: Sphinx, Markdown

## Architecture

### High-Level Architecture
```
┌─────────────────┐
│  Data Sources   │ (ESPN, Sportradar, NFLverse, Weather)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Fetchers  │ (src/data/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Builder │ (src/features/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Models      │ (XGBoost, calibration)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predictions    │ (src/predict/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │ (src/evaluation/)
└─────────────────┘
```

### Key Components

1. **Data Layer** (`src/data/`)
   - Fetches data from external sources
   - Transforms raw data into standardized format
   - Handles caching and rate limiting

2. **Feature Layer** (`src/features/`)
   - Builds features from raw data
   - Calculates advanced metrics (EPA, efficiency, etc.)
   - Handles feature engineering

3. **Model Layer** (`src/predict/`)
   - Trains and loads ML models
   - Makes predictions
   - Handles model versioning

4. **Evaluation Layer** (`src/evaluation/`)
   - Evaluates model performance
   - Calibrates probabilities
   - Generates metrics and reports

5. **Utilities** (`src/utils/`)
   - Common utilities (caching, logging, I/O)
   - Retry and circuit breaker patterns
   - Connection pooling

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git
- 8GB+ RAM recommended
- API keys for data sources (see `INSTALL.md`)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nfl-predictions
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run tests**
   ```bash
   pytest tests/
   ```

6. **Run the pipeline**
   ```bash
   python tools/run_pipeline.py --year 2024
   ```

### IDE Setup

**VS Code** (recommended):
- Install Python extension
- Install Pylance for type checking
- Use workspace settings in `.vscode/settings.json`

**PyCharm**:
- Mark `src/` as Sources Root
- Enable pytest as test runner
- Configure Python interpreter to use venv

## Code Organization

```
nfl-predictions/
├── src/                    # Source code
│   ├── data/              # Data fetchers and transformers
│   ├── features/          # Feature engineering
│   ├── predict/           # Prediction logic
│   ├── evaluation/        # Model evaluation
│   ├── analysis/          # Analysis scripts
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── data/                  # Data storage
│   ├── reference/         # Reference data (teams, stadiums)
│   └── cache/             # Cached data
├── results/               # Model outputs
├── docs/                  # Documentation
│   └── adr/              # Architecture Decision Records
├── tools/                 # CLI tools and scripts
└── tuning/               # Hyperparameter tuning results
```

## Key Patterns

### 1. Data Loader Factory Pattern
**Purpose**: Consistent data loading with caching and validation

```python
from src.utils.data_loader import load_parquet

# Simple usage
df = load_parquet("data/games.parquet")

# Advanced usage
from src.utils.data_loader import ParquetDataLoader

loader = ParquetDataLoader(
    "data/games.parquet",
    columns=["game_id", "home_team", "away_team"],
    cache_enabled=True
)
df = loader.load()
```

**See**: [ADR 0001](adr/0001-data-loader-factory-pattern.md)

### 2. Abstract Base Fetchers
**Purpose**: Standardized interface for data fetching

```python
from src.data.base_fetcher import APIDataFetcher

class MyAPIFetcher(APIDataFetcher):
    def _fetch_raw(self, **kwargs):
        response = self.session.get(self._build_url("endpoint"))
        df = pd.DataFrame(response.json())
        return FetchResult(success=True, data=df)

fetcher = MyAPIFetcher(
    name="my_api",
    base_url="https://api.example.com",
    api_key="secret"
)
result = fetcher.fetch(save_to_file=True)
```

**See**: [ADR 0002](adr/0002-abstract-base-fetchers.md)

### 3. Retry Pattern
**Purpose**: Handle transient failures automatically

```python
from src.utils.retry import retry, retry_on_rate_limit

@retry(max_attempts=3, delay=1.0)
def fetch_data():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

@retry_on_rate_limit(max_attempts=5, delay=60.0)
def fetch_with_rate_limit():
    return api_client.get_data()
```

**See**: [ADR 0003](adr/0003-retry-pattern.md)

### 4. Circuit Breaker Pattern
**Purpose**: Prevent cascading failures

```python
from src.utils.circuit_breaker import circuit_breaker

@circuit_breaker("api_service", failure_threshold=5, timeout=60.0)
def call_external_api():
    return api.get_data()
```

### 5. Connection Pooling
**Purpose**: Reuse HTTP connections for better performance

```python
from src.utils.http_pool import pooled_session

with pooled_session() as session:
    response = session.get("https://api.example.com/data")
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_loaders.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run property-based tests
pytest tests/test_property_based.py

# Run contract tests
pytest tests/test_api_contracts.py
```

### Test Organization

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Property-Based Tests**: Test invariants with hypothesis
- **Contract Tests**: Validate API response schemas

### Writing Tests

```python
import pytest
from src.utils.data_loader import ParquetDataLoader

def test_load_parquet_file(tmp_path):
    """Test loading a parquet file."""
    # Arrange
    df = pd.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.to_parquet(file_path)
    
    # Act
    loader = ParquetDataLoader(file_path)
    result = loader.load()
    
    # Assert
    assert result is not None
    assert len(result) == 3
```

## Common Tasks

### Adding a New Data Source

1. Create fetcher class inheriting from appropriate base:
   ```python
   from src.data.base_fetcher import APIDataFetcher
   
   class NewAPIFetcher(APIDataFetcher):
       def _fetch_raw(self, **kwargs):
           # Implementation
           pass
   ```

2. Add tests in `tests/test_data_fetchers.py`

3. Document in `data/README.md`

4. Add to pipeline in `tools/run_pipeline.py`

### Adding a New Feature

1. Create feature function in `src/features/`:
   ```python
   def calculate_new_feature(df: pd.DataFrame) -> pd.DataFrame:
       """Calculate new feature."""
       df['new_feature'] = df['col1'] * df['col2']
       return df
   ```

2. Add to feature builder in `src/features/build_features.py`

3. Add tests

4. Document feature in docstring

### Running the Pipeline

```bash
# Full pipeline for a season
python tools/run_pipeline.py --year 2024

# Specific steps
python tools/run_pipeline.py --year 2024 --steps fetch,features

# With custom config
python tools/run_pipeline.py --year 2024 --config custom_config.yaml
```

### Making Predictions

```python
from src.predict.predict_upcoming import predict_upcoming_games

predictions = predict_upcoming_games(
    week=10,
    season=2024,
    model_path="results/model.pkl"
)
```

## Best Practices

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings for public functions
- Keep functions small and focused
- Use meaningful variable names

### Error Handling
- Use specific exception types
- Log errors with context
- Use retry decorators for transient failures
- Use circuit breakers for external services

### Performance
- Use connection pooling for HTTP requests
- Cache expensive computations
- Use vectorized operations with pandas/numpy
- Profile before optimizing

### Documentation
- Write clear docstrings
- Update README when adding features
- Create ADRs for architectural decisions
- Comment complex logic

### Git Workflow
- Create feature branches
- Write descriptive commit messages
- Keep commits atomic
- Rebase before merging

## Resources

### Documentation
- [Installation Guide](../INSTALL.md)
- [Architecture Decision Records](adr/)
- [API Documentation](api/) (generated with Sphinx)
- [Project Goals](../GOALS.md)

### External Resources
- [NFLverse Documentation](https://nflverse.nflverse.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

### Getting Help
- Check existing issues on GitHub
- Review ADRs for architectural context
- Ask in team chat
- Pair with experienced team member

## Next Steps

1. **Read the codebase**: Start with `src/utils/` and `src/data/`
2. **Run the tests**: Understand how components work
3. **Make a small change**: Fix a bug or add a test
4. **Review ADRs**: Understand architectural decisions
5. **Pair with team**: Learn from experienced developers

Welcome to the team! 🏈