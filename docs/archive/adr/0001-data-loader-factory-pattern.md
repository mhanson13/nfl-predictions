# ADR 0001: Data Loader Factory Pattern

## Status
Accepted

## Context
The codebase had duplicate parquet reading logic scattered across multiple files (`predict_upcoming.py`, `build_features.py`, `evaluate_predictions.py`). Each implementation had slightly different error handling, caching strategies, and fallback mechanisms. This led to:

- ~200 lines of duplicate code
- Inconsistent error handling across modules
- Difficult maintenance when changing data loading behavior
- No centralized caching strategy
- Hard to test data loading logic in isolation

## Decision
We implemented a **Data Loader Factory Pattern** with the following components:

1. **BaseDataLoader**: Abstract base class defining the interface for all data loaders
   - Provides caching, validation, and error handling
   - Enforces consistent behavior across implementations

2. **Concrete Loaders**:
   - `ParquetDataLoader`: For single parquet files with column selection and filtering
   - `CSVDataLoader`: For CSV files with dtype specification
   - `MultiFileDataLoader`: For loading and concatenating multiple files

3. **DataLoaderFactory**: Factory class for creating appropriate loaders based on file type

4. **Convenience Functions**: `load_parquet()` and `load_csv()` for simple use cases

## Consequences

### Positive
- **DRY Principle**: Eliminated ~200 lines of duplicate code
- **Consistency**: All data loading uses the same error handling and caching
- **Testability**: Easy to mock and test data loading in isolation
- **Extensibility**: New loader types can be added without modifying existing code
- **Performance**: Centralized caching reduces redundant file reads
- **Maintainability**: Changes to data loading logic only need to be made in one place

### Negative
- **Learning Curve**: Developers need to understand the factory pattern
- **Indirection**: One more layer of abstraction to navigate
- **Migration Effort**: Existing code needs to be updated to use new loaders

### Neutral
- **File Location**: `src/utils/data_loader.py` (344 lines)
- **Dependencies**: Integrates with existing `src.utils.cache` and `src.utils.schemas`

## Implementation Notes
- All existing duplicate `_read_parquet_with_fallback()` functions were replaced with `read_df()` from `src.utils.io`
- The factory automatically selects the appropriate loader based on file extension
- Caching is optional and can be enabled per-loader instance
- Schema validation is integrated but optional

## Alternatives Considered

### 1. Keep Duplicate Code
**Rejected**: Violates DRY principle and makes maintenance difficult

### 2. Simple Utility Functions
**Rejected**: Doesn't provide enough flexibility for different loading strategies

### 3. Strategy Pattern Without Factory
**Rejected**: Factory pattern provides better encapsulation and easier instantiation

## References
- Implementation: `src/utils/data_loader.py`
- Tests: `tests/test_data_loaders.py`
- Related ADR: [0002-abstract-base-fetchers.md](0002-abstract-base-fetchers.md)