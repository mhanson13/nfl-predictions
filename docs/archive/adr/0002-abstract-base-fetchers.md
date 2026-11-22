# ADR 0002: Abstract Base Classes for Data Fetchers

## Status
Accepted

## Context
The codebase has multiple data fetchers for different sources (ESPN, Sportradar, NFLverse, weather APIs, etc.). Each fetcher had its own implementation with:

- Inconsistent error handling patterns
- Different retry strategies
- No standardized caching approach
- Varying logging practices
- Difficult to test in isolation
- No clear contract for what a "fetcher" should do

This made it hard to:
- Add new data sources
- Maintain existing fetchers
- Test data fetching logic
- Ensure consistent behavior across sources

## Decision
We implemented **Abstract Base Classes (ABCs)** for data fetchers with a clear hierarchy:

```
BaseDataFetcher (abstract)
├── APIDataFetcher (abstract)
│   ├── ESPNFetcher
│   ├── SportsradarFetcher
│   └── WeatherAPIFetcher
├── WebScraperFetcher (abstract)
│   ├── NFLComScraper
│   └── YahooScraper
├── FileDataFetcher
└── BatchDataFetcher (abstract)
```

### Key Components

1. **BaseDataFetcher**: Core abstract class
   - Defines `_fetch_raw()` abstract method
   - Provides caching, validation, error handling
   - Standardizes logging and metrics
   - Returns `FetchResult` dataclass

2. **APIDataFetcher**: For REST APIs
   - Handles authentication (API keys, OAuth)
   - Builds URLs with query parameters
   - Manages rate limiting
   - Provides retry logic

3. **WebScraperFetcher**: For web scraping
   - Manages user agents
   - Handles session management
   - Respects robots.txt
   - Implements polite delays

4. **FileDataFetcher**: For file-based sources
   - Lists files by pattern
   - Handles multiple file formats
   - Manages file system operations

5. **BatchDataFetcher**: For batch processing
   - Splits items into batches
   - Processes batches sequentially
   - Aggregates results

## Consequences

### Positive
- **Consistency**: All fetchers follow the same interface
- **Testability**: Easy to mock and test with concrete implementations
- **Extensibility**: New fetchers inherit common functionality
- **Maintainability**: Changes to common logic only need to be made once
- **Type Safety**: Abstract methods enforce implementation
- **Documentation**: Clear contract for what fetchers must provide
- **Separation of Concerns**: Each fetcher type handles its specific concerns

### Negative
- **Complexity**: More classes and inheritance to understand
- **Overhead**: Abstract base classes add some runtime overhead
- **Migration**: Existing fetchers need to be refactored to use new base classes

### Neutral
- **File Location**: `src/data/base_fetcher.py` (382 lines)
- **Dependencies**: Uses `src.utils.cache`, `src.utils.logging_config`, `src.utils.http`

## Implementation Details

### FetchResult Dataclass
```python
@dataclass
class FetchResult:
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### Usage Example
```python
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

## Design Patterns Used

1. **Template Method**: `fetch()` defines the algorithm, subclasses implement `_fetch_raw()`
2. **Factory Method**: Subclasses create their own fetch implementations
3. **Strategy**: Different fetcher types use different strategies
4. **Decorator**: Can be combined with `@retry` and `@circuit_breaker`

## Integration with Other Patterns

- **Retry Decorator** (ADR 0003): Fetchers can use `@retry` on `_fetch_raw()`
- **Circuit Breaker** (ADR 0004): Fetchers can use `@circuit_breaker` for resilience
- **Connection Pooling** (ADR 0005): API fetchers use pooled sessions

## Testing Strategy

- Unit tests for each base class
- Mock implementations for testing
- Contract tests to ensure API compliance
- Integration tests with real data sources

## Alternatives Considered

### 1. No Abstraction
**Rejected**: Leads to code duplication and inconsistency

### 2. Composition Over Inheritance
**Considered**: Could use composition, but inheritance provides clearer hierarchy and type relationships

### 3. Protocol/Interface Only
**Rejected**: Python's ABC provides better enforcement and documentation

## Migration Path

1. Create base classes (✅ Complete)
2. Update existing fetchers to inherit from base classes
3. Add tests for new base classes (✅ Complete)
4. Deprecate old fetcher implementations
5. Remove deprecated code after migration period

## References
- Implementation: `src/data/base_fetcher.py`
- Tests: `tests/test_data_fetchers.py`
- Related ADRs:
  - [0001-data-loader-factory-pattern.md](0001-data-loader-factory-pattern.md)
  - [0003-retry-pattern.md](0003-retry-pattern.md)
  - [0004-circuit-breaker-pattern.md](0004-circuit-breaker-pattern.md)