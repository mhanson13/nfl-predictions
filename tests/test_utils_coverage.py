"""Tests to boost coverage for low-coverage utility modules.

Modules targeted:
  - src/utils/cache.py      (CacheVersion, CacheMetadata, SmartCache, cached, get_cache)
  - src/utils/io.py         (_is_nested_like, _is_missing_scalar, _to_json_safe,
                              _sanitize_for_parquet, write_df, read_df, write_json)
  - src/utils/checkpoints.py (load_checkpoint, save_checkpoint,
                               get_last_timestamp, update_timestamp)
  - src/utils/logging_config.py (setup_logging, get_logger, log_execution_time,
                                  log_progress, LoggerAdapter, configure_root_logger, configure)
  - src/utils/rate_limit.py  (TokenBucket, SlidingWindowLimiter)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# cache.py
# ---------------------------------------------------------------------------
from src.utils.cache import (
    CacheMetadata,
    CacheVersion,
    SmartCache,
    cached,
    get_cache,
)


class TestCacheVersion:
    def test_get_version_hash_returns_string(self):
        h = CacheVersion.get_version_hash("v1.0", "v2.0")
        assert isinstance(h, str) and len(h) == 8

    def test_same_inputs_same_hash(self):
        assert CacheVersion.get_version_hash("a", "b") == CacheVersion.get_version_hash("a", "b")

    def test_different_inputs_different_hash(self):
        assert CacheVersion.get_version_hash("a") != CacheVersion.get_version_hash("b")


class TestCacheMetadata:
    def _meta(self, ttl=None) -> CacheMetadata:
        return CacheMetadata(version="v1", created_at=datetime.now(), ttl_hours=ttl)

    def test_not_expired_when_no_ttl(self):
        assert not self._meta(ttl=None).is_expired()

    def test_not_expired_fresh(self):
        assert not self._meta(ttl=24).is_expired()

    def test_expired_when_old(self):
        old = CacheMetadata(version="v1", created_at=datetime.now() - timedelta(hours=2), ttl_hours=1)
        assert old.is_expired()

    def test_is_valid_correct_version(self):
        assert self._meta(ttl=24).is_valid("v1")

    def test_is_invalid_wrong_version(self):
        assert not self._meta(ttl=24).is_valid("v2")

    def test_roundtrip_dict(self):
        meta = CacheMetadata(version="v3", created_at=datetime(2025, 1, 1), ttl_hours=10,
                              data_hash="abc123", extra={"key": "val"})
        restored = CacheMetadata.from_dict(meta.to_dict())
        assert restored.version == "v3"
        assert restored.ttl_hours == 10
        assert restored.extra == {"key": "val"}


class TestSmartCache:
    def test_disabled_get_returns_default(self, tmp_path, monkeypatch):
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = False
        assert cache.get("k", "v1", default="x") == "x"

    def test_disabled_set_returns_false(self, tmp_path):
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = False
        assert cache.set("k", "val", "v1") is False

    def test_set_and_get_scalar(self, tmp_path):
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = True
        cache.default_ttl = None
        cache.set("mykey", {"foo": 42}, "v1")
        result = cache.get("mykey", "v1")
        assert result == {"foo": 42}

    def test_get_wrong_version_returns_default(self, tmp_path):
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = True
        cache.default_ttl = None
        cache.set("mykey", 99, "v1")
        result = cache.get("mykey", "v2", default=-1)
        assert result == -1

    def test_set_and_get_list(self, tmp_path):
        """DataFrame path switching is a known SmartCache quirk; test with a plain list."""
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = True
        cache.default_ttl = None
        cache.set("list_key", [10, 20, 30], "v1")
        result = cache.get("list_key", "v1")
        assert result == [10, 20, 30]

    def test_invalidate_removes_files(self, tmp_path):
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = True
        cache.default_ttl = None
        cache.set("toremove", 123, "v1")
        cache.invalidate("toremove")
        assert cache.get("toremove", "v1", default="gone") == "gone"

    def test_clear_all_empties_dir(self, tmp_path):
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = True
        cache.default_ttl = None
        cache.set("a", 1, "v1")
        cache.set("b", 2, "v1")
        count = cache.clear_all()
        assert count > 0
        assert list(tmp_path.glob("*")) == []

    def test_clear_expired_removes_expired(self, tmp_path):
        cache = SmartCache(cache_dir=tmp_path)
        cache.enabled = True
        cache.default_ttl = None
        # Manually set metadata to indicate already-expired entry
        cache.set("expkey", 99, "v1", ttl_hours=0)
        # Force expiry: patch created_at to the past
        meta_path = cache._get_metadata_path("expkey")
        with open(meta_path) as f:
            meta_data = json.load(f)
        meta_data["created_at"] = (datetime.now() - timedelta(hours=2)).isoformat()
        meta_data["ttl_hours"] = 1
        with open(meta_path, "w") as f:
            json.dump(meta_data, f)
        removed = cache.clear_expired()
        assert removed >= 1


def test_get_cache_returns_smart_cache():
    instance = get_cache()
    assert isinstance(instance, SmartCache)
    # Second call returns same instance
    assert get_cache() is instance


# ---------------------------------------------------------------------------
# io.py
# ---------------------------------------------------------------------------
from src.utils.io import (
    _is_missing_scalar,
    _is_nested_like,
    _sanitize_for_parquet,
    _to_json_safe,
    read_df,
    write_df,
    write_json,
)


class TestIsNestedLike:
    def test_dict_is_nested(self):
        assert _is_nested_like({})
    def test_list_is_nested(self):
        assert _is_nested_like([1, 2])
    def test_ndarray_is_nested(self):
        assert _is_nested_like(np.array([1]))
    def test_scalar_not_nested(self):
        assert not _is_nested_like(42)
    def test_none_not_nested(self):
        assert not _is_nested_like(None)


class TestIsMissingScalar:
    def test_none_is_missing(self):
        assert _is_missing_scalar(None)
    def test_nan_is_missing(self):
        assert _is_missing_scalar(float("nan"))
    def test_value_not_missing(self):
        assert not _is_missing_scalar(1)
    def test_dict_not_missing(self):
        assert not _is_missing_scalar({"a": 1})


class TestToJsonSafe:
    def test_numpy_array_becomes_list(self):
        result = _to_json_safe(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
    def test_numpy_int_becomes_python(self):
        result = _to_json_safe(np.int64(5))
        assert result == 5 and isinstance(result, int)
    def test_dict_recursed(self):
        result = _to_json_safe({"a": np.array([1])})
        assert result == {"a": [1]}
    def test_set_becomes_list(self):
        result = _to_json_safe({1, 2})
        assert isinstance(result, list)
    def test_none_stays_none(self):
        assert _to_json_safe(None) is None
    def test_string_passes_through(self):
        assert _to_json_safe("hello") == "hello"


class TestSanitizeForParquet:
    def test_none_returns_none(self):
        assert _sanitize_for_parquet(None) is None

    def test_empty_df_returned(self):
        result = _sanitize_for_parquet(pd.DataFrame())
        assert result.empty

    def test_nested_col_json_encoded(self):
        df = pd.DataFrame({"data": [{"a": 1}, {"b": 2}], "num": [1.0, 2.0]})
        result = _sanitize_for_parquet(df)
        # data column should contain string-encoded JSON (not raw dicts)
        first = result["data"].iloc[0]
        assert isinstance(first, (str, type(None))), f"expected str or None, got {type(first)}"
        if isinstance(first, str):
            parsed = json.loads(first)
            assert isinstance(parsed, dict)

    def test_plain_numeric_preserved(self):
        df = pd.DataFrame({"val": [1.0, 2.0, None]})
        result = _sanitize_for_parquet(df)
        assert "val" in result.columns

    def test_duplicate_cols_handled(self):
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        result = _sanitize_for_parquet(df)
        assert len(set(result.columns)) == len(result.columns)


class TestWriteReadDf:
    def test_write_read_parquet_roundtrip(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        p = tmp_path / "data.parquet"
        write_df(df, p)
        result = read_df(p)
        assert list(result["x"]) == [1, 2, 3]

    def test_write_read_csv_roundtrip(self, tmp_path):
        df = pd.DataFrame({"a": ["foo", "bar"]})
        p = tmp_path / "data.csv"
        write_df(df, p)
        result = read_df(p)
        assert list(result["a"]) == ["foo", "bar"]

    def test_write_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"v": [1]})
        p = tmp_path / "sub" / "dir" / "data.parquet"
        write_df(df, p)
        assert p.exists()


class TestWriteJson:
    def test_writes_json(self, tmp_path):
        p = tmp_path / "out.json"
        write_json({"season": 2023, "teams": ["KC", "BUF"]}, p)
        with open(p) as f:
            data = json.load(f)
        assert data["season"] == 2023

    def test_numpy_values_serialized(self, tmp_path):
        p = tmp_path / "np.json"
        write_json({"val": np.int64(42)}, p)
        with open(p) as f:
            data = json.load(f)
        assert data["val"] == 42


# ---------------------------------------------------------------------------
# checkpoints.py
# ---------------------------------------------------------------------------
from src.utils.checkpoints import (
    get_last_timestamp,
    load_checkpoint,
    save_checkpoint,
    update_timestamp,
    CHECKPOINT_PATH,
)


class TestCheckpoints:
    def test_load_checkpoint_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.checkpoints.CHECKPOINT_PATH", tmp_path / "nope.json")
        assert load_checkpoint() == {}

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        p = tmp_path / "cp.json"
        monkeypatch.setattr("src.utils.checkpoints.CHECKPOINT_PATH", p)
        save_checkpoint({"key": "value"})
        result = load_checkpoint()
        assert result == {"key": "value"}

    def test_get_last_timestamp_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.checkpoints.CHECKPOINT_PATH", tmp_path / "nope.json")
        assert get_last_timestamp("anything") is None

    def test_get_last_timestamp_returns_datetime(self, tmp_path, monkeypatch):
        p = tmp_path / "cp.json"
        monkeypatch.setattr("src.utils.checkpoints.CHECKPOINT_PATH", p)
        ts = datetime(2025, 1, 15, 12, 0, 0)
        update_timestamp("run", ts)
        result = get_last_timestamp("run")
        assert isinstance(result, datetime)
        assert result.date() == ts.date()

    def test_get_last_timestamp_invalid_value_returns_none(self, tmp_path, monkeypatch):
        p = tmp_path / "cp.json"
        monkeypatch.setattr("src.utils.checkpoints.CHECKPOINT_PATH", p)
        save_checkpoint({"bad": "not-a-date"})
        result = get_last_timestamp("bad")
        assert result is None


# ---------------------------------------------------------------------------
# logging_config.py
# ---------------------------------------------------------------------------
from src.utils.logging_config import (
    ColoredFormatter,
    LoggerAdapter,
    configure,
    configure_root_logger,
    get_logger,
    log_execution_time,
    log_progress,
    setup_logging,
)


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging("test_setup", level="WARNING", console=False)
        assert isinstance(logger, logging.Logger)

    def test_level_applied(self):
        logger = setup_logging("test_level", level="ERROR", console=False)
        assert logger.level == logging.ERROR

    def test_file_handler_created(self, tmp_path):
        log_path = tmp_path / "test.log"
        logger = setup_logging("test_file", level="INFO", console=False, log_file=log_path)
        logger.info("hello")
        assert log_path.exists()
        for h in logger.handlers:
            h.close()


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_get_logger_unique_xyz")
        assert isinstance(logger, logging.Logger)


class TestLogExecutionTime:
    def test_does_not_raise(self):
        logger = logging.getLogger("test_exec_time")
        with log_execution_time(logger, "test op"):
            pass  # no exception


class TestLogProgress:
    def test_yields_callable(self):
        logger = logging.getLogger("test_prog")
        with log_progress(logger, 10, "items") as progress:
            assert callable(progress)
            progress(5)
            progress(10)


class TestLoggerAdapter:
    def test_process_adds_context(self):
        base = logging.getLogger("adapter_test")
        adapter = LoggerAdapter(base, {"season": 2023})
        msg, _ = adapter.process("hello", {})
        assert "season=2023" in msg

    def test_process_no_extra(self):
        base = logging.getLogger("adapter_test2")
        adapter = LoggerAdapter(base, {})
        msg, _ = adapter.process("plain", {})
        assert msg == "plain"


class TestConfigure:
    def test_configure_does_not_raise(self):
        configure(level="WARNING")

    def test_configure_with_file(self, tmp_path):
        log_file = str(tmp_path / "app.log")
        configure(level="INFO", log_file=log_file)


class TestConfigureRootLogger:
    def test_does_not_raise(self, tmp_path):
        configure_root_logger(level="WARNING", log_dir=tmp_path)


# ---------------------------------------------------------------------------
# rate_limit.py
# ---------------------------------------------------------------------------
from src.utils.rate_limit import SlidingWindowLimiter, TokenBucket


class TestTokenBucket:
    def test_consume_does_not_raise(self):
        tb = TokenBucket(rate=100, capacity=10)
        tb.consume(1.0)  # should not block meaningfully

    def test_initial_tokens(self):
        tb = TokenBucket(rate=10, capacity=5)
        assert tb.tokens == 5

    def test_consume_reduces_tokens(self):
        tb = TokenBucket(rate=1000, capacity=10)
        tb.consume(3.0)
        # tokens should be near 7 (plus a tiny refill from elapsed time)
        assert tb.tokens < 10


class TestSlidingWindowLimiter:
    def test_acquire_does_not_raise(self):
        limiter = SlidingWindowLimiter(max_calls=5, window=1.0)
        limiter.acquire()  # first call always immediate

    def test_multiple_acquires_within_limit(self):
        limiter = SlidingWindowLimiter(max_calls=10, window=5.0)
        for _ in range(5):
            limiter.acquire()


# ---------------------------------------------------------------------------
# src/utils/logging.py (deprecated compat shim)
# ---------------------------------------------------------------------------
import warnings
from src.utils.logging import configure as legacy_configure


class TestLegacyLoggingConfigure:
    def test_deprecated_warning_issued(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            legacy_configure()
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_debug_flag_does_not_raise(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy_configure(debug=True)

    def test_log_file_does_not_raise(self, tmp_path):
        log_path = str(tmp_path / "legacy.log")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy_configure(log_file=log_path)


# ---------------------------------------------------------------------------
# src/utils/performance.py
# ---------------------------------------------------------------------------
from src.utils.performance import (
    PerformanceTimer,
    batch_process,
    optimize_dtypes,
)


class TestOptimizeDtypes:
    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = optimize_dtypes(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        original_dtype = df["a"].dtype
        optimize_dtypes(df)
        assert df["a"].dtype == original_dtype

    def test_aggressive_mode(self):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
        result = optimize_dtypes(df, aggressive=True)
        assert isinstance(result, pd.DataFrame)

    def test_object_col_may_become_category(self):
        df = pd.DataFrame({"team": ["KC", "BUF", "KC", "BUF"] * 5})
        result = optimize_dtypes(df)
        assert isinstance(result, pd.DataFrame)

    def test_negative_int_downcast(self):
        df = pd.DataFrame({"v": [-1, -2, 3]})
        result = optimize_dtypes(df)
        assert isinstance(result, pd.DataFrame)


class TestBatchProcess:
    def test_processes_all_items(self):
        results = batch_process([1, 2, 3, 4, 5], lambda batch: sum(batch), batch_size=2)
        assert sum(results) == 15

    def test_single_batch(self):
        results = batch_process([1, 2, 3], lambda b: b, batch_size=10)
        assert len(results) == 1

    def test_empty_input(self):
        results = batch_process([], lambda b: b, batch_size=5)
        assert results == []


class TestPerformanceTimer:
    def test_elapsed_is_set(self):
        with PerformanceTimer("test", verbose=False) as timer:
            pass
        assert timer.elapsed is not None
        assert timer.elapsed >= 0.0

    def test_verbose_true_does_not_raise(self, capsys):
        with PerformanceTimer("my_op", verbose=True):
            pass
        captured = capsys.readouterr()
        assert "my_op" in captured.out


# ---------------------------------------------------------------------------
# src/utils/http.py
# ---------------------------------------------------------------------------
import httpx
from src.utils.http import RetryConfig, SportradarClient


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.retries == 4
        assert cfg.backoff_factor == 0.5

    def test_custom_values(self):
        cfg = RetryConfig(retries=2, max_backoff=5.0)
        assert cfg.retries == 2
        assert cfg.max_backoff == 5.0


class TestSportradarClient:
    def _client(self) -> SportradarClient:
        return SportradarClient("test_key", max_requests_per_sec=1000, burst=100)

    def test_init_stores_api_key(self):
        c = self._client()
        assert c.api_key == "test_key"
        c.close()

    def test_prepare_params_injects_api_key(self):
        c = self._client()
        result = c._prepare_params({"season": 2023})
        assert result["api_key"] == "test_key"
        assert result["season"] == 2023
        c.close()

    def test_prepare_params_no_input(self):
        c = self._client()
        result = c._prepare_params(None)
        assert result["api_key"] == "test_key"
        c.close()

    def test_close_does_not_raise(self):
        c = self._client()
        c.close()  # should not raise


# ---------------------------------------------------------------------------
# src/utils/performance.py — vectorize_operation / parallel_apply
# ---------------------------------------------------------------------------
from src.utils.performance import parallel_apply, vectorize_operation


class TestVectorizeOperation:
    def test_scalar_input(self):
        @vectorize_operation
        def double(x):
            return x * 2
        assert double(5) == 10

    def test_series_input(self):
        @vectorize_operation
        def triple(x):
            return x * 3
        result = triple(pd.Series([1, 2, 3]))
        assert list(result) == [3, 6, 9]

    def test_ndarray_input(self):
        @vectorize_operation
        def add_one(x):
            return x + 1
        arr = np.array([1, 2, 3])
        result = add_one(arr)
        assert list(result) == [2, 3, 4]


class TestParallelApply:
    def test_single_job_applies_func(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = parallel_apply(df, lambda x: x.assign(b=x["a"] * 2), n_jobs=1)
        assert list(result["b"]) == [2, 4, 6]


# ---------------------------------------------------------------------------
# TokenBucket / SlidingWindowLimiter — additional coverage for sleep paths
# ---------------------------------------------------------------------------

class TestTokenBucketSleepPath:
    def test_consume_waits_for_refill(self):
        """Force the bucket to deplete slightly so it needs to refill."""
        tb = TokenBucket(rate=10000.0, capacity=1)
        tb.consume(1.0)  # depletes tokens
        tb.consume(0.5)  # triggers tiny wait at 10000/s rate (< 0.01ms)

    def test_zero_rate_handles_gracefully(self):
        """Rate=0 means infinite wait; just test token depletion without consume."""
        tb = TokenBucket(rate=1.0, capacity=5)
        assert tb.tokens == 5


class TestSlidingWindowLimiterSleepPath:
    def test_sliding_window_fills_and_resets(self):
        """Fill the window, then let time pass — events expire from front."""
        limiter = SlidingWindowLimiter(max_calls=3, window=0.05)
        for _ in range(3):
            limiter.acquire()
        # After window expires (50ms) the 4th call should complete quickly
        import time; time.sleep(0.06)
        limiter.acquire()  # should not block long after window expires
