"""Coverage tests for small utility modules.

Modules targeted:
  - src/utils/io.py         (_is_nested_like, _is_missing_scalar, _to_json_safe,
                              _sanitize_for_parquet, write_df, read_df, write_json)
  - src/utils/logging_config.py (setup_logging, get_logger, log_execution_time,
                                  log_progress, LoggerAdapter, configure_root_logger, configure)
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# io.py
# ---------------------------------------------------------------------------
from src.utils.io import (
    _is_missing_scalar,
    _is_nested_like,
    _sanitize_for_parquet,
    _to_json_safe,
    df_path_exists,
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

    def test_read_df_prefers_newer_csv_fallback_for_parquet_path(self, tmp_path):
        parquet_path = tmp_path / "data.parquet"
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": [1], "y": ["old"]}).to_parquet(parquet_path, index=False)
        pd.DataFrame({"x": [2], "y": ["fresh"]}).to_csv(csv_path, index=False)
        os.utime(parquet_path, (1000, 1000))
        os.utime(csv_path, (2000, 2000))

        result = read_df(parquet_path, columns=["x"])

        assert result.columns.tolist() == ["x"]
        assert result["x"].tolist() == [2]

    def test_read_df_uses_csv_fallback_when_parquet_missing(self, tmp_path):
        parquet_path = tmp_path / "data.parquet"
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": [3]}).to_csv(csv_path, index=False)

        assert df_path_exists(parquet_path)
        result = read_df(parquet_path)

        assert result["x"].tolist() == [3]


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
# logging_config.py
# ---------------------------------------------------------------------------
from src.utils.logging_config import (
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
