# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REF_DIR = DATA_DIR / "reference"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------

def _is_nested_like(x: Any) -> bool:
    """Return True if x is a structure Arrow/Parquet won't like as-is."""
    return isinstance(x, (dict, list, tuple, set, np.ndarray))


def _is_missing_scalar(x: Any) -> bool:
    """
    Missing check ONLY for scalars (safe for json/arrow).
    Any nested-like value (dict/list/ndarray/...) is treated as present.
    """
    if _is_nested_like(x):
        return False
    try:
        return pd.isna(x)
    except Exception:
        return False


def _to_json_safe(x: Any) -> Any:
    """
    Recursively convert nested structures to JSON-safe Python types:
      - numpy arrays -> lists (recursively)
      - sets/tuples  -> lists
      - numpy scalars -> native Python scalars
      - dict/list elements -> processed recursively
      - missing scalars -> None
    """
    # numpy array -> list (then recurse)
    if isinstance(x, np.ndarray):
        return _to_json_safe(x.tolist())

    # numpy scalars -> python scalars
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    # dict -> recurse values
    if isinstance(x, dict):
        return {str(k): _to_json_safe(v) for k, v in x.items()}

    # list/tuple/set -> list with recursion
    if isinstance(x, (list, tuple, set)):
        return [_to_json_safe(v) for v in list(x)]

    # scalar missing -> None
    if _is_missing_scalar(x):
        return None

    # everything else is fine (str, int, float, bool, datetime-like, None)
    return x


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any object column containing nested data to JSON strings.
    Keep plain scalar columns as-is (mapping NaN/NaT -> None).
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    # Ensure columns are unique to avoid DataFrame return on out[col]
    def _dedup_cols(cols):
        seen = {}
        new = []
        for c in [str(x) for x in cols]:
            n = seen.get(c, 0)
            if n == 0:
                new.append(c)
            else:
                new.append(f"{c}_{n}")
            seen[c] = n + 1
        return new
    if len(set(map(str, out.columns))) != len(out.columns):
        out.columns = _dedup_cols(out.columns)

    for col in out.columns:
        series = out[col]
        # If duplicate columns slipped through and we still got a DataFrame, stringify it
        if isinstance(series, pd.DataFrame):
            out[col] = series.apply(lambda row: json.dumps(_to_json_safe(row.to_dict()), ensure_ascii=False), axis=1)
            continue
        if series.dtype == "object":
            # If any nested-like values exist, JSON-encode the column
            sample = series.dropna().head(50)
            if sample.apply(_is_nested_like).any():
                out[col] = series.apply(lambda v: json.dumps(_to_json_safe(v), ensure_ascii=False)
                                          if v is not None else None)
            else:
                # Normalize missing scalars to None
                out[col] = series.apply(lambda v: None if _is_missing_scalar(v) else v)

    # Replace any remaining NaN/NaT in non-object columns with None (Arrow-friendly)
    out = out.where(pd.notna(out), None)

    return out


# ---------- public IO API ----------

def write_df(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _sanitize_for_parquet(df)
    try:
        if path.suffix == ".parquet":
            clean.to_parquet(path, index=False)
            return path
        elif path.suffix in {".csv", ""}:
            output_path = path.with_suffix(".csv")
            clean.to_csv(output_path, index=False, encoding="utf-8")
            return output_path
        else:
            output_path = path.with_suffix(".parquet")
            clean.to_parquet(output_path, index=False)
            return output_path
    except Exception as exc:
        # Fallback to CSV if Parquet still complains
        output_path = path.with_suffix(".csv")
        clean.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[io] Warning: failed to write {path}; wrote CSV fallback {output_path} ({exc})")
        return output_path


def read_df(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read a DataFrame from parquet or CSV with automatic fallback handling.
    
    For parquet files, automatically retries with fastparquet engine if the
    default PyArrow engine fails (e.g., due to schema mismatches).
    
    Args:
        path: File path to read.
        **kwargs: Additional arguments forwarded to pd.read_parquet or pd.read_csv.
        
    Returns:
        DataFrame loaded from the file.
        
    Raises:
        Exception: If both primary and fallback engines fail for parquet,
                   or if CSV reading fails.
    """
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path, **kwargs)
        except Exception as primary_exc:
            # Try fastparquet as fallback
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["engine"] = "fastparquet"
            try:
                df_fallback = pd.read_parquet(path, **fallback_kwargs)
            except Exception:
                raise primary_exc
            print(f"[io] Warning: fell back to fastparquet for {path} ({primary_exc})")
            return df_fallback
    return pd.read_csv(path, **kwargs)


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(obj), f, indent=2, ensure_ascii=False)
