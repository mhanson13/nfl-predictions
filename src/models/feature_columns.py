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

from typing import List

import numpy as np
import pandas as pd

# Columns containing these substrings are treated as potential leakage targets
# and excluded from modeling features.
LEAKAGE_PATTERNS = ("score", "margin", "win", "result")

# Keep model inputs limited to pregame data or explicitly lagged/rolling
# performance windows. Raw same-week/team-season production columns are leakage
# when historical games are scored after the season has completed.
PREGAME_DIFF_PREFIXES = (
    "sched_",
    "inj_",
    "qb_",
    "news_",
    "team_news_",
    "weather_",
    "altitude_",
    "indoor_",
    "travel_",
    "timezone_",
    "rest_",
    "rivalry_",
    "crowd_noise_",
    "fan_hostility_",
)
ROLLING_PERFORMANCE_PREFIXES = (
    "pass_epa_",
    "drive_",
    "off_adj_eff_",
    "def_adj_eff_",
    "rz_",
    "pressures_",
    "sack_rate_",
)
LAGGED_FEATURE_MARKERS = ("_rolling", "_prev", "_lag")


def _is_safe_model_diff(col: str) -> bool:
    lowname = col.lower()
    if any(p in lowname for p in LEAKAGE_PATTERNS):
        return False
    if "_post" in lowname or "post_" in lowname:
        return False
    if lowname.startswith(PREGAME_DIFF_PREFIXES):
        return True
    if lowname.startswith(ROLLING_PERFORMANCE_PREFIXES):
        return any(marker in lowname for marker in LAGGED_FEATURE_MARKERS)
    return False


def make_feature_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Create *_diff columns (home - away) for numeric feature pairs.

    This mirrors the feature engineering performed during model training so
    downstream consumers (e.g., SHAP) can reproduce the exact matrix.
    """
    feat_df = df.copy()
    cols = set(feat_df.columns)
    diffs: dict[str, pd.Series] = {}

    def is_numeric(series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series)

    for col in list(cols):
        if not isinstance(col, str) or not col.endswith("_home"):
            continue
        base = col[:-5]
        away_col = base + "_away"
        if away_col not in cols:
            continue
        lowname = base.lower()
        if any(p in lowname for p in LEAKAGE_PATTERNS):
            continue
        series_home = feat_df[col]
        series_away = feat_df[away_col]
        if is_numeric(series_home) and is_numeric(series_away):
            diffs[f"{base}_diff"] = series_home.astype(float) - series_away.astype(float)

    if diffs:
        diff_df = pd.DataFrame(diffs, index=feat_df.index)
        feat_df = pd.concat([feat_df, diff_df], axis=1)
    return feat_df.copy()


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of usable modeling columns from a feature DataFrame.

    Selection heuristic (same as model training):
      * numeric *_diff columns with enough support/variance and no leakage keywords
      * numeric weather_* columns and roof_is_dome
      * prefixed wx_ engineered deltas
    """
    try:
        dedup_mask = ~pd.Index(df.columns).duplicated()
        df1 = df.loc[:, dedup_mask]
    except Exception:
        df1 = df

    def _series_from_col(name: str) -> pd.Series:
        obj = df1[name]
        if isinstance(obj, pd.DataFrame):
            first_col = obj.columns[0]
            return obj[first_col]
        if not isinstance(obj, pd.Series):
            try:
                return pd.Series(obj)
            except Exception:
                return pd.Series([pd.NA] * len(df1))
        return obj

    def _good_numeric(col: str) -> bool:
        s_obj = _series_from_col(col)
        s = pd.to_numeric(s_obj, errors="coerce")
        if s.notna().sum() < 50:
            return False
        var_val = s.var(skipna=True)
        try:
            var_float = float(np.asarray(var_val, dtype=float).item())
        except Exception:
            var_float = float("nan")
        return np.isfinite(var_float) and var_float > 1e-9

    keep: list[str] = []
    diff_candidates = [c for c in df1.columns if isinstance(c, str) and c.endswith("_diff")]
    for col in diff_candidates:
        if not _is_safe_model_diff(col):
            continue
        if _good_numeric(col):
            keep.append(col)

    weather_candidates: list[str] = []
    for col in df1.columns:
        if not isinstance(col, str):
            continue
        if col.startswith("weather_") or col == "roof_is_dome":
            s_obj = _series_from_col(col)
            if pd.api.types.is_numeric_dtype(s_obj):
                weather_candidates.append(col)
    for col in weather_candidates:
        if _good_numeric(col):
            keep.append(col)

    for col in df1.columns:
        if isinstance(col, str) and col.startswith("wx_") and _good_numeric(col):
            keep.append(col)

    seen: set[str] = set()
    ordered = [c for c in keep if not (c in seen or seen.add(c))]
    return ordered
