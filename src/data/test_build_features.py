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

from src.utils.io import PROC_DIR, read_df
import pandas as pd
p = PROC_DIR / "matchup_features.parquet"
df = read_df(p)
home_away_pairs = sorted({c[:-5] for c in df.columns if c.endswith("_home")} & {c[:-5] for c in df.columns if c.endswith("_away")})
nums = [c for c in df.columns if c.endswith(("_home","_away")) and pd.api.types.is_numeric_dtype(df[c])]
print("home/away numeric columns (sample):", nums[:15])
print("pair count:", len(home_away_pairs))
print("pairs (sample):", home_away_pairs[:15])
print("rows:", len(df))

