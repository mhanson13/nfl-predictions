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

import pandas as pd
from pathlib import Path
f = Path('data/processed/matchup_features.parquet')
df = pd.read_parquet(f)
# Filter season 2025 week 7
mask = (df['season']==2025) & (df['week']==7)
d = df.loc[mask].copy()
print('rows', len(d))
wx_cols = [c for c in d.columns if c.startswith('weather_') or c in ('kickoff','roof','venue_lat','venue_lon','roof_is_dome')]
print('weather cols present:', wx_cols)
print('non-null counts:\n', d[wx_cols].notna().sum().sort_values(ascending=False))
print(d[['home_team','away_team','game_id','roof','kickoff','weather_temp_f']].to_string(index=False))