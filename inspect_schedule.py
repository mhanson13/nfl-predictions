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
from src.utils.io import RAW_DIR
p1 = RAW_DIR / 'espn_schedule.parquet'
p2 = RAW_DIR / 'nfl_schedules.parquet'
for p in [p1,p2]:
    if p.exists():
        df = pd.read_parquet(p)
        print('----', p.name, 'rows', len(df))
        print('cols:', list(df.columns))
        d = df[(df.get('season')==2025) & (df.get('week')==7)]
        print('sample 2025 wk7:')
        print(d.head(10).to_string(index=False))