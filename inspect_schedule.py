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