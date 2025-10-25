import pandas as pd
from pathlib import Path
p = Path('data/processed/weather_games.parquet')
df = pd.read_parquet(p)
print('rows', len(df))
print('columns', list(df.columns))
print(df.head(20).to_string(index=False))