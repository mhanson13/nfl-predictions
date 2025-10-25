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