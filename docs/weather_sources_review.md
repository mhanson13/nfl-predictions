# Weather sources review (2026-11-13)

**Scope:** decide which weather provider should be primary. Nothing has been
removed.

**Question:** we call four weather sources. Is Xweather (15,000 free requests a
month, 100 a minute) a better primary source?

## What we use today

`build_features._load_weather_sources` keeps one source per game, in this
priority order: **Visual Crossing > Tomorrow.io > NOAA**. Climatological
normals (`weather_fallback.py`) fill whatever is left.

| Source | Module | Role | What the data actually shows |
|---|---|---|---|
| Visual Crossing (paid) | `src/data/visualcrossing.py` (`--only-missing`) | Highest priority, history and live | 2017–2024: 100% observations (`obs`). **2025: 148 of 258 games are forecasts; 2026: 161 of 162 are forecasts**, including games already played. Rows are fetched once and never refreshed. |
| Tomorrow.io (free) | `src/data/weather.py` | Live forecast for the publish window | Good week 10 hourly forecasts (e.g. GB 42°F), but **never selected**: Visual Crossing outranks it for every game. Hit HTTP 429 rate limits on 2 of 16 calls in week 10. |
| NOAA | `src/data/noaa.py` | Lowest priority | 6,290 rows, but **temperature and wind are filled for under 1% of rows**. In practice it contributes venue context only. |
| Climatology | `src/data/weather_fallback.py` | Last resort | Filled 214 games in the week 10 build (international/neutral sites and gaps). |

**Why it matters:** weather features carry about 15% of the residual model's
SHAP importance. `weather_rain_index_diff` is the third most important feature.

## Data-quality finding (paused for a decision)

Visual Crossing week 10 values look like **seasonal normals, not forecasts**.
They were pulled more than 15 days ahead and never refreshed.

| Game | Visual Crossing (used) | Tomorrow.io (unused) |
|---|---|---|
| MIN @ GB | 53.8°F, wind 7.0 | 42.1°F, wind 1.1 |
| HOU @ CLE | 59.4°F, wind 12.6 | 44.6°F, wind 10.4 |
| PIT @ CIN | 61.5°F, wind 10.2 | 43.1°F, wind 2.3 |

The same stale forecasts stay attached to completed 2025–2026 games. The model
therefore trains on observations through 2024, mixed forecasts and normals for
2025–26, and scores live games on normals. That is a train/serve mismatch on
an important feature family.

## Xweather

- The **forecasts** endpoint (`/forecasts/{lat,lon}`, `filter=1hr`) gives hourly temperature, feels-like, wind speed, gusts and direction, `pop`, `precipIN`, `snowIN`, humidity, visibility and weather codes, up to 15 days ahead.
- The **conditions** endpoint (`/conditions/{lat,lon}?from=&to=`) covers **2004 through 15 days ahead**, hourly, from one model. Each day of a range costs one access.
- Limits come back in `X-RateLimit-*` headers; going over returns HTTP 429.

**Request budget:**

| Use | Accesses |
|---|---|
| History backfill, 2017–2026 (~2,770 games, one day each) | ~2,770, once |
| Live, 14–16 games × ~3 runs a week | ~200 a month |

Both fit inside 15,000 a month. Spreading the backfill over two or three days
keeps it well under the per-minute limit.

**Main attraction:** one provider and one schema for both training (past games)
and serving (forecasts). That removes the observation/forecast mismatch above.

**Unknowns, to verify with a key:**
- whether the free tier includes historical `conditions` (the docs do not say);
- how accurate Xweather is against Visual Crossing observations on 2017–2024 games.

## Recommendation

1. **Fix Visual Crossing refresh now** (small and independent of the provider choice):
   - re-pull games whose stored row is `fcst` once kickoff has passed, to get observations;
   - re-pull upcoming games within the forecast window on every run, instead of `--only-missing` skipping them.
2. **Trial Xweather** once `XWEATHER_CLIENT_ID` / `XWEATHER_CLIENT_SECRET` are in `secrets.env`:
   - fetch a sample of about 200 historical games plus the live slate;
   - compare with Visual Crossing observations (temperature, wind, precipitation error);
   - run a walk-forward A/B on the weather features.
3. **Then pick the primary** by measured agreement and walk-forward Brier.
   - Tomorrow.io and NOAA currently add nothing to the features. They can be demoted later; nothing is removed yet.
