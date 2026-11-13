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

## Status (2026-11-13)

### Visual Crossing refresh: fixed

`src/data/visualcrossing.py` now does three things on each run:
- re-fetches played games still holding a forecast, to get observations;
- re-fetches upcoming games within 15 days, so forecasts stay current;
- stops fetching games further out, and drops cached rows for them, because those values are climatology. The cache records `vx_fetched_at`.

Use `--no-refresh` for the old behaviour.

A live check confirmed the fix:
- TB @ CHI (played): observation 55.2°F, against a cached forecast of 47.5°F.
- MIN @ GB (upcoming): fresh forecast 38.6°F, against a cached 54.3°F.

The first run re-fetches about 280 played games (148 from 2025, 131 from 2026)
and 31 upcoming ones. After that, each run re-fetches only the upcoming
window.

### Xweather trial: built

`src/data/xweather.py` is the client for the `conditions` endpoint:
- raw responses are cached under `data/raw/xweather/`;
- requests are paced to stay under 100 a minute;
- each run has a hard request budget.

`src/evaluation/weather_source_trial.py` runs the trial:

```powershell
python -m src.evaluation.weather_source_trial --history 150   # one-off: Xweather archive vs VX observations (~150 requests)
python -m src.evaluation.weather_source_trial --live          # snapshot this slate's forecasts from all sources
```

When Xweather credentials are present, the pipeline runs the `--live` snapshot
on every run. It is capped at 40 requests and never fails the run.

The trial writes four files under `analysis/weather_trial/`:
- `history_detail.csv`
- `history_summary.csv`
- `forecast_snapshots.csv` (appended each run)
- `forecast_skill.csv`: each source's forecast error against observations, by lead time (<1d, 1-3d, 3-7d)

**Decision rule:** after 3-4 weeks of snapshots, promote the source with the
lowest kickoff temperature and wind MAE at the 1-3 day lead (when picks are
made) to primary. Then run the walk-forward A/B on the weather features.

### Xweather history check (2026-11-13, 150 outdoor games, 2017-2024)

Xweather archive compared with Visual Crossing observations, kickoff +/-3h:

| Metric | n | MAE | Bias (Xweather − VX) | Correlation |
|---|---|---|---|---|
| Kickoff temperature (°F) | 150 | 1.8 | +0.1 | 0.99 |
| Window temperature (°F) | 150 | 1.7 | +0.7 | 0.99 |
| Wind (mph) | 150 | 2.5 | **−2.0** | 0.70 |
| Gust (mph) | 105 | 6.6 | **−6.4** | 0.63 |
| Precipitation total (in) | 150 | 0.02 | 0.00 | 0.82 |
| Precipitation yes/no | 150 | — | — | 93% agree |

**How to read it:**
- Temperature and precipitation agree closely.
- Wind and gusts disagree systematically: Xweather is calmer, and the
  correlation is only about 0.7. Neither source is ground truth. Visual
  Crossing uses airport station observations; Xweather blends observations
  into its own analysis.
- Wind is the weather variable that matters most for football. For the model,
  *consistency* between training and live data matters more than which source
  is "right". If Xweather becomes primary, backfill its history too (about
  2,770 requests) so training and serving use the same wind scale. Never mix
  sources.
- `forecast_skill.csv` scores forecasts against Visual Crossing observations.
  On wind it therefore favours Visual Crossing's own scale. Judge forecast skill
  mainly on temperature and precipitation, and on wind *correlation*, not MAE.

**Spend so far:** about 256 requests, cached under `data/raw/xweather/`.

**Quota headers:**
- `X-Ratelimit-Limit-Period: 15000`; it resets on the 13th of each month.
- `X-Ratelimit-Remaining-Period` reported 14,990 after these calls, so the
  provider's count is not one-per-request. Watch the header rather than local
  counts.

## Actual venue location and kickoff time (2026-11-13)

All weather sources now resolve **where and when the game is actually played**.
The shared code is `src/utils/venues.py`, backed by
`data/reference/venues.csv`: 61 venues keyed by nflverse `stadium_id`, with
aliases, coordinates and roof type.

**Location**

| | Detail |
|---|---|
| Resolution order | Stadium name, then `stadium_id`, then home team's current stadium. The name wins over the id: the 2026 PHI @ JAX game is tagged `JAX00` but played at Tottenham |
| Games now fixed | 119 games since 2017 played away from the home team's current stadium: international games, Chargers at StubHub Center, Rams at the Coliseum, Raiders in Oakland, Super Bowls |
| Games previously missing | Rams (`LA`) and Raiders (`OAK`) home games had *no* coordinates before, so they got no fetched weather at all |

**Kickoff time**

The old Visual Crossing and NOAA resolvers parsed the 24-hour `gametime` as
`%I:%M %p`. That failed, so every game defaulted to 18:00 in the home team's
standard-time offset. 93% of cached Visual Crossing windows are off by an
hour or more (median 5 hours). Kickoff is now `gameday` + `gametime` in
US/Eastern, which is what nflverse publishes. Tomorrow.io already computed
it correctly.

**Roof**

Fixed-roof venues override the schedule label. Four 2026 international
games change (MCG, Stade de France and Allianz Arena were marked `dome`).
Retractable venues keep the per-game open/closed value. This feeds
`roof_is_dome` in `build_features`.

**What re-fetching happens**

- Every run, automatically: cached rows more than 5 km from the actual venue
  (67 games) and never-fetched past games (about 130, mostly Rams and Raiders).
- Opt-in only: the 2,355 historical rows centred on the wrong kickoff time.
  Pass `python -m src.data.visualcrossing --seasons 2017 ... --refetch-misaligned-kickoff`.
  That is about 2,350 Visual Crossing calls, once.

**Xweather**

The cache key now includes coordinates and kickoff, so a corrected place or
time never reuses an old response. The history comparison requests Xweather
at exactly the place and time Visual Crossing stored, so it stays
like-for-like. Its results above remain valid.

**Not changed yet:** travel features (`away_travel_distance_km`,
`timezone_diff_hours`) still use team home locations. For international
games both teams actually travel.
