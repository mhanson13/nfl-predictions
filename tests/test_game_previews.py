"""Game previews: schedule + market data only, exact columns, odds sources and reuse (no network)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from src.data import game_previews as gp

NOW = datetime(2026, 12, 8, 10, 0, tzinfo=timezone.utc)  # Tuesday after week 13 is graded


def _schedule():
    rows = [
        # game_id, gameday, gametime, away, home, location, roof, spread_line, total_line, home_ml, away_ml
        ("2026_14_MIN_NE", "2026-12-10", "20:15", "MIN", "NE", "Home", "outdoors", 3.0, 44.5, -155, 130),
        ("2026_14_TB_BAL", "2026-12-13", "13:00", "TB", "BAL", "Home", "outdoors", 6.5, 47.5, -280, 230),
        ("2026_14_LAC_LV", "2026-12-13", "16:05", "LAC", "LV", "Home", "dome", -2.5, 42.0, 120, -142),
        ("2026_14_BUF_GB", "2026-12-13", "20:20", "BUF", "GB", "Home", None, np.nan, np.nan, np.nan, np.nan),
        ("2026_14_PIT_JAX", "2026-12-14", "20:15", "PIT", "JAX", "Neutral", "outdoors", 1.0, 41.0, -118, -102),
        ("2026_13_KC_LA", "2026-12-03", "20:15", "KC", "LA", "Home", "dome", 1.5, 45.5, -120, 100),
    ]
    df = pd.DataFrame(rows, columns=["game_id", "gameday", "gametime", "away_team", "home_team", "location", "roof",
                                     "spread_line", "total_line", "home_moneyline", "away_moneyline"])
    df["season"] = 2026
    df["week"] = df["game_id"].str.slice(5, 7).astype(int)
    df["game_type"] = "REG"
    df["stadium"] = df["home_team"].map({"NE": "Gillette Stadium", "BAL": "M&T Bank Stadium",
                                          "LV": "Allegiant Stadium", "GB": "Lambeau Field"})  # JAX/LA: unresolved
    return df


def _api(**overrides):
    rows = [
        {"home_abbr": "NE", "away_abbr": "MIN", "commence": pd.Timestamp("2026-12-11T01:15:00Z"),
         "home_ml": -150.0, "away_ml": 126.0, "spread_home": -2.5, "total": 44.0},
        {"home_abbr": "BAL", "away_abbr": "TB", "commence": pd.Timestamp("2026-12-13T18:00:00Z"),
         "home_ml": -290.0, "away_ml": 235.0, "spread_home": -6.5, "total": 47.5},
        # Same teams a month earlier: outside the match window, must not be used.
        {"home_abbr": "LV", "away_abbr": "LAC", "commence": pd.Timestamp("2026-11-08T21:05:00Z"),
         "home_ml": 999.0, "away_ml": -999.0, "spread_home": 9.5, "total": 99.0},
    ]
    return pd.DataFrame(rows)


def _build(api=None, **kw):
    sched = _schedule()
    return gp.build_previews(sched, 2026, 14, api_odds=_api() if api is None else api,
                             odds_captured_at=NOW, generated_at=NOW, **kw)


def test_columns_are_exactly_the_agreed_schema_in_order():
    frame = _build()
    assert list(frame.columns) == gp.PREVIEW_COLUMNS
    assert gp.PREVIEW_COLUMNS == [
        "season", "week", "game_id", "home_team", "away_team", "kickoff", "pick_expected_at", "roof",
        "venue_name", "venue_city", "neutral_site", "spread_home", "total", "home_ml", "away_ml",
        "odds_source", "odds_captured_at", "generated_at",
    ]
    assert len(frame) == 5 and set(frame["week"]) == {14}  # week 13 game excluded


def test_no_model_like_columns_and_validation_refuses_them():
    frame = _build()
    assert not [c for c in frame.columns if gp.MODEL_COLUMN_RE.search(c)]
    for extra in ("home_win_prob", "pick", "pred_home_margin", "market_moneyline_prob_edge", "confidence"):
        with pytest.raises(gp.PreviewError):
            gp.validate_preview_frame(frame.assign(**{extra: 0.5}))
    with pytest.raises(gp.PreviewError):
        gp.validate_preview_frame(frame[list(reversed(gp.PREVIEW_COLUMNS))])


def test_builder_never_reads_predictions():
    source = Path(gp.__file__).read_text(encoding="utf-8")
    assert "predictions.csv" not in source
    assert "predict_upcoming" not in source.replace("``predict_upcoming``'s", "")  # docstring mention only
    # The only predictions/ path is the preview output directory.
    assert re.findall(r'Path\("predictions[^"]*"\)', source) == ['Path("predictions/previews")']


def test_odds_prefer_the_odds_api_then_nflverse_then_blank():
    frame = _build().set_index("game_id")
    ne = frame.loc["2026_14_MIN_NE"]
    assert (ne.spread_home, ne.total, ne.home_ml, ne.away_ml) == (-2.5, 44.0, -150, 126)
    assert ne.odds_source == gp.ODDS_SOURCE_API and ne.odds_captured_at == "2026-12-08T10:00:00Z"
    # LV: the only API event is a month away -> nflverse line; nflverse spread_line is the home margin.
    lv = frame.loc["2026_14_LAC_LV"]
    assert (lv.spread_home, lv.total, lv.home_ml, lv.away_ml) == (2.5, 42.0, 120, -142)
    assert lv.odds_source == gp.ODDS_SOURCE_NFLVERSE
    # GB: no API event and no nflverse line -> row kept with blank odds.
    gb = frame.loc["2026_14_BUF_GB"]
    assert pd.isna(gb.spread_home) and pd.isna(gb.home_ml) and gb.odds_source == "" and gb.odds_captured_at == ""


def test_without_the_odds_api_every_line_comes_from_nflverse():
    frame = _build(api=pd.DataFrame()).set_index("game_id")
    assert frame.loc["2026_14_TB_BAL", "spread_home"] == -6.5
    assert frame.loc["2026_14_MIN_NE", "odds_source"] == gp.ODDS_SOURCE_NFLVERSE


def test_kickoff_pick_time_roof_and_neutral_site():
    frame = _build().set_index("game_id")
    assert frame.loc["2026_14_MIN_NE", "kickoff"] == "2026-12-11T01:15:00Z"  # 20:15 ET Thursday
    # Pick run window opens 3.5 h before the slot's first kickoff; SNF belongs to the Sunday slot.
    assert frame.loc["2026_14_MIN_NE", "pick_expected_at"] == "2026-12-10T21:45:00Z"
    assert frame.loc["2026_14_BUF_GB", "pick_expected_at"] == "2026-12-13T14:30:00Z"
    assert frame.loc["2026_14_PIT_JAX", "pick_expected_at"] == "2026-12-14T21:45:00Z"
    assert frame.loc["2026_14_LAC_LV", "roof"] == "dome" and frame.loc["2026_14_BUF_GB", "roof"] == ""
    assert bool(frame.loc["2026_14_PIT_JAX", "neutral_site"]) and not bool(frame.loc["2026_14_TB_BAL", "neutral_site"])
    assert frame.loc["2026_14_MIN_NE", "venue_name"] == "Gillette Stadium"
    assert frame.loc["2026_14_MIN_NE", "venue_city"] == "Foxborough"
    assert all(isinstance(v, str) for v in frame["venue_name"]) and all(isinstance(v, str) for v in frame["roof"])


def test_median_moneyline_is_taken_in_decimal_odds():
    # A plain median of American prices across a pick'em gives an impossible -2.5.
    assert gp._median_moneyline([-105, 100]) == -102
    # Decimal median of 1.909, 1.952, 2.0, 2.05 is 1.976 -> -102 (never a price between -100 and +100).
    assert gp._median_moneyline([-110, -105, 100, 105]) == -102
    assert gp._median_moneyline([120, 130]) == 125
    assert gp._median_moneyline([]) is np.nan or np.isnan(gp._median_moneyline([]))


def test_summarize_api_odds_reads_all_three_markets():
    event = {
        "home_team": "New England Patriots", "away_team": "Minnesota Vikings", "commence_time": "2026-12-11T01:15:00Z",
        "bookmakers": [
            {"key": "draftkings", "markets": [
                {"key": "h2h", "outcomes": [{"name": "New England Patriots", "price": -150},
                                            {"name": "Minnesota Vikings", "price": 130}]},
                {"key": "spreads", "outcomes": [{"name": "New England Patriots", "price": -110, "point": -3.0},
                                                {"name": "Minnesota Vikings", "price": -110, "point": 3.0}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 44.5},
                                               {"name": "Under", "price": -110, "point": 44.5}]},
            ]},
            {"key": "fanduel", "markets": [
                {"key": "spreads", "outcomes": [{"name": "New England Patriots", "price": -105, "point": -2.5}]},
            ]},
        ],
    }
    row = gp.summarize_api_odds([event]).iloc[0]
    assert (row.home_abbr, row.away_abbr) == ("NE", "MIN")
    assert (row.home_ml, row.away_ml, row.spread_home, row.total) == (-150, 130, -2.75, 44.5)


def test_run_writes_both_files_and_a_recent_build_is_reused(tmp_path, monkeypatch):
    calls = []

    def fetcher():
        calls.append(1)
        return [], ""

    sched = _schedule()
    frame, note = gp.run(2026, 14, out_dir=tmp_path, now=NOW, schedule=sched, odds_fetcher=fetcher)
    current, archive = gp.week_paths(2026, 14, tmp_path)
    assert current.read_text() == archive.read_text()
    text = archive.read_text()
    assert ",true," in text and ",True," not in text  # neutral_site written lower case
    on_disk = pd.read_csv(archive)
    assert list(on_disk.columns) == gp.PREVIEW_COLUMNS and len(calls) == 1
    assert on_disk.loc[on_disk.game_id == "2026_14_MIN_NE", "home_ml"].item() == -155  # an integer, not -155.0
    assert "-155.0" not in text

    # A lock-busy retry 2 h later reuses the build: no second Odds API call, same content.
    later = NOW + timedelta(hours=2)
    again, note = gp.run(2026, 14, out_dir=tmp_path, now=later, schedule=sched, odds_fetcher=fetcher,
                         max_age=timedelta(hours=6))
    assert len(calls) == 1 and note.startswith("reused")
    assert archive.read_text() == text
    # Past the max age it rebuilds.
    gp.run(2026, 14, out_dir=tmp_path, now=NOW + timedelta(hours=7), schedule=sched, odds_fetcher=fetcher,
           max_age=timedelta(hours=6))
    assert len(calls) == 2


def test_a_failed_odds_call_falls_back_to_nflverse(tmp_path):
    frame, note = gp.run(2026, 14, out_dir=tmp_path, now=NOW, schedule=_schedule(),
                         odds_fetcher=lambda: (None, "Odds API request failed"))
    assert "Warning: Odds API request failed" in note
    assert set(frame["odds_source"]) == {gp.ODDS_SOURCE_NFLVERSE, ""}


def test_content_key_ignores_timestamps_but_not_lines():
    frame = _build()
    later = frame.assign(generated_at="2026-12-09T17:00:00Z", odds_captured_at="2026-12-09T17:00:00Z")
    assert gp.content_key(frame) == gp.content_key(later)
    moved = frame.copy()
    moved.loc[0, "spread_home"] = -3.5
    assert gp.content_key(moved) != gp.content_key(frame)


def test_next_week_and_missing_week():
    sched = _schedule()
    assert gp.next_week(sched, 2026, 13) == 14
    assert gp.next_week(sched, 2026, 14) is None
    with pytest.raises(gp.PreviewError):
        gp.build_previews(sched, 2026, 15, api_odds=None, odds_captured_at=None, generated_at=NOW)
