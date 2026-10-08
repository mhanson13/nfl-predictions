from __future__ import annotations

import json

import pandas as pd

from src.data.odds_api_historical import DEFAULT_PLAYER_PROP_MARKETS
from src.data.odds_api_backfill import (
    build_backfill_manifest,
    estimate_quota,
    fetch_event_odds,
    match_events_to_manifest,
    _manifest_target_markets,
    normalize_historical_player_prop_lines,
)


def test_default_oddsapi_player_prop_markets_include_novelty_td_history():
    assert "player_1st_td" in DEFAULT_PLAYER_PROP_MARKETS
    assert "player_last_td" in DEFAULT_PLAYER_PROP_MARKETS
    assert "player_tds_over" in DEFAULT_PLAYER_PROP_MARKETS


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 1,
                "game_type": "REG",
                "game_id": "2023_01_CAR_ATL",
                "gameday": "2023-09-10",
                "gametime": "13:00",
                "away_team": "CAR",
                "home_team": "ATL",
            },
            {
                "season": 2023,
                "week": 1,
                "game_type": "REG",
                "game_id": "2023_01_SEA_LA",
                "gameday": "2023-09-10",
                "gametime": "16:25",
                "away_team": "SEA",
                "home_team": "LA",
            },
            {
                "season": 2023,
                "week": 2,
                "game_type": "REG",
                "game_id": "2023_02_CAR_ATL",
                "gameday": "2023-09-17",
                "gametime": "13:00",
                "away_team": "CAR",
                "home_team": "ATL",
            },
            {
                "season": 2023,
                "week": 1,
                "game_type": "POST",
                "game_id": "2023_01_DAL_PHI",
                "gameday": "2023-09-10",
                "gametime": "13:00",
                "away_team": "DAL",
                "home_team": "PHI",
            },
        ]
    )


def test_build_backfill_manifest_uses_game_snapshots_and_week_event_window():
    manifest = build_backfill_manifest(
        _schedule(),
        start_season=2023,
        end_season=2023,
        current_date="2023-09-11",
        include_postseason=False,
        markets=["player_pass_yds", "player_rush_yds"],
        bookmakers=["draftkings", "fanduel"],
    )

    assert len(manifest) == 2
    first = manifest.iloc[0]
    second = manifest.iloc[1]
    assert first["game_id"] == "2023_01_CAR_ATL"
    assert first["kickoff_time_utc"] == "2023-09-10T17:00:00Z"
    assert first["snapshot_time_utc"] == "2023-09-10T16:50:00Z"
    assert first["events_snapshot_time_utc"] == "2023-09-10T16:50:00Z"
    assert second["commence_time_to_utc"] == "2023-09-11T08:25:00Z"
    assert json.loads(first["target_markets"]) == ["player_pass_yds", "player_rush_yds"]
    assert json.loads(first["target_bookmakers"]) == ["draftkings", "fanduel"]

    quota = estimate_quota(manifest, markets=["player_pass_yds", "player_rush_yds"])

    assert quota == {
        "events_requests": 1,
        "event_markets_requests": 2,
        "event_odds_requests_worst_case": 2,
        "discovery_requests": 3,
        "event_odds_credits_worst_case": 40,
    }


def test_match_events_to_manifest_handles_full_team_names_and_rams_la_abbreviation():
    manifest = build_backfill_manifest(
        _schedule(),
        start_season=2023,
        end_season=2023,
        current_date="2023-09-11",
        include_postseason=False,
    )
    events = pd.DataFrame(
        [
            {
                "requested_season": 2023,
                "requested_week": 1,
                "event_id": "evt-atl",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "commence_time": "2023-09-10T17:00:00Z",
            },
            {
                "requested_season": 2023,
                "requested_week": 1,
                "event_id": "evt-la",
                "home_team": "Los Angeles Rams",
                "away_team": "Seattle Seahawks",
                "commence_time": "2023-09-10T20:25:00Z",
            },
        ]
    )

    matched = match_events_to_manifest(manifest, events)

    assert dict(zip(matched["game_id"], matched["oddsapi_event_id"])) == {
        "2023_01_CAR_ATL": "evt-atl",
        "2023_01_SEA_LA": "evt-la",
    }
    assert matched["events_status"].tolist() == ["matched", "matched"]


def test_normalize_historical_player_prop_lines_pairs_over_under_rows():
    odds = pd.DataFrame(
        [
            {
                "requested_season": 2023,
                "requested_week": 1,
                "requested_game_id": "2023_01_CAR_ATL",
                "event_id": "evt-atl",
                "requested_date": "2023-09-10T16:50:00Z",
                "snapshot_timestamp": "2023-09-10T16:50:00Z",
                "commence_time": "2023-09-10T17:00:00Z",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "market_last_update": "2023-09-10T16:45:00Z",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
            },
            {
                "requested_season": 2023,
                "requested_week": 1,
                "requested_game_id": "2023_01_CAR_ATL",
                "event_id": "evt-atl",
                "requested_date": "2023-09-10T16:50:00Z",
                "snapshot_timestamp": "2023-09-10T16:50:00Z",
                "commence_time": "2023-09-10T17:00:00Z",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "market_last_update": "2023-09-10T16:45:00Z",
                "outcome_name": "Under",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
            },
        ]
    )

    lines = normalize_historical_player_prop_lines(odds, vendors=["draftkings"])

    assert len(lines) == 1
    row = lines.iloc[0]
    assert row["line_source"] == "odds_api_historical"
    assert row["season"] == 2023
    assert row["week"] == 1
    assert row["game_id"] == "2023_01_CAR_ATL"
    assert row["oddsapi_event_id"] == "evt-atl"
    assert row["market"] == "qb_passing_yards"
    assert row["player_name_key"] == "q:one"
    assert row["player_name_prefix_key"] == "qb:one"
    assert row["line"] == 245.5
    assert row["sportsbook"] == "draftkings"
    assert row["over_odds"] == -110
    assert row["under_odds"] == -110
    assert row["implied_probability"] == 0.5
    assert bool(row["line_is_comparable"]) is True


def test_normalize_historical_player_prop_lines_keeps_anytime_td_yes_rows():
    odds = pd.DataFrame(
        [
            {
                "requested_season": 2026,
                "requested_week": 3,
                "requested_game_id": "2026_03_ATL_GB",
                "event_id": "evt-gb",
                "requested_date": "2026-09-25T00:05:00Z",
                "snapshot_timestamp": "2026-09-25T00:00:36Z",
                "commence_time": "2026-09-25T00:15:00Z",
                "home_team": "Green Bay Packers",
                "away_team": "Atlanta Falcons",
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
                "market_last_update": "2026-09-25T00:00:18Z",
                "outcome_name": "Yes",
                "player_name": "Bijan Robinson",
                "price": -140,
                "point": pd.NA,
            },
        ]
    )

    lines = normalize_historical_player_prop_lines(odds, vendors=["draftkings"])

    assert len(lines) == 1
    row = lines.iloc[0]
    assert row["market"] == "anytime_td"
    assert row["line"] == 0.5
    assert row["over_odds"] == -140
    assert pd.isna(row["under_odds"])
    assert round(row["implied_probability"], 6) == round(140 / 240, 6)
    assert bool(row["line_is_comparable"]) is False


def test_normalize_historical_player_prop_lines_maps_novelty_td_markets():
    base = {
        "requested_season": 2026,
        "requested_week": 3,
        "requested_game_id": "2026_03_ATL_GB",
        "event_id": "evt-gb",
        "requested_date": "2026-09-25T00:05:00Z",
        "snapshot_timestamp": "2026-09-25T00:00:36Z",
        "commence_time": "2026-09-25T00:15:00Z",
        "home_team": "Green Bay Packers",
        "away_team": "Atlanta Falcons",
        "bookmaker_key": "draftkings",
        "market_last_update": "2026-09-25T00:00:18Z",
        "player_name": "Bijan Robinson",
    }
    odds = pd.DataFrame(
        [
            {**base, "market_key": "player_1st_td", "outcome_name": "Yes", "price": 900, "point": pd.NA},
            {**base, "market_key": "player_last_td", "outcome_name": "Yes", "price": 1000, "point": pd.NA},
            {**base, "market_key": "player_tds_over", "outcome_name": "Over", "price": 450, "point": 1.5},
            {**base, "market_key": "player_tds_over", "outcome_name": "Over", "price": 1400, "point": 2.5},
        ]
    )

    lines = normalize_historical_player_prop_lines(odds, vendors=["draftkings"])

    by_market = lines.set_index("market")
    assert set(by_market.index) == {"first_td", "last_td", "two_plus_tds", "three_plus_tds"}
    assert by_market["line"].eq(0.5).all()
    assert by_market.loc["first_td", "over_odds"] == 900
    assert by_market.loc["last_td", "over_odds"] == 1000
    assert by_market.loc["two_plus_tds", "over_odds"] == 450
    assert by_market.loc["three_plus_tds", "over_odds"] == 1400
    assert by_market["under_odds"].isna().all()
    assert by_market["line_is_comparable"].eq(False).all()


def test_normalize_historical_player_prop_lines_maps_two_way_touchdown_milestones():
    base = {
        "requested_season": 2026,
        "requested_week": 3,
        "requested_game_id": "2026_03_ATL_GB",
        "event_id": "evt-gb",
        "requested_date": "2026-09-25T00:05:00Z",
        "snapshot_timestamp": "2026-09-25T00:00:36Z",
        "commence_time": "2026-09-25T00:15:00Z",
        "home_team": "Green Bay Packers",
        "away_team": "Atlanta Falcons",
        "bookmaker_key": "draftkings",
        "market_key": "player_tds",
        "market_last_update": "2026-09-25T00:00:18Z",
        "player_name": "Bijan Robinson",
        "point": 1.5,
    }
    odds = pd.DataFrame(
        [
            {**base, "outcome_name": "Over", "price": 250},
            {**base, "outcome_name": "Under", "price": -320},
        ]
    )

    lines = normalize_historical_player_prop_lines(odds, vendors=["draftkings"])

    assert len(lines) == 1
    row = lines.iloc[0]
    assert row["market"] == "two_plus_tds"
    assert row["line"] == 0.5
    assert row["over_odds"] == 250
    assert row["under_odds"] == -320
    assert bool(row["line_is_comparable"]) is False


def test_manifest_target_markets_drive_quota_display_markets():
    manifest = build_backfill_manifest(
        _schedule(),
        start_season=2023,
        end_season=2023,
        current_date="2023-09-11",
        include_postseason=False,
        markets=["player_anytime_td"],
    )

    assert _manifest_target_markets(manifest, ["player_pass_yds", "player_rush_yds"]) == ["player_anytime_td"]


def test_fetch_event_odds_dry_run_requires_market_discovery(tmp_path):
    manifest = build_backfill_manifest(
        _schedule(),
        start_season=2023,
        end_season=2023,
        current_date="2023-09-11",
        include_postseason=False,
    )
    manifest["oddsapi_event_id"] = ["evt-atl", "evt-la"]

    updated, existing, planned, estimated_credits = fetch_event_odds(
        manifest,
        pd.DataFrame(),
        api_key=None,
        output=tmp_path / "odds.parquet",
        sport_key="americanfootball_nfl",
        regions=["us"],
        odds_format="american",
        include_links=False,
        include_sids=False,
        retries=0,
        sleep=0,
        timeout=1,
        dry_run=True,
        resume=True,
        max_requests=None,
    )

    assert existing.empty
    assert planned == 0
    assert estimated_credits == 0
    assert updated["event_odds_status"].tolist() == ["pending", "pending"]


def test_fetch_event_odds_dry_run_handles_manifest_before_event_discovery(tmp_path):
    manifest = build_backfill_manifest(
        _schedule(),
        start_season=2023,
        end_season=2023,
        current_date="2023-09-11",
        include_postseason=False,
    )

    updated, existing, planned, estimated_credits = fetch_event_odds(
        manifest,
        pd.DataFrame(),
        api_key=None,
        output=tmp_path / "odds.parquet",
        sport_key="americanfootball_nfl",
        regions=["us"],
        odds_format="american",
        include_links=False,
        include_sids=False,
        retries=0,
        sleep=0,
        timeout=1,
        dry_run=True,
        resume=True,
        max_requests=None,
    )

    assert existing.empty
    assert planned == 0
    assert estimated_credits == 0
    assert updated["event_odds_status"].tolist() == ["missing_event", "missing_event"]


def test_fetch_event_odds_resume_requests_missing_new_market(tmp_path):
    manifest = build_backfill_manifest(
        _schedule(),
        start_season=2023,
        end_season=2023,
        current_date="2023-09-11",
        include_postseason=False,
        markets=["player_anytime_td"],
    )
    manifest["oddsapi_event_id"] = ["evt-atl", "evt-la"]

    markets = pd.DataFrame(
        [
            {
                "event_id": "evt-atl",
                "requested_date": manifest.loc[0, "snapshot_time_utc"],
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
            },
            {
                "event_id": "evt-la",
                "requested_date": manifest.loc[1, "snapshot_time_utc"],
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
            },
        ]
    )
    existing_odds = pd.DataFrame(
        [
            {
                "event_id": "evt-atl",
                "requested_date": manifest.loc[0, "snapshot_time_utc"],
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
            },
            {
                "event_id": "evt-la",
                "requested_date": manifest.loc[1, "snapshot_time_utc"],
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
            },
        ]
    )
    odds_output = tmp_path / "odds.parquet"
    existing_odds.to_parquet(odds_output)

    updated, _existing, planned, estimated_credits = fetch_event_odds(
        manifest,
        markets,
        api_key=None,
        output=odds_output,
        sport_key="americanfootball_nfl",
        regions=["us"],
        odds_format="american",
        include_links=False,
        include_sids=False,
        retries=0,
        sleep=0,
        timeout=1,
        dry_run=True,
        resume=True,
        max_requests=None,
    )

    assert planned == 1
    assert estimated_credits == 10
    assert updated["event_odds_status"].tolist() == ["pending", "done"]
