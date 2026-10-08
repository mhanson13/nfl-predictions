from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.player_props.features import build_player_prop_features, build_player_prop_novelty_features
from src.player_props.predict import (
    _attach_edge_recommendations,
    _attach_prop_odds,
    _read_frame_if_exists,
    build_novelty_prediction_candidates,
    build_prediction_candidates,
    main,
    predict_player_props,
    prepare_player_prop_odds,
)
from src.player_props.train import train_and_evaluate


def _labels() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("qb-1", "QB One", "QB", "QB", "qb_passing_yards", [210.0, 240.0, 260.0, 280.0]),
        ("qb-1", "QB One", "QB", "QB", "qb_passing_tds", [1.0, 2.0, 1.0, 3.0]),
        ("qb-1", "QB One", "QB", "QB", "anytime_td", [0.0, 1.0, 0.0, 1.0]),
        ("rb-1", "RB One", "RB", "RB", "rb_rushing_yards", [40.0, 60.0, 70.0, 80.0]),
        ("rb-1", "RB One", "RB", "RB", "rb_rushing_tds", [0.0, 1.0, 0.0, 1.0]),
        ("rb-1", "RB One", "RB", "RB", "anytime_td", [0.0, 1.0, 0.0, 1.0]),
        ("wr-1", "WR One", "WR", "WR", "wrte_receiving_yards", [35.0, 45.0, 55.0, 65.0]),
        ("wr-1", "WR One", "WR", "WR", "wrte_receiving_tds", [1.0, 0.0, 1.0, 0.0]),
        ("wr-1", "WR One", "WR", "WR", "anytime_td", [1.0, 0.0, 1.0, 0.0]),
        ("dl-1", "DL One", "DE", "DL", "def_sacks", [0.0, 1.0, 0.0, 1.0]),
    ]
    weeks = [(2025, 1), (2025, 2), (2026, 1), (2026, 2)]
    for player_id, name, position, position_group, market, values in specs:
        for (season, week), value in zip(weeks, values):
            team = "ATL" if position_group != "DL" else "CAR"
            opponent = "CAR" if team == "ATL" else "ATL"
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week:02d}_CAR_ATL",
                    "team": team,
                    "opponent": opponent,
                    "player_id": player_id,
                    "player_name": name,
                    "position": position,
                    "position_group": position_group,
                    "market": market,
                    "actual_value": value,
                    "actual_over_zero": value > 0,
                    "played_flag": True,
                    "active_flag": True,
                    "value_type": "count" if market.endswith("_tds") or market in {"anytime_td", "def_sacks"} else "yards",
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def _novelty_labels() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weeks = [(2025, 1), (2025, 2), (2026, 1), (2026, 2)]
    values = {
        "first_td": [1.0, 0.0, 1.0, 0.0],
        "last_td": [0.0, 1.0, 0.0, 1.0],
        "two_plus_tds": [0.0, 1.0, 0.0, 1.0],
        "three_plus_tds": [0.0, 0.0, 0.0, 1.0],
    }
    for market, market_values in values.items():
        for (season, week), value in zip(weeks, market_values):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week:02d}_CAR_ATL",
                    "team": "ATL",
                    "opponent": "CAR",
                    "player_id": "rb-1",
                    "player_name": "RB One",
                    "position": "RB",
                    "position_group": "RB",
                    "market": market,
                    "actual_value": value,
                    "actual_over_zero": value > 0,
                    "played_flag": True,
                    "active_flag": True,
                    "value_type": "count",
                    "source": "nfl_pbp",
                }
            )
    return pd.DataFrame(rows)


def _projection_sources(prediction_dir: Path) -> tuple[Path, Path, Path]:
    qb_path = prediction_dir / "predictions_players_qb.csv"
    offense_path = prediction_dir / "predictions_players_offense.csv"
    defense_path = prediction_dir / "predictions_players_defense.csv"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "team_side": "home",
                "player_id": "qb-1",
                "player_name": "QB One",
                "projected_passing_yards": 250.0,
                "projected_passing_tds": 1.8,
                "projected_rushing_tds": 0.4,
                "games_sampled": 4,
                "player_rank": 1,
            }
        ]
    ).to_csv(qb_path, index=False)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "team_side": "home",
                "player_id": "rb-1",
                "player_name": "RB One",
                "projected_rushing_yards": 75.0,
                "projected_rushing_tds": 0.6,
                "projected_receiving_yards": 8.0,
                "projected_receiving_tds": 0.1,
                "projected_total_tds": 0.7,
                "games_sampled": 4,
                "player_rank": 1,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "team_side": "home",
                "player_id": "wr-1",
                "player_name": "WR One",
                "projected_rushing_yards": 1.0,
                "projected_rushing_tds": 0.0,
                "projected_receiving_yards": 60.0,
                "projected_receiving_tds": 0.7,
                "projected_total_tds": 0.7,
                "games_sampled": 4,
                "player_rank": 2,
            },
        ]
    ).to_csv(offense_path, index=False)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "team": "CAR",
                "team_side": "away",
                "player_id": "dl-1",
                "player_name": "DL One",
                "projected_sacks": 0.6,
                "games_sampled": 4,
                "player_rank": 1,
            }
        ]
    ).to_csv(defense_path, index=False)
    return qb_path, offense_path, defense_path


def _write_rosters(path: Path) -> None:
    pd.DataFrame(
        [
            {"season": 2026, "week": 3, "gsis_id": "rb-1", "football_name": "RB One", "position": "RB"},
            {"season": 2026, "week": 3, "gsis_id": "wr-1", "football_name": "WR One", "position": "WR"},
            {"season": 2026, "week": 3, "gsis_id": "dl-1", "football_name": "DL One", "position": "DE"},
        ]
    ).to_parquet(path, index=False)


def _write_models(labels: pd.DataFrame, models_dir: Path, novelty_labels: pd.DataFrame | None = None) -> None:
    offense_features, defense_features = build_player_prop_features(labels)
    novelty_features = (
        build_player_prop_novelty_features(novelty_labels)
        if novelty_labels is not None and not novelty_labels.empty
        else pd.DataFrame()
    )
    _, _, bundles = train_and_evaluate(
        offense_features=offense_features,
        defense_features=defense_features,
        novelty_features=novelty_features,
        model_type="ridge",
        min_train_rows=2,
        generated_at="2026-09-22T00:00:00+00:00",
    )
    for market, bundle in bundles.items():
        family = str(bundle["market_family"])
        target_dir = models_dir / family
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, target_dir / f"{market}_model.pkl")


def test_build_prediction_candidates_routes_markets_by_position(tmp_path):
    labels = _labels()
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)

    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )

    assert set(candidates["market"]) == {
        "qb_passing_yards",
        "qb_passing_tds",
        "rb_rushing_yards",
        "rb_rushing_tds",
        "wrte_receiving_yards",
        "wrte_receiving_tds",
        "anytime_td",
        "def_sacks",
    }
    rb = candidates[candidates["market"].eq("rb_rushing_yards")].iloc[0]
    wr = candidates[candidates["market"].eq("wrte_receiving_yards")].iloc[0]
    assert rb["player_id"] == "rb-1"
    assert wr["player_id"] == "wr-1"


def test_build_novelty_prediction_candidates_expands_touchdown_markets(tmp_path):
    labels = _labels()
    qb_path, offense_path, _defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)

    candidates = build_novelty_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        rosters_path=rosters_path,
    )

    assert set(candidates["market"]) == {"first_td", "last_td", "two_plus_tds", "three_plus_tds"}
    assert set(candidates["market_family"]) == {"novelty"}
    assert set(candidates["player_id"]) == {"qb-1", "rb-1", "wr-1"}
    rb_two_plus = candidates[(candidates["player_id"] == "rb-1") & (candidates["market"] == "two_plus_tds")].iloc[0]
    assert rb_two_plus["baseline_projection"] == 0.7


def test_predict_player_props_writes_split_model_outputs(tmp_path):
    labels = _labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )

    qb, offense, defense, novelty = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    assert set(qb["market"]) == {"qb_passing_yards", "qb_passing_tds"}
    assert set(offense["market"]) == {
        "rb_rushing_yards",
        "rb_rushing_tds",
        "wrte_receiving_yards",
        "wrte_receiving_tds",
        "anytime_td",
    }
    assert defense["market"].tolist() == ["def_sacks"]
    assert qb["projection"].notna().all()
    assert offense["projection"].notna().all()
    assert qb[qb["market"].eq("qb_passing_tds")]["prob_over"].notna().all()
    assert offense[offense["market"].str.endswith("_tds")]["prob_over"].notna().all()
    assert offense[offense["market"].eq("anytime_td")]["prob_over"].notna().all()
    assert defense["prob_over"].notna().all()
    assert novelty.empty
    assert int(qb.iloc[0]["sample_size"]) == 4


def test_predict_player_props_adds_line_offered_def_sack_candidates(tmp_path):
    labels = pd.concat(
        [
            _labels(),
            pd.DataFrame(
                [
                    {
                        "season": season,
                        "week": week,
                        "game_id": f"{season}_{week:02d}_CAR_ATL",
                        "team": "CAR",
                        "opponent": "ATL",
                        "player_id": "edge-2",
                        "player_name": "EDGE Two",
                        "position": "DE",
                        "position_group": "DL",
                        "market": "def_sacks",
                        "actual_value": value,
                        "actual_over_zero": value > 0,
                        "played_flag": True,
                        "active_flag": True,
                        "value_type": "count",
                        "source": "test",
                    }
                    for (season, week), value in zip(
                        [(2025, 1), (2025, 2), (2026, 1), (2026, 2)],
                        [1.0, 0.0, 1.0, 0.0],
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "fanduel",
                "market_key": "player_sacks",
                "outcome_name": "1+ Sacks",
                "player_name": "EDGE Two",
                "price": 170,
                "point": None,
                "market_last_update": "2026-09-22T12:00:00Z",
            }
        ]
    )
    odds = prepare_player_prop_odds(raw_odds, season=2026, week=3, vendor="fanduel")
    player_identity = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "team": "CAR",
                "player_id": "edge-2",
                "alias_name": "EDGE Two",
                "alias_confidence": 1.0,
            }
        ]
    )
    availability = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "player_id": "edge-2",
                "player_name": "EDGE Two",
                "team": "CAR",
                "current_team": "CAR",
                "position": "DE",
                "position_group": "DL",
                "active_roster_flag": True,
                "practice_squad_flag": False,
                "injured_reserve_flag": False,
                "availability_source": "espn",
                "availability_priority": 10,
            }
        ]
    )

    _, _, defense, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        availability=availability,
        player_identity=player_identity,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    assert set(defense["player_id"]) == {"dl-1", "edge-2"}
    edge = defense[defense["player_id"].eq("edge-2")].iloc[0]
    assert edge["line"] == 1.0
    assert edge["sportsbook"] == "fanduel"
    assert edge["baseline_projection"] == 0.5
    assert int(edge["games_sampled"]) == 4
    assert int(edge["sample_size"]) == 4
    assert edge["roster_validation_flag"] == "ok"
    assert bool(edge["bettable_flag"]) is True


def test_predict_player_props_marks_anytime_td_fallback_lines_unbettable(tmp_path):
    labels = _labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
                "outcome_name": "Anytime Touchdown Scorer",
                "player_name": "RB One",
                "price": 150,
                "point": None,
            }
        ]
    )
    odds = prepare_player_prop_odds(raw_odds, season=2026, week=3, vendor="draftkings")

    _, offense, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    rb_td = offense[(offense["player_id"].eq("rb-1")) & (offense["market"].eq("rb_rushing_tds"))].iloc[0]
    assert rb_td["line"] == 0.5
    assert rb_td["line_type"] == "anytime_td_fallback"
    assert rb_td["prop_quality_flag"] == "fallback_line"
    assert bool(rb_td["bettable_flag"]) is False

    anytime_td = offense[(offense["player_id"].eq("rb-1")) & (offense["market"].eq("anytime_td"))].iloc[0]
    assert anytime_td["line"] == 0.5
    assert anytime_td["line_type"] == "milestone"
    assert bool(anytime_td["bettable_flag"]) is True


def test_predict_player_props_writes_novelty_research_only_output(tmp_path):
    labels = _labels()
    novelty_labels = _novelty_labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir, novelty_labels=novelty_labels)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    novelty_candidates = build_novelty_prediction_candidates(
        labels=novelty_labels,
        qb_path=qb_path,
        offense_path=offense_path,
        rosters_path=rosters_path,
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "first_td",
                "player_name_key": "r:one",
                "line": 0.5,
                "line_type": "milestone",
                "implied_probability": 0.2,
                "sportsbook": "draftkings",
                "over_odds": 400,
                "under_odds": pd.NA,
                "line_updated_at": "2026-09-22T12:00:00Z",
                "line_is_comparable": True,
            }
        ]
    )

    _, _, _, novelty = predict_player_props(
        labels=labels,
        candidates=candidates,
        novelty_labels=novelty_labels,
        novelty_candidates=novelty_candidates,
        models_dir=models_dir,
        odds=odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    assert set(novelty["market"]) == {"first_td", "last_td", "two_plus_tds", "three_plus_tds"}
    first_td = novelty[(novelty["market"] == "first_td") & (novelty["player_id"] == "rb-1")].iloc[0]
    assert first_td["line"] == 0.5
    assert first_td["line_type"] == "milestone"
    assert first_td["edge_recommendation_tier"] == "research_only"
    assert bool(first_td["bettable_flag"]) is False


def test_prepare_player_prop_odds_maps_propline_draftkings_lines_to_model_markets():
    odds = pd.DataFrame(
        [
            {
                "event_id": "101",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
                "market_last_update": "2026-09-22T12:00:00Z",
            },
            {
                "event_id": "101",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Under",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
                "market_last_update": "2026-09-22T12:00:00Z",
            }
        ]
    )

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    row = prepared.iloc[0]
    assert row["game_id"] == "2026_03_CAR_ATL"
    assert row["market"] == "qb_passing_yards"
    assert row["player_name_key"] == "q:one"
    assert row["line"] == 245.5
    assert row["sportsbook"] == "draftkings"
    assert row["implied_probability"] == 0.5


def test_prepare_player_prop_odds_prefers_source_game_id_when_available():
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_04_CAR_ATL",
                "event_id": "101",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
                "market_last_update": "2026-09-22T12:00:00Z",
            },
            {
                "game_id": "2026_04_CAR_ATL",
                "event_id": "101",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Under",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
                "market_last_update": "2026-09-22T12:00:00Z",
            },
        ]
    )

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    assert prepared.iloc[0]["game_id"] == "2026_04_CAR_ATL"


def test_prepare_player_prop_odds_maps_rams_to_local_la_and_strips_team_suffix():
    odds = pd.DataFrame(
        [
            {
                "event_id": "101",
                "home_team": "Denver Broncos",
                "away_team": "Los Angeles Rams",
                "bookmaker_key": "draftkings",
                "market_key": "player_sacks",
                "outcome_name": "1+ Sacks",
                "player_name": "Byron Young (LAR)",
                "price": 199,
                "point": None,
                "market_last_update": "2026-09-22T12:00:00Z",
            }
        ]
    )

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    row = prepared.iloc[0]
    assert row["game_id"] == "2026_03_LA_DEN"
    assert row["market"] == "def_sacks"
    assert row["player_name_key"] == "b:young"
    assert row["player_name_prefix_key"] == "by:young"
    assert row["line"] == 1.0
    assert row["line_type"] == "milestone"


def test_prepare_player_prop_odds_maps_propline_touchdown_lines_to_model_markets():
    rows = []
    for market_key, player_name, line in [
        ("player_pass_tds", "QB One", 1.5),
        ("player_rush_tds", "RB One", 0.5),
        ("player_reception_tds", "WR One", 0.5),
    ]:
        for side, price in [("Over", -110), ("Under", -110)]:
            rows.append(
                {
                    "event_id": "101",
                    "home_team": "Atlanta Falcons",
                    "away_team": "Carolina Panthers",
                    "bookmaker_key": "draftkings",
                    "market_key": market_key,
                    "outcome_name": side,
                    "player_name": player_name,
                    "price": price,
                    "point": line,
                    "market_last_update": "2026-09-22T12:00:00Z",
                }
            )
    odds = pd.DataFrame(rows)

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    by_player = prepared.set_index("player_name_key")
    assert by_player.loc["q:one", "market"] == "qb_passing_tds"
    assert by_player.loc["r:one", "market"] == "rb_rushing_tds"
    assert by_player.loc["w:one", "market"] == "wrte_receiving_tds"
    assert by_player.loc["q:one", "line"] == 1.5
    assert by_player.loc["r:one", "line"] == 0.5
    assert by_player.loc["w:one", "line"] == 0.5
    assert by_player["implied_probability"].eq(0.5).all()


def test_prepare_player_prop_odds_maps_anytime_td_scorer_market():
    odds = pd.DataFrame(
        [
            {
                "event_id": "101",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
                "outcome_name": "RB One",
                "player_name": "RB One",
                "price": 150,
                "point": None,
                "market_last_update": "2026-09-22T12:00:00Z",
            }
        ]
    )

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    row = prepared.iloc[0]
    assert row["market"] == "anytime_td"
    assert row["line"] == 0.5
    assert row["line_type"] == "milestone"
    assert row["over_odds"] == 150
    assert pd.isna(row["under_odds"])
    assert round(row["implied_probability"], 6) == round(100 / 250, 6)


def test_prepare_player_prop_odds_maps_novelty_td_scorer_markets():
    odds = pd.DataFrame(
        [
            {
                "event_id": "101",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": market_key,
                "outcome_name": "RB One",
                "player_name": "RB One",
                "price": 150,
                "point": None,
                "market_last_update": "2026-09-22T12:00:00Z",
            }
            for market_key in ["player_1st_td", "player_last_td", "player_2plus_td", "player_3plus_td"]
        ]
    )

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    by_market = prepared.set_index("market")
    assert by_market.loc["first_td", "line"] == 0.5
    assert by_market.loc["last_td", "line"] == 0.5
    assert by_market.loc["two_plus_tds", "line"] == 0.5
    assert by_market.loc["three_plus_tds", "line"] == 0.5
    assert by_market["line_type"].eq("milestone").all()
    assert by_market["over_odds"].eq(150).all()
    assert by_market["under_odds"].isna().all()


def test_read_frame_if_exists_uses_csv_fallback_for_missing_parquet(tmp_path):
    fallback = tmp_path / "availability.csv"
    pd.DataFrame([{"player_id": "qb-1", "current_team": "ATL"}]).to_csv(fallback, index=False)

    frame = _read_frame_if_exists(tmp_path / "availability.parquet")

    assert frame["player_id"].tolist() == ["qb-1"]
    assert frame["current_team"].tolist() == ["ATL"]


def test_prepare_player_prop_odds_normalizes_sportsbook_team_aliases():
    odds = pd.DataFrame(
        [
            {
                "event_id": "101",
                "home_team": "GB Packers",
                "away_team": "ATL Falcons",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
            }
        ]
    )

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    assert prepared.iloc[0]["game_id"] == "2026_03_ATL_GB"


def test_prepare_player_prop_odds_handles_propline_milestone_lines():
    odds = pd.DataFrame(
        [
            {
                "event_id": "101",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_rush_yds",
                "outcome_name": "70+ Rushing Yards",
                "player_name": "RB One",
                "price": -120,
                "point": None,
                "market_last_update": "2026-09-22T12:00:00Z",
            }
        ]
    )

    prepared = prepare_player_prop_odds(odds, season=2026, week=3, vendor="draftkings")

    row = prepared.iloc[0]
    assert row["market"] == "rb_rushing_yards"
    assert row["line"] == 70.0
    assert row["line_type"] == "milestone"
    assert row["over_odds"] == -120
    assert pd.isna(row["under_odds"])
    assert round(row["implied_probability"], 6) == round(120 / 220, 6)


def test_attach_prop_odds_prefers_canonical_line_over_milestone_ladder():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name": "QB One",
                "team": "ATL",
                "projection": 242.0,
            }
        ]
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
            },
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Under",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
            },
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "200+ Passing Yards",
                "player_name": "QB One",
                "price": -250,
                "point": None,
            },
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "260+ Passing Yards",
                "player_name": "QB One",
                "price": 150,
                "point": None,
            },
        ]
    )
    odds = prepare_player_prop_odds(raw_odds, season=2026, week=3, vendor="draftkings")

    attached = _attach_prop_odds(predictions, odds)

    row = attached.iloc[0]
    assert row["line"] == 245.5
    assert row["line_type"] == "canonical"
    assert row["under_odds"] == -110
    assert row["line_options_count"] == 1
    assert row["line_range"] == 0.0


def test_prepare_player_prop_odds_marks_extra_two_way_points_as_alternate():
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "hardrock",
                "market_key": "player_pass_yds",
                "outcome_name": side,
                "player_name": "QB One",
                "price": price,
                "point": point,
            }
            for point, over_price, under_price in [
                (199.5, -250, 175),
                (224.5, -130, -110),
                (249.5, 145, -200),
            ]
            for side, price in [("Over", over_price), ("Under", under_price)]
        ]
    )

    prepared = prepare_player_prop_odds(raw_odds, season=2026, week=3, vendor="hardrock")

    line_types = dict(zip(prepared["line"], prepared["line_type"]))
    assert line_types[199.5] == "alternate"
    assert line_types[224.5] == "alternate"
    assert line_types[249.5] == "alternate"


def test_attach_prop_odds_reduces_milestone_ladder_to_nearest_line_per_book():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name": "QB One",
                "team": "ATL",
                "projection": 218.0,
            }
        ]
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": outcome_name,
                "player_name": "QB One",
                "price": price,
                "point": None,
            }
            for outcome_name, price in [
                ("180+ Passing Yards", -350),
                ("210+ Passing Yards", -145),
                ("220+ Passing Yards", -115),
                ("260+ Passing Yards", 190),
            ]
        ]
    )
    odds = prepare_player_prop_odds(raw_odds, season=2026, week=3, vendor="draftkings")

    attached = _attach_prop_odds(predictions, odds)

    row = attached.iloc[0]
    assert row["line"] == 220.0
    assert row["line_type"] == "milestone"
    assert row["line_options_count"] == 1
    assert row["line_range"] == 0.0
    assert bool(row["line_variance_flag"]) is False


def test_attach_prop_odds_selects_best_price_for_equivalent_one_sided_lines():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "anytime_td",
                "player_name": "RB One",
                "team": "ATL",
                "projection": 0.18,
            }
        ]
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": sportsbook,
                "market_key": "player_anytime_td",
                "outcome_name": "Anytime Touchdown Scorer",
                "player_name": "RB One",
                "price": price,
                "point": None,
            }
            for sportsbook, price in [("draftkings", 250), ("hardrock", 320), ("fanduel", 275)]
        ]
    )
    odds = prepare_player_prop_odds(
        raw_odds,
        season=2026,
        week=3,
        vendor=["draftkings", "hardrock", "fanduel"],
    )

    attached = _attach_prop_odds(predictions, odds)

    row = attached.iloc[0]
    assert row["line"] == 0.5
    assert row["line_type"] == "milestone"
    assert row["sportsbook"] == "hardrock"
    assert row["over_odds"] == 320
    assert row["line_book_count"] == 3
    assert row["line_options_count"] == 3


def test_attach_prop_odds_uses_labeled_anytime_td_fallback_for_stat_td_markets():
    predictions = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "market": "rb_rushing_tds",
                "player_id": "rb-1",
                "player_name": "R.One",
                "team": "ATL",
                "projection": 0.32,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "market": "wrte_receiving_tds",
                "player_id": "wr-1",
                "player_name": "W.One",
                "team": "ATL",
                "projection": 0.24,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_tds",
                "player_id": "qb-1",
                "player_name": "QB One",
                "team": "ATL",
                "projection": 1.6,
            },
        ]
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
                "outcome_name": "Anytime Touchdown Scorer",
                "player_name": player_name,
                "price": price,
                "point": None,
            }
            for player_name, price in [("RB One", 150), ("WR One", 225), ("QB One", 400)]
        ]
    )
    odds = prepare_player_prop_odds(raw_odds, season=2026, week=3, vendor="draftkings")
    identity = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "team": "ATL",
                "player_id": "rb-1",
                "alias_name": "RB One",
                "alias_confidence": 1.0,
            },
            {
                "season": 2026,
                "week": 3,
                "team": "ATL",
                "player_id": "wr-1",
                "alias_name": "WR One",
                "alias_confidence": 1.0,
            },
        ]
    )

    attached = _attach_prop_odds(predictions, odds, identity=identity)

    by_market = attached.set_index("market")
    assert by_market.loc["rb_rushing_tds", "line"] == 0.5
    assert by_market.loc["rb_rushing_tds", "line_type"] == "anytime_td_fallback"
    assert by_market.loc["rb_rushing_tds", "over_odds"] == 150
    assert by_market.loc["wrte_receiving_tds", "line"] == 0.5
    assert by_market.loc["wrte_receiving_tds", "line_type"] == "anytime_td_fallback"
    assert by_market.loc["wrte_receiving_tds", "over_odds"] == 225
    assert pd.isna(by_market.loc["qb_passing_tds", "line"])


def test_attach_edge_recommendations_uses_market_policy_threshold():
    predictions = pd.DataFrame(
        [
            {
                "market": "anytime_td",
                "sportsbook": "draftkings",
                "edge_side": "over",
                "edge_probability_edge": 0.12,
                "edge_ev": 0.4,
                "bettable_flag": True,
            },
            {
                "market": "anytime_td",
                "sportsbook": "draftkings",
                "edge_side": "over",
                "edge_probability_edge": 0.01,
                "edge_ev": 0.1,
                "bettable_flag": True,
            },
            {
                "market": "def_sacks",
                "sportsbook": "draftkings",
                "edge_side": "over",
                "edge_probability_edge": 0.20,
                "edge_ev": 0.5,
                "bettable_flag": True,
            },
        ]
    )
    policy = pd.DataFrame(
        [
            {
                "scope": "market",
                "market": "anytime_td",
                "sportsbook": pd.NA,
                "strategy": "prob_edge",
                "threshold": 0.05,
                "roi": 0.10,
                "n_bets": 120,
                "recommendation_tier": "strong",
            }
        ]
    )

    attached = _attach_edge_recommendations(predictions, policy)

    assert attached["edge_recommendation_flag"].tolist() == [True, False, False]
    assert attached["edge_recommendation_tier"].tolist() == ["strong", "below_threshold", "research_only"]
    assert attached.loc[0, "edge_policy_threshold"] == 0.05
    assert attached.loc[0, "edge_policy_n_bets"] == 120


def test_predict_player_props_attaches_draftkings_line_and_edge(tmp_path):
    labels = _labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name_key": "q:one",
                "bdl_team": "ATL",
                "line": 245.5,
                "implied_probability": 0.5,
                "sportsbook": "draftkings",
                "over_odds": -110,
                "under_odds": -110,
                "line_updated_at": "2026-09-22T12:00:00Z",
            }
        ]
    )

    qb, _, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb[qb["market"].eq("qb_passing_yards")].iloc[0]
    assert row["line"] == 245.5
    assert row["sportsbook"] == "draftkings"
    assert row["edge"] == row["projection"] - 245.5


def test_predict_player_props_adds_line_over_probability_from_history(tmp_path):
    labels = _labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name_key": "q:one",
                "line": 245.5,
                "implied_probability": 0.5,
                "sportsbook": "draftkings",
                "over_odds": -110,
                "under_odds": -110,
                "line_updated_at": "2026-09-22T12:00:00Z",
            }
        ]
    )
    over_probability_history = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_CAR_ATL",
                "market": "qb_passing_yards",
                "player_id": "qb-hist",
                "model_projection": 240.0,
                "line": 235.5,
                "actual_value": 250.0,
                "implied_probability": 0.5,
            }
        ]
    )

    qb, _, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        over_probability_history=over_probability_history,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb[qb["market"].eq("qb_passing_yards")].iloc[0]
    assert row["prob_over"] == 0.5
    assert row["prob_edge"] == 0.0
    assert row["prob_over_method"] == "market_implied_fallback"
    assert row["prob_over_sample_size"] == 1
    assert round(row["over_break_even_probability"], 6) == round(110 / 210, 6)
    assert round(row["under_break_even_probability"], 6) == round(110 / 210, 6)
    assert round(row["over_ev"], 6) == round((0.5 * (100 / 110)) - 0.5, 6)
    assert round(row["under_ev"], 6) == round((0.5 * (100 / 110)) - 0.5, 6)
    assert pd.isna(row["edge_side"])
    assert pd.isna(row["edge_odds"])
    assert pd.isna(row["edge_break_even_probability"])
    assert row["edge_ev"] == 0.0


def test_predict_player_props_reports_line_variance_across_books(tmp_path):
    labels = _labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name_key": "q:one",
                "line": 245.5,
                "implied_probability": 0.5,
                "sportsbook": "draftkings",
                "over_odds": -110,
                "under_odds": -110,
                "line_updated_at": "2026-09-22T12:00:00Z",
                "line_is_comparable": True,
            },
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name_key": "q:one",
                "line": 249.5,
                "implied_probability": 0.5,
                "sportsbook": "fanduel",
                "over_odds": -110,
                "under_odds": -110,
                "line_updated_at": "2026-09-22T12:01:00Z",
                "line_is_comparable": True,
            },
        ]
    )

    qb, _, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb[qb["market"].eq("qb_passing_yards")].iloc[0]
    assert row["line_book_count"] == 2
    assert row["line_options_count"] == 2
    assert row["line_min"] == 245.5
    assert row["line_max"] == 249.5
    assert row["line_range"] == 4.0
    assert row["line_consensus"] == 247.5
    assert bool(row["line_variance_flag"]) is False


def test_attach_prop_odds_uses_strict_player_prefix_when_initial_key_collides():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_ATL_GB",
                "market": "rb_rushing_yards",
                "player_name": "Bi.Robinson",
                "team": "ATL",
                "projection": 79.0,
            },
            {
                "game_id": "2026_03_ATL_GB",
                "market": "rb_rushing_yards",
                "player_name": "Br.Robinson",
                "team": "ATL",
                "projection": 35.0,
            },
        ]
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "GB Packers",
                "away_team": "ATL Falcons",
                "bookmaker_key": "draftkings",
                "market_key": "player_rush_yds",
                "outcome_name": "Over",
                "player_name": "Bijan Robinson",
                "price": -110,
                "point": 78.5,
            },
            {
                "home_team": "GB Packers",
                "away_team": "ATL Falcons",
                "bookmaker_key": "draftkings",
                "market_key": "player_rush_yds",
                "outcome_name": "Over",
                "player_name": "Brian Robinson Jr.",
                "price": -110,
                "point": 22.5,
            },
        ]
    )
    odds = prepare_player_prop_odds(raw_odds, season=2026, week=3, vendor="draftkings")

    attached = _attach_prop_odds(predictions, odds)

    lines = dict(zip(attached["player_name"], attached["line"]))
    assert lines["Bi.Robinson"] == 78.5
    assert lines["Br.Robinson"] == 22.5
    assert attached["line_range"].fillna(0).eq(0).all()


def test_attach_prop_odds_uses_identity_and_does_not_reuse_matched_line_for_collision():
    predictions = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_BAL_BUF",
                "market": "anytime_td",
                "player_id": "ty-johnson",
                "player_name": "T.Johnson",
                "team": "BUF",
                "projection": 0.22,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_BAL_BUF",
                "market": "anytime_td",
                "player_id": "taron-johnson",
                "player_name": "T.Johnson",
                "team": "BUF",
                "projection": 0.01,
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_BAL_BUF",
                "market": "anytime_td",
                "player_name_key": "t:johnson",
                "player_name_prefix_key": "ty:johnson",
                "line": 0.5,
                "implied_probability": 0.357143,
                "sportsbook": "draftkings",
                "over_odds": 180,
                "under_odds": pd.NA,
                "line_updated_at": "2026-09-22T12:00:00Z",
            }
        ]
    )
    identity = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "team": "BUF",
                "player_id": "ty-johnson",
                "alias_name": "Ty Johnson",
                "alias_name_key": "t:johnson",
                "alias_prefix_key": "ty:johnson",
                "alias_confidence": 1.0,
            },
            {
                "season": 2026,
                "week": 3,
                "team": "BUF",
                "player_id": "taron-johnson",
                "alias_name": "Taron Johnson",
                "alias_name_key": "t:johnson",
                "alias_prefix_key": "ta:johnson",
                "alias_confidence": 1.0,
            },
        ]
    )

    attached = _attach_prop_odds(predictions, odds, identity=identity)

    lines = dict(zip(attached["player_id"], attached["line"]))
    assert lines["ty-johnson"] == 0.5
    assert pd.isna(lines["taron-johnson"])


def test_attach_prop_odds_matches_condensed_initial_names_across_books():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_LAC_BUF",
                "market": "wrte_receiving_yards",
                "player_name": "D.Moore",
                "team": "BUF",
                "projection": 41.7,
            }
        ]
    )
    raw_odds = pd.DataFrame(
        [
            {
                "home_team": "Buffalo Bills",
                "away_team": "Los Angeles Chargers",
                "bookmaker_key": "draftkings",
                "market_key": "player_reception_yds",
                "outcome_name": "Over",
                "player_name": "D.J. Moore",
                "price": -113,
                "point": 49.5,
            },
            {
                "home_team": "Buffalo Bills",
                "away_team": "Los Angeles Chargers",
                "bookmaker_key": "draftkings",
                "market_key": "player_reception_yds",
                "outcome_name": "Under",
                "player_name": "D.J. Moore",
                "price": -111,
                "point": 49.5,
            },
            {
                "home_team": "Buffalo Bills",
                "away_team": "Los Angeles Chargers",
                "bookmaker_key": "fanduel",
                "market_key": "player_reception_yds",
                "outcome_name": "Over",
                "player_name": "D.J. Moore",
                "price": -114,
                "point": 49.5,
            },
            {
                "home_team": "Buffalo Bills",
                "away_team": "Los Angeles Chargers",
                "bookmaker_key": "fanduel",
                "market_key": "player_reception_yds",
                "outcome_name": "Under",
                "player_name": "D.J. Moore",
                "price": -114,
                "point": 49.5,
            },
            {
                "home_team": "Buffalo Bills",
                "away_team": "Los Angeles Chargers",
                "bookmaker_key": "hardrock",
                "market_key": "player_reception_yds",
                "outcome_name": "Over",
                "player_name": "DJ Moore",
                "price": -115,
                "point": 49.5,
            },
            {
                "home_team": "Buffalo Bills",
                "away_team": "Los Angeles Chargers",
                "bookmaker_key": "hardrock",
                "market_key": "player_reception_yds",
                "outcome_name": "Under",
                "player_name": "DJ Moore",
                "price": -115,
                "point": 49.5,
            },
        ]
    )
    odds = prepare_player_prop_odds(
        raw_odds,
        season=2026,
        week=3,
        vendor=["draftkings", "fanduel", "hardrock"],
    )

    attached = _attach_prop_odds(predictions, odds)

    row = attached.iloc[0]
    assert row["line"] == 49.5
    assert row["line_book_count"] == 3
    assert row["line_options_count"] == 3
    assert row["line_range"] == 0.0


def test_predict_player_props_flags_injury_exclusions(tmp_path):
    labels = _labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name_key": "q:one",
                "line": 245.5,
                "implied_probability": 0.5,
                "sportsbook": "draftkings",
                "over_odds": -110,
                "under_odds": -110,
                "line_updated_at": "2026-09-22T12:00:00Z",
            }
        ]
    )
    injuries = pd.DataFrame(
        [
            {
                "player_id": "qb-1",
                "injury_status": "Out",
                "injury_practice_status": "Did Not Participate In Practice",
                "injury_body_part": "Hamstring",
                "injury_report_week": 3,
                "injury_source": "nflverse",
                "injury_last_updated": "2026-09-22T12:00:00Z",
                "injury_reported": True,
                "injury_exclusion_flag": True,
                "prop_quality_flag": "injury_exclusion",
            }
        ]
    )

    qb, _, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        injuries=injuries,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb.iloc[0]
    assert row["injury_status"] == "Out"
    assert bool(row["injury_exclusion_flag"]) is True
    assert row["prop_quality_flag"] == "injury_exclusion"
    assert bool(row["bettable_flag"]) is False


def test_predict_player_props_blocks_confirmed_roster_mismatch(tmp_path):
    labels = _labels()
    models_dir = tmp_path / "models"
    _write_models(labels, models_dir)
    qb_path, offense_path, defense_path = _projection_sources(tmp_path / "predictions")
    rosters_path = tmp_path / "rosters.parquet"
    _write_rosters(rosters_path)
    candidates = build_prediction_candidates(
        labels=labels,
        qb_path=qb_path,
        offense_path=offense_path,
        defense_path=defense_path,
        rosters_path=rosters_path,
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name_key": "q:one",
                "line": 245.5,
                "implied_probability": 0.5,
                "sportsbook": "draftkings",
                "over_odds": -110,
                "under_odds": -110,
                "line_updated_at": "2026-09-22T12:00:00Z",
            }
        ]
    )
    availability = pd.DataFrame(
        [
            {
                "player_id": "qb-1",
                "player_name_key": "q:one",
                "current_team": "CAR",
                "active_roster_flag": True,
                "practice_squad_flag": False,
                "injured_reserve_flag": False,
                "availability_source": "espn",
            }
        ]
    )

    qb, _, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        availability=availability,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb.iloc[0]
    assert row["current_team"] == "CAR"
    assert row["roster_validation_flag"] == "team_mismatch"
    assert row["prop_quality_flag"] == "roster_mismatch"
    assert bool(row["bettable_flag"]) is False


def test_player_prop_predict_cli_writes_outputs(tmp_path):
    labels = _labels()
    novelty_labels = _novelty_labels()
    labels_path = tmp_path / "labels.parquet"
    novelty_labels_path = tmp_path / "novelty_labels.parquet"
    models_dir = tmp_path / "models"
    prediction_dir = tmp_path / "predictions"
    rosters_path = tmp_path / "rosters.parquet"
    labels.to_parquet(labels_path, index=False)
    novelty_labels.to_parquet(novelty_labels_path, index=False)
    _write_models(labels, models_dir, novelty_labels=novelty_labels)
    _projection_sources(prediction_dir)
    _write_rosters(rosters_path)

    rc = main(
        [
            "--season",
            "2026",
            "--week",
            "3",
            "--labels",
            str(labels_path),
            "--novelty-labels",
            str(novelty_labels_path),
            "--models-dir",
            str(models_dir),
            "--prediction-dir",
            str(prediction_dir),
            "--rosters",
            str(rosters_path),
            "--qb-output",
            str(prediction_dir / "player_props_qb.csv"),
            "--offense-output",
            str(prediction_dir / "player_props_offense.csv"),
            "--defense-output",
            str(prediction_dir / "player_props_defense.csv"),
            "--novelty-output",
            str(prediction_dir / "player_props_novelty.csv"),
        ]
    )

    assert rc == 0
    assert set(pd.read_csv(prediction_dir / "player_props_qb.csv")["market"]) == {
        "qb_passing_yards",
        "qb_passing_tds",
    }
    assert set(pd.read_csv(prediction_dir / "player_props_offense.csv")["market"]) == {
        "rb_rushing_yards",
        "rb_rushing_tds",
        "wrte_receiving_yards",
        "wrte_receiving_tds",
        "anytime_td",
    }
    assert pd.read_csv(prediction_dir / "player_props_defense.csv")["market"].tolist() == ["def_sacks"]
    assert set(pd.read_csv(prediction_dir / "player_props_novelty.csv")["market"]) == {
        "first_td",
        "last_td",
        "two_plus_tds",
        "three_plus_tds",
    }
