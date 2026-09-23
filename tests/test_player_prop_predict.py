from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.player_props.features import build_player_prop_features
from src.player_props.predict import (
    _attach_prop_odds,
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
        ("rb-1", "RB One", "RB", "RB", "rb_rushing_yards", [40.0, 60.0, 70.0, 80.0]),
        ("wr-1", "WR One", "WR", "WR", "wrte_receiving_yards", [35.0, 45.0, 55.0, 65.0]),
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
                    "value_type": "count" if market == "def_sacks" else "yards",
                    "source": "test",
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
                "projected_receiving_yards": 8.0,
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
                "projected_receiving_yards": 60.0,
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


def _write_models(labels: pd.DataFrame, models_dir: Path) -> None:
    offense_features, defense_features = build_player_prop_features(labels)
    _, _, bundles = train_and_evaluate(
        offense_features=offense_features,
        defense_features=defense_features,
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
        "rb_rushing_yards",
        "wrte_receiving_yards",
        "def_sacks",
    }
    rb = candidates[candidates["market"].eq("rb_rushing_yards")].iloc[0]
    wr = candidates[candidates["market"].eq("wrte_receiving_yards")].iloc[0]
    assert rb["player_id"] == "rb-1"
    assert wr["player_id"] == "wr-1"


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

    qb, offense, defense = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    assert qb["market"].tolist() == ["qb_passing_yards"]
    assert set(offense["market"]) == {"rb_rushing_yards", "wrte_receiving_yards"}
    assert defense["market"].tolist() == ["def_sacks"]
    assert qb["projection"].notna().all()
    assert offense["projection"].notna().all()
    assert defense["prob_over"].notna().all()
    assert int(qb.iloc[0]["sample_size"]) == 4


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
    assert row["over_odds"] == -120
    assert pd.isna(row["under_odds"])
    assert round(row["implied_probability"], 6) == round(120 / 220, 6)


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

    qb, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb.iloc[0]
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

    qb, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        over_probability_history=over_probability_history,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb.iloc[0]
    assert row["prob_over"] == 0.5
    assert row["prob_edge"] == 0.0
    assert row["prob_over_method"] == "market_implied_fallback"
    assert row["prob_over_sample_size"] == 1
    assert round(row["over_break_even_probability"], 6) == round(110 / 210, 6)
    assert round(row["under_break_even_probability"], 6) == round(110 / 210, 6)
    assert round(row["over_ev"], 6) == round((0.5 * (100 / 110)) - 0.5, 6)
    assert round(row["under_ev"], 6) == round((0.5 * (100 / 110)) - 0.5, 6)
    assert row["edge_side"] == "over"
    assert row["edge_odds"] == -110
    assert row["edge_ev"] < 0


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

    qb, _, _ = predict_player_props(
        labels=labels,
        candidates=candidates,
        models_dir=models_dir,
        odds=odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    row = qb.iloc[0]
    assert row["line_book_count"] == 2
    assert row["line_options_count"] == 2
    assert row["line_min"] == 245.5
    assert row["line_max"] == 249.5
    assert row["line_range"] == 4.0
    assert row["line_consensus"] == 247.5
    assert row["line_variance_flag"] is False


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

    qb, _, _ = predict_player_props(
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

    qb, _, _ = predict_player_props(
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
    labels_path = tmp_path / "labels.parquet"
    models_dir = tmp_path / "models"
    prediction_dir = tmp_path / "predictions"
    rosters_path = tmp_path / "rosters.parquet"
    labels.to_parquet(labels_path, index=False)
    _write_models(labels, models_dir)
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
    assert pd.read_csv(prediction_dir / "player_props_qb.csv")["market"].tolist() == ["qb_passing_yards"]
    assert set(pd.read_csv(prediction_dir / "player_props_offense.csv")["market"]) == {
        "rb_rushing_yards",
        "wrte_receiving_yards",
    }
    assert pd.read_csv(prediction_dir / "player_props_defense.csv")["market"].tolist() == ["def_sacks"]
    assert (prediction_dir / "player_props_novelty.csv").exists()
