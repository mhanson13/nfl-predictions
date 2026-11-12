from __future__ import annotations

import pandas as pd
import pytest

from src.player_props.quality_report import (
    build_line_gap_audit,
    build_line_coverage_report,
    build_quality_reports,
    build_roster_line_audit,
    load_prop_predictions,
    main,
)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_group": "qb",
                "market": "qb_passing_yards",
                "line": 245.5,
                "edge": 4.5,
                "sportsbook": "draftkings",
                "line_range": 3.0,
                "line_book_count": 2,
                "line_variance_flag": False,
                "injury_reported": False,
                "injury_exclusion_flag": False,
                "prop_quality_flag": "ok",
                "roster_validation_flag": "ok",
                "bettable_flag": True,
                "edge_recommendation_flag": True,
                "edge_recommendation_tier": "strong",
                "edge_probability_edge": 0.08,
                "edge_ev": 0.12,
            },
            {
                "prediction_group": "offense",
                "market": "rb_rushing_yards",
                "line": 91.5,
                "edge": -2.0,
                "sportsbook": "fanduel",
                "line_range": 20.0,
                "line_book_count": 3,
                "line_variance_flag": True,
                "injury_reported": True,
                "injury_exclusion_flag": False,
                "prop_quality_flag": "injury_review",
                "roster_validation_flag": "team_mismatch",
                "bettable_flag": True,
                "edge_recommendation_flag": False,
                "edge_recommendation_tier": "below_threshold",
                "edge_probability_edge": 0.01,
                "edge_ev": -0.02,
            },
            {
                "prediction_group": "offense",
                "market": "wrte_receiving_yards",
                "line": None,
                "edge": None,
                "sportsbook": "",
                "line_range": None,
                "line_book_count": None,
                "line_variance_flag": False,
                "injury_reported": True,
                "injury_exclusion_flag": True,
                "prop_quality_flag": "injury_exclusion",
                "roster_validation_flag": "inactive_roster",
                "bettable_flag": False,
                "edge_recommendation_flag": False,
                "edge_recommendation_tier": "unavailable",
                "edge_probability_edge": None,
                "edge_ev": None,
            },
        ]
    )


def test_build_quality_reports_summarizes_bettable_lines_and_flags():
    detail, summary = build_quality_reports(
        _sample_predictions(),
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    overall = summary[summary["scope"].eq("overall")].iloc[0]
    assert overall["rows"] == 3
    assert overall["rows_with_lines"] == 2
    assert overall["exact_line_rows"] == 2
    assert overall["fallback_line_rows"] == 0
    assert overall["missing_line_rows"] == 1
    assert overall["bettable_rows"] == 2
    assert overall["edge_recommended_rows"] == 1
    assert overall["edge_strong_rows"] == 1
    assert overall["edge_below_threshold_rows"] == 1
    assert overall["edge_no_positive_ev_rows"] == 0
    assert overall["edge_unavailable_rows"] == 1
    assert overall["injury_review_rows"] == 1
    assert overall["injury_exclusion_rows"] == 1
    assert overall["roster_validated_rows"] == 3
    assert overall["roster_unknown_rows"] == 0
    assert overall["roster_mismatch_rows"] == 1
    assert overall["roster_exclusion_rows"] == 1
    assert overall["line_variance_rows"] == 1
    assert overall["avg_line_range"] == 11.5
    assert overall["avg_abs_edge"] == 3.25
    assert overall["avg_edge_probability_edge"] == pytest.approx(0.045)
    assert overall["avg_edge_ev"] == pytest.approx(0.05)

    no_line = detail[detail["sportsbook"].eq("no_line")].iloc[0]
    assert no_line["market"] == "wrte_receiving_yards"
    assert no_line["rows_with_lines"] == 0
    assert no_line["missing_line_rows"] == 1


def test_build_line_coverage_report_counts_matchable_and_unmatched_lines():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name": "QB One",
                "line": 245.5,
                "sportsbook": "draftkings",
            },
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "rb_rushing_yards",
                "player_name": "RB One",
                "line": None,
                "sportsbook": "",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
            },
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Under",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
            },
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_rush_yds",
                "outcome_name": "Over",
                "player_name": "Other Back",
                "price": -120,
                "point": 50.5,
            },
        ]
    )

    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    overall = coverage[coverage["scope"].eq("overall")].iloc[0]
    assert overall["raw_rows"] == 3
    assert overall["prepared_line_rows"] == 2
    assert overall["offered_game_player_pairs"] == 2
    assert overall["prediction_rows"] == 2
    assert overall["matchable_prediction_rows"] == 1
    assert overall["published_line_rows"] == 1
    assert overall["offered_pairs_not_in_predictions"] == 1


def test_line_coverage_counts_published_lines_as_matchable_when_identity_path_attached_them():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name": "Q.One",
                "line": 245.5,
                "sportsbook": "draftkings",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name_key": "quentin:one",
                "player_name_prefix_key": "qu:one",
                "sportsbook": "draftkings",
                "line": 245.5,
            },
        ]
    )

    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    overall = coverage[coverage["scope"].eq("overall")].iloc[0]
    assert overall["published_line_rows"] == 1
    assert overall["matchable_prediction_rows"] == 1
    assert overall["prediction_rows_without_offered_line"] == 0

    book = coverage[
        coverage["scope"].eq("market_sportsbook") & coverage["sportsbook"].eq("draftkings")
    ].iloc[0]
    assert book["published_line_rows"] == 1
    assert book["matchable_prediction_rows"] == 1


def test_line_coverage_marks_matched_book_as_available_not_selected_when_another_book_was_selected():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name": "QB One",
                "line": 245.5,
                "sportsbook": "draftkings",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
                "commence_time": "2026-09-24T00:15:00Z",
                "market_last_update": "2026-09-22T12:00:00Z",
            },
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "hardrock",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -105,
                "point": 245.5,
                "commence_time": "2026-09-24T00:15:00Z",
                "market_last_update": "2026-09-22T12:30:00Z",
            },
        ]
    )

    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=2026,
        week=3,
        vendors=["draftkings", "hardrock"],
        generated_at="2026-09-22T00:15:00+00:00",
    )

    hardrock = coverage[
        coverage["scope"].eq("market_sportsbook")
        & coverage["market"].eq("qb_passing_yards")
        & coverage["sportsbook"].eq("hardrock")
    ].iloc[0]

    assert hardrock["matchable_prediction_rows"] == 1
    assert hardrock["published_line_rows"] == 0
    assert hardrock["selected_elsewhere_rows"] == 1
    assert hardrock["coverage_status"] == "available_not_selected"
    assert hardrock["raw_games"] == 1
    assert hardrock["earliest_kickoff"] == "2026-09-24T00:15:00+00:00"
    assert hardrock["latest_market_update"] == "2026-09-22T12:30:00+00:00"
    assert hardrock["hours_until_earliest_kickoff"] == 48.0


def test_line_coverage_does_not_count_identity_matched_offer_as_missing_prediction():
    predictions = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CAR_ATL",
                "market": "def_sacks",
                "team": "CAR",
                "player_id": "edge-2",
                "player_name": "E.Two",
                "line": 1.0,
                "sportsbook": "fanduel",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "def_sacks",
                "player_name_key": "e:two",
                "player_name_prefix_key": "ed:two",
                "sportsbook": "fanduel",
                "line": 1.0,
            },
        ]
    )
    identity = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "team": "CAR",
                "player_id": "edge-2",
                "alias_name": "EDGE Two",
                "alias_name_key": "e:two",
                "alias_prefix_key": "ed:two",
                "alias_confidence": 1.0,
            }
        ]
    )

    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=2026,
        week=3,
        identity=identity,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    market = coverage[coverage["scope"].eq("market") & coverage["market"].eq("def_sacks")].iloc[0]
    assert market["published_line_rows"] == 1
    assert market["offered_pairs_not_in_predictions"] == 0


def test_line_coverage_reports_requested_zero_row_markets_with_status():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "rb_rushing_tds",
                "player_name": "RB One",
                "line": None,
                "sportsbook": "",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_pass_yds",
                "outcome_name": "Over",
                "player_name": "QB One",
                "price": -110,
                "point": 245.5,
                "requested_markets": "player_pass_yds,player_rush_tds,player_1st_td",
                "requested_bookmakers": "draftkings,fanduel",
            },
        ]
    )

    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    rushing_td = coverage[
        coverage["scope"].eq("market")
        & coverage["market"].eq("rb_rushing_tds")
        & coverage["sportsbook"].eq("all")
    ].iloc[0]
    assert rushing_td["raw_rows"] == 0
    assert rushing_td["prediction_rows"] == 1
    assert rushing_td["coverage_status"] == "vendor_no_rows"

    first_td = coverage[
        coverage["scope"].eq("market")
        & coverage["market"].eq("first_td")
        & coverage["sportsbook"].eq("all")
    ].iloc[0]
    assert first_td["raw_rows"] == 0
    assert first_td["prediction_rows"] == 0
    assert first_td["coverage_status"] == "vendor_no_rows"


def test_line_coverage_counts_anytime_td_fallback_for_stat_td_markets():
    predictions = pd.DataFrame(
        [
            {
                "game_id": "2026_03_CAR_ATL",
                "market": "rb_rushing_tds",
                "player_name": "RB One",
                "line": 0.5,
                "line_type": "anytime_td_fallback",
                "sportsbook": "draftkings",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_anytime_td",
                "outcome_name": "Anytime Touchdown Scorer",
                "player_name": "RB One",
                "price": 150,
                "point": None,
                "requested_markets": "player_anytime_td,player_rush_tds",
                "requested_bookmakers": "draftkings",
            },
        ]
    )

    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=2026,
        week=3,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    rushing_td = coverage[
        coverage["scope"].eq("market")
        & coverage["market"].eq("rb_rushing_tds")
        & coverage["sportsbook"].eq("all")
    ].iloc[0]
    assert rushing_td["raw_rows"] == 0
    assert rushing_td["prepared_line_rows"] == 1
    assert rushing_td["exact_prepared_line_rows"] == 0
    assert rushing_td["fallback_prepared_line_rows"] == 1
    assert rushing_td["matchable_prediction_rows"] == 1
    assert rushing_td["published_line_rows"] == 1
    assert rushing_td["exact_published_line_rows"] == 0
    assert rushing_td["fallback_published_line_rows"] == 1
    assert rushing_td["coverage_status"] == "fallback_only"


def test_build_roster_line_audit_expands_unknown_rows_by_sportsbook_line_availability():
    predictions = pd.DataFrame(
        [
            {
                "prediction_group": "offense",
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "current_team": "ATL",
                "player_id": "rb1",
                "player_name": "RB One",
                "position": "RB",
                "position_group": "RB",
                "market": "rb_rushing_yards",
                "line": 50.5,
                "sportsbook": "draftkings",
                "over_odds": -110,
                "under_odds": -110,
                "line_book_count": 2,
                "roster_validation_flag": "unknown",
                "roster_validation_source": "espn",
                "availability_status": "unknown",
                "active_current_roster": False,
                "prop_quality_flag": "ok",
            },
            {
                "prediction_group": "offense",
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "current_team": pd.NA,
                "player_id": "rb2",
                "player_name": "RB Two",
                "position": "RB",
                "position_group": "RB",
                "market": "rb_rushing_yards",
                "line": pd.NA,
                "sportsbook": "",
                "roster_validation_flag": "unknown",
                "availability_status": "unknown",
                "prop_quality_flag": "ok",
            },
            {
                "prediction_group": "qb",
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "player_id": "qb1",
                "player_name": "QB One",
                "market": "qb_passing_yards",
                "line": 245.5,
                "sportsbook": "draftkings",
                "roster_validation_flag": "ok",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": book,
                "market_key": "player_rush_yds",
                "outcome_name": side,
                "player_name": "RB One",
                "price": -110,
                "point": 50.5,
            }
            for book in ["draftkings", "fanduel"]
            for side in ["Over", "Under"]
        ]
        + [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_rush_yds",
                "outcome_name": "Over",
                "player_name": "Other Back",
                "price": -120,
                "point": 44.5,
            }
        ]
    )

    audit = build_roster_line_audit(
        predictions,
        odds,
        season=2026,
        week=3,
        vendors=["draftkings", "fanduel", "hardrock"],
        generated_at="2026-09-22T00:00:00+00:00",
    )

    assert set(audit["player_id"]) == {"rb1", "rb2"}
    assert len(audit) == 6

    rb1_dk = audit[audit["player_id"].eq("rb1") & audit["sportsbook"].eq("draftkings")].iloc[0]
    assert rb1_dk["line_availability_status"] == "published"
    assert bool(rb1_dk["published_line_flag"]) is True
    assert rb1_dk["matchable_line_rows"] == 1

    rb1_fd = audit[audit["player_id"].eq("rb1") & audit["sportsbook"].eq("fanduel")].iloc[0]
    assert rb1_fd["line_availability_status"] == "matchable_not_published"
    assert rb1_fd["matchable_line_rows"] == 1

    rb2_dk = audit[audit["player_id"].eq("rb2") & audit["sportsbook"].eq("draftkings")].iloc[0]
    assert rb2_dk["line_availability_status"] == "no_matching_line"
    assert rb2_dk["raw_rows"] == 3
    assert rb2_dk["prepared_line_rows"] == 2

    rb2_hr = audit[audit["player_id"].eq("rb2") & audit["sportsbook"].eq("hardrock")].iloc[0]
    assert rb2_hr["line_availability_status"] == "vendor_no_rows"


def test_build_line_gap_audit_reports_prediction_and_offer_gaps():
    predictions = pd.DataFrame(
        [
            {
                "prediction_group": "offense",
                "game_id": "2026_03_CAR_ATL",
                "team": "ATL",
                "player_id": "rb1",
                "player_name": "RB One",
                "position": "RB",
                "position_group": "RB",
                "market": "rb_rushing_yards",
                "line": pd.NA,
                "sportsbook": "",
                "prop_quality_flag": "ok",
                "roster_validation_flag": "ok",
                "availability_status": "active",
            },
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_rush_yds",
                "outcome_name": "Over",
                "player_name": "Other Back",
                "price": -110,
                "point": 44.5,
            },
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": "player_rush_yds",
                "outcome_name": "Under",
                "player_name": "Other Back",
                "price": -110,
                "point": 44.5,
            },
        ]
    )

    audit = build_line_gap_audit(
        predictions,
        odds,
        season=2026,
        week=3,
        vendors=["draftkings"],
        generated_at="2026-09-22T00:00:00+00:00",
    )

    assert set(audit["gap_type"]) == {"prediction_without_offered_line", "offered_pair_without_prediction"}
    prediction_gap = audit[audit["gap_type"].eq("prediction_without_offered_line")].iloc[0]
    assert prediction_gap["player_id"] == "rb1"
    assert "no prepared PropLine offer matched" in prediction_gap["reason"]
    offer_gap = audit[audit["gap_type"].eq("offered_pair_without_prediction")].iloc[0]
    assert offer_gap["sportsbook"] == "draftkings"
    assert offer_gap["offered_player_name_key"] == "o:back"
    assert offer_gap["offer_gap_category"] == "identity_unmatched_offer"
    assert "could not be matched" in offer_gap["reason"]


def test_build_line_gap_audit_classifies_offer_gaps_with_identity_context():
    predictions = pd.DataFrame()
    odds = pd.DataFrame(
        [
            {
                "home_team": "ATL Falcons",
                "away_team": "CAR Panthers",
                "bookmaker_key": "draftkings",
                "market_key": market_key,
                "outcome_name": outcome_name,
                "player_name": player_name,
                "price": -110,
                "point": point,
            }
            for market_key, outcome_name, player_name, point in [
                ("player_anytime_td", "Anytime Touchdown Scorer", "ATL Falcons D/ST", None),
                ("player_1st_td", "First Touchdown Scorer", "No Touchdown Scorer", None),
                ("player_rush_yds", "Over", "QB One", 34.5),
                ("player_rush_yds", "Under", "QB One", 34.5),
                ("player_rush_yds", "Over", "RB One", 62.5),
                ("player_rush_yds", "Under", "RB One", 62.5),
            ]
        ]
    )
    identity = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "team": "ATL",
                "player_id": "qb-1",
                "alias_name": "QB One",
                "position": "QB",
                "position_group": "QB",
                "alias_confidence": 1.0,
            },
            {
                "season": 2026,
                "week": 3,
                "team": "ATL",
                "player_id": "rb-1",
                "alias_name": "RB One",
                "position": "RB",
                "position_group": "RB",
                "alias_confidence": 1.0,
            },
        ]
    )
    availability = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "player_id": player_id,
                "current_team": "ATL",
                "active_roster_flag": True,
                "practice_squad_flag": False,
                "injured_reserve_flag": False,
                "availability_status": "active",
            }
            for player_id in ["qb-1", "rb-1"]
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "player_id": "rb-1",
                "market": "rb_rushing_yards",
            }
        ]
    )

    audit = build_line_gap_audit(
        predictions,
        odds,
        season=2026,
        week=3,
        vendors=["draftkings"],
        identity=identity,
        availability=availability,
        labels=labels,
        generated_at="2026-09-22T00:00:00+00:00",
    )

    by_player = audit.set_index("player_name")
    assert by_player.loc["ATL Falcons D/ST", "offer_gap_category"] == "non_player_dst"
    assert by_player.loc["No Touchdown Scorer", "offer_gap_category"] == "non_player_no_touchdown_scorer"
    assert by_player.loc["QB One", "offer_gap_category"] == "unsupported_position_for_modeled_market"
    assert by_player.loc["QB One", "offered_position_group"] == "QB"
    assert by_player.loc["RB One", "offer_gap_category"] == "candidate_expansion_gap"
    assert by_player.loc["RB One", "offered_player_id"] == "rb-1"
    assert "true prediction-universe expansion candidate" in by_player.loc["RB One", "reason"]


def test_load_prop_predictions_includes_novelty_file(tmp_path):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    pd.DataFrame(
        [
            {
                "market": "first_td",
                "player_name": "TD One",
                "line": None,
                "sportsbook": "",
                "bettable_flag": False,
                "edge_recommendation_tier": "research_only",
            },
        ]
    ).to_csv(prediction_dir / "player_props_novelty.csv", index=False)

    predictions = load_prop_predictions(prediction_dir)

    assert len(predictions) == 1
    assert predictions.iloc[0]["prediction_group"] == "novelty"
    assert predictions.iloc[0]["market"] == "first_td"


def test_quality_report_cli_writes_weekly_outputs(tmp_path):
    prediction_dir = tmp_path / "predictions"
    output_dir = prediction_dir / "evaluation" / "player_props"
    prediction_dir.mkdir()
    _sample_predictions().query("prediction_group == 'qb'").to_csv(prediction_dir / "player_props_qb.csv", index=False)
    _sample_predictions().query("prediction_group == 'offense'").to_csv(
        prediction_dir / "player_props_offense.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "market": "first_td",
                "player_name": "TD One",
                "line": None,
                "sportsbook": "",
                "bettable_flag": False,
                "edge_recommendation_tier": "research_only",
            },
        ]
    ).to_csv(prediction_dir / "player_props_novelty.csv", index=False)

    rc = main(
        [
            "--season",
            "2026",
            "--week",
            "3",
            "--prediction-dir",
            str(prediction_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    assert (output_dir / "prop_quality_week_03.csv").exists()
    assert (output_dir / "prop_quality_summary_week_03.csv").exists()
    assert (output_dir / "prop_line_coverage_week_03.csv").exists()
    assert (output_dir / "prop_roster_line_audit_week_03.csv").exists()
    assert (output_dir / "prop_line_gap_audit_week_03.csv").exists()
    summary = pd.read_csv(output_dir / "prop_quality_summary_week_03.csv")
    assert "novelty" in set(summary["prediction_group"])
    assert summary[summary["scope"].eq("overall")].iloc[0]["rows"] == 4
