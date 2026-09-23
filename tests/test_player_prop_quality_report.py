from __future__ import annotations

import pandas as pd

from src.player_props.quality_report import build_line_coverage_report, build_quality_reports, main


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
    assert overall["missing_line_rows"] == 1
    assert overall["bettable_rows"] == 2
    assert overall["injury_review_rows"] == 1
    assert overall["injury_exclusion_rows"] == 1
    assert overall["roster_validated_rows"] == 3
    assert overall["roster_unknown_rows"] == 0
    assert overall["roster_mismatch_rows"] == 1
    assert overall["roster_exclusion_rows"] == 1
    assert overall["line_variance_rows"] == 1
    assert overall["avg_line_range"] == 11.5
    assert overall["avg_abs_edge"] == 3.25

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


def test_quality_report_cli_writes_weekly_outputs(tmp_path):
    prediction_dir = tmp_path / "predictions"
    output_dir = prediction_dir / "evaluation" / "player_props"
    prediction_dir.mkdir()
    _sample_predictions().query("prediction_group == 'qb'").to_csv(prediction_dir / "player_props_qb.csv", index=False)
    _sample_predictions().query("prediction_group == 'offense'").to_csv(
        prediction_dir / "player_props_offense.csv",
        index=False,
    )

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
