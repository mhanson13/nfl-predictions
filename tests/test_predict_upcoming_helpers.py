import pandas as pd

import src.predict.predict_upcoming as predict_upcoming
from src.predict.predict_upcoming import (
    QB_PLAYER_PROJECTION_COLUMNS,
    _apply_probability_calibrator,
    _best_probability_signal,
    _cleanup_prediction_columns,
    _attach_market_policy_fields,
    _attach_odds_spreads,
    _fallback_if_probability_collapsed,
    _filter_unstarted_games,
    _has_probability_signal,
    _is_collapsed_probability,
    _is_saturated_probability_signal,
    _player_qb_predictions,
    _should_skip_volatility_shrinkage,
    _stabilize_volatility_threshold_for_slate,
    _volatility_artifact_skip_reason,
)


def test_probability_signal_helpers_detect_collapsed_and_variable_series():
    assert _is_collapsed_probability(pd.Series([0.15, 0.15, 0.15]))
    assert not _has_probability_signal(pd.Series([0.15, 0.15, 0.15]))

    assert not _is_collapsed_probability(pd.Series([0.11, 0.15, 0.23]))
    assert _has_probability_signal(pd.Series([0.11, 0.15, 0.23]))
    assert _is_saturated_probability_signal(pd.Series([0.98, 0.98, 0.99, 1.0]))
    assert not _is_saturated_probability_signal(pd.Series([0.58, 0.62, 0.66, 0.69]))


def test_fallback_restores_signal_when_stage_collapses():
    candidate = pd.Series([0.15, 0.15, 0.15])
    fallback = pd.Series([0.03, 0.12, 0.24])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="caps",
    )

    assert reason == "caps_collapsed"
    assert restored.tolist() == [0.03, 0.12, 0.24]


def test_fallback_clips_extreme_restored_probabilities():
    candidate = pd.Series([0.15, 0.15, 0.15])
    fallback = pd.Series([0.0, 0.5, 1.0])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="calibration",
    )

    assert reason == "calibration_collapsed"
    assert restored.tolist() == [0.02, 0.5, 0.98]


def test_fallback_does_not_clip_away_near_one_signal():
    candidate = pd.Series([0.98, 0.98, 0.98])
    fallback = pd.Series([0.996, 0.998, 1.0])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="volatility",
    )

    assert reason == "volatility_collapsed"
    assert restored.tolist() == [0.996, 0.998, 0.999]


def test_fallback_restores_signal_when_stage_severely_compresses():
    candidate = pd.Series([0.15, 0.15, 0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
    fallback = pd.Series([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.17, 0.19])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="caps",
    )

    assert reason == "caps_collapsed"
    assert restored.tolist() == fallback.tolist()


def test_fallback_restores_signal_when_stage_range_compresses():
    candidate = pd.Series([0.8225, 0.8230, 0.8234, 0.8240])
    fallback = pd.Series([0.62, 0.64, 0.66, 0.68])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="volatility",
    )

    assert reason == "volatility_collapsed"
    assert restored.tolist() == fallback.tolist()


def test_fallback_restores_raw_signal_when_calibration_saturates():
    candidate = pd.Series([0.78, 0.96, 0.97, 0.98, 0.979, 0.976, 0.974, 0.971])
    fallback = pd.Series([0.54, 0.65, 0.66, 0.68, 0.67, 0.66, 0.65, 0.64])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="calibration",
    )

    assert reason == "calibration_saturated"
    assert restored.tolist() == fallback.tolist()


def test_fallback_allows_unsaturated_calibration_shift():
    candidate = pd.Series([0.58, 0.62, 0.65, 0.69])
    fallback = pd.Series([0.54, 0.58, 0.61, 0.64])

    restored, reason = _fallback_if_probability_collapsed(
        candidate,
        fallback,
        stage="calibration",
    )

    assert reason is None
    assert restored.tolist() == candidate.tolist()


def test_best_probability_signal_skips_collapsed_columns():
    df = pd.DataFrame(
        {
            "home_win_prob_capped": [0.98, 0.98, 0.98],
            "home_win_prob_calibrated": [0.41, 0.52, 0.63],
            "home_win_prob_model_raw": [0.31, 0.33, 0.35],
        }
    )

    signal, source = _best_probability_signal(df, df.index)

    assert source == "home_win_prob_calibrated"
    assert signal.tolist() == [0.41, 0.52, 0.63]


def test_best_probability_signal_prefers_usable_capped_probabilities():
    df = pd.DataFrame(
        {
            "home_win_prob_capped": [0.15, 0.23, 0.57],
            "home_win_prob_calibrated": [0.00, 0.035, 0.571],
            "home_win_prob_model_raw": [0.17, 0.18, 0.54],
        }
    )

    signal, source = _best_probability_signal(df, df.index)

    assert source == "home_win_prob_capped"
    assert signal.tolist() == [0.15, 0.23, 0.57]


def test_best_probability_signal_prefers_widest_available_range():
    df = pd.DataFrame(
        {
            "home_win_prob_capped": [0.996, 0.997, 0.998],
            "home_win_prob_calibrated": [0.996, 0.998, 0.999],
            "home_win_prob_model_raw": [0.62, 0.65, 0.68],
        }
    )

    signal, source = _best_probability_signal(df, df.index)

    assert source == "home_win_prob_model_raw"
    assert signal.tolist() == [0.62, 0.65, 0.68]


def test_should_skip_volatility_shrinkage_when_coverage_is_extreme():
    skip, coverage = _should_skip_volatility_shrinkage([1, 1, 1, 0], min_coverage=0.02, max_coverage=0.70)

    assert skip
    assert coverage == 0.75

    skip, coverage = _should_skip_volatility_shrinkage([0, 0, 0, 0], min_coverage=0.02, max_coverage=0.70)

    assert skip
    assert coverage == 0.0

    skip, coverage = _should_skip_volatility_shrinkage([1, 0, 0, 0], min_coverage=0.02, max_coverage=0.70)

    assert not skip
    assert coverage == 0.25


def test_stabilize_volatility_threshold_floors_sparse_slate_coverage():
    threshold, coverage, reason = _stabilize_volatility_threshold_for_slate(
        pd.Series([0.61, 0.48, 0.42, 0.31, 0.20, 0.10]),
        0.60,
        min_coverage=0.30,
        max_coverage=0.80,
    )

    assert threshold == 0.48
    assert coverage == 2 / 6
    assert reason == "coverage_floor"


def test_stabilize_volatility_threshold_keeps_explicitly_usable_threshold():
    threshold, coverage, reason = _stabilize_volatility_threshold_for_slate(
        pd.Series([0.61, 0.58, 0.42, 0.31]),
        0.50,
        min_coverage=0.25,
        max_coverage=0.75,
    )

    assert threshold == 0.50
    assert coverage == 0.50
    assert reason is None


def test_cleanup_preserves_zero_volatility_diagnostics():
    df = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "volatility_prob": [0.0, 0.0],
            "volatility_label": [0, 0],
            "unused_all_zero": [0, 0],
        }
    )

    cleaned = _cleanup_prediction_columns(df)

    assert "volatility_prob" in cleaned.columns
    assert "volatility_label" in cleaned.columns
    assert "unused_all_zero" not in cleaned.columns


def test_volatility_artifact_skip_reason_rejects_weak_artifacts():
    artifact = type("Artifact", (), {"metrics": {"auc": 0.49, "pred_positive_rate": 0.40}})()

    assert _volatility_artifact_skip_reason(artifact).startswith("auc=")

    artifact = type("Artifact", (), {"metrics": {"auc": 0.61, "pred_positive_rate": 0.01}})()

    assert _volatility_artifact_skip_reason(artifact).startswith("pred_positive_rate=")

    artifact = type("Artifact", (), {"metrics": {"auc": 0.61, "pred_positive_rate": 0.20}})()

    assert _volatility_artifact_skip_reason(artifact) is None


def test_apply_probability_calibrator_uses_predict_proba_when_available():
    class FakeClassifier:
        def predict_proba(self, arr):
            return [[1 - float(value[0]), float(value[0])] for value in arr]

        def predict(self, arr):
            return [1 for _ in arr]

    result = _apply_probability_calibrator(FakeClassifier(), pd.Series([0.2, 0.8]))

    assert result.tolist() == [0.2, 0.8]


def test_filter_unstarted_games_drops_started_kickoffs_and_keeps_unknowns():
    games = pd.DataFrame(
        [
            {"game_id": "started", "kickoff": "2026-09-25T00:15:00Z"},
            {"game_id": "future", "kickoff": "2026-09-27T17:00:00Z"},
            {"game_id": "unknown", "kickoff": None},
        ]
    )

    result = _filter_unstarted_games(
        games,
        now=pd.Timestamp("2026-09-26T12:00:00Z"),
    )

    assert result["game_id"].tolist() == ["future", "unknown"]


def test_filter_unstarted_games_treats_date_only_as_end_of_day():
    games = pd.DataFrame(
        [
            {"game_id": "thursday", "date": "2026-09-24"},
            {"game_id": "sunday", "date": "2026-09-27"},
        ]
    )

    result = _filter_unstarted_games(
        games,
        now=pd.Timestamp("2026-09-27T15:00:00Z"),
    )

    assert result["game_id"].tolist() == ["sunday"]


def test_filter_unstarted_games_prefers_gameday_gametime_over_stale_kickoff():
    games = pd.DataFrame(
        [
            {
                "game_id": "started",
                "gameday": "2026-10-04",
                "gametime": "13:00",
                "kickoff": "2026-10-04T23:00:00Z",
            },
            {
                "game_id": "future",
                "gameday": "2026-10-04",
                "gametime": "20:20",
                "kickoff": "2026-10-04T23:00:00Z",
            },
        ]
    )

    result = _filter_unstarted_games(
        games,
        now=pd.Timestamp("2026-10-04T19:12:00Z"),
    )

    assert result["game_id"].tolist() == ["future"]


def test_qb_projection_writes_empty_current_file_when_no_rows_survive(tmp_path):
    pred_games = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
            }
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "season_type": 2,
                "recent_team": "CCC",
                "player_id": "00-test",
                "player_name": "Test QB",
                "position_group": "QB",
                "passing_yards": 100,
                "passing_tds": 1,
                "rushing_tds": 0,
                "attempts": 20,
                "interceptions": 0,
                "completions": 10,
                "rushing_yards": 5,
            }
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "season": 2026,
                "team": "BBB",
                "gsis_id": "00-other",
                "jersey_number": 1,
                "full_name": "Other QB",
                "status": "ACT",
                "position": "QB",
                "depth_chart_position": "QB",
            }
        ]
    )
    out_path = tmp_path / "players_qb.csv"

    result = _player_qb_predictions(pred_games, str(out_path), stats, roster, debug=True)

    assert result is not None
    assert result.empty
    assert out_path.exists()
    exported = pd.read_csv(out_path)
    assert exported.empty
    assert list(exported.columns) == QB_PLAYER_PROJECTION_COLUMNS


def test_load_player_stats_frame_accepts_release_team_and_reg_season_type(tmp_path, monkeypatch):
    raw_path = tmp_path / "nfl_player_stats.parquet"
    raw_path.touch()
    raw = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "season_type": "REG",
                "team": "den",
                "player_id": "00-test",
                "player_name": "Test Player",
            },
            {
                "season": 2026,
                "week": 2,
                "season_type": "POST",
                "team": "KC",
                "player_id": "00-post",
                "player_name": "Post Player",
            },
        ]
    )

    monkeypatch.setattr(predict_upcoming, "RAW_DIR", tmp_path)
    monkeypatch.setattr(predict_upcoming, "read_df", lambda path: raw)

    result = predict_upcoming._load_player_stats_frame()

    assert result["player_id"].tolist() == ["00-test"]
    assert result["recent_team"].tolist() == ["DEN"]


def test_attach_odds_spreads_includes_moneyline_columns():
    preds = pd.DataFrame(
        [
            {
                "home_team": "DAL",
                "away_team": "TB",
            }
        ]
    )
    odds = [
        {
            "home_team": "Dallas Cowboys",
            "away_team": "Tampa Bay Buccaneers",
            "commence_time": "2026-10-09T00:15:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "last_update": "2026-10-06T22:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Dallas Cowboys", "price": -130},
                                {"name": "Tampa Bay Buccaneers", "price": 110},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Dallas Cowboys", "point": -2.5, "price": -110},
                                {"name": "Tampa Bay Buccaneers", "point": 2.5, "price": -110},
                            ],
                        },
                    ],
                }
            ],
        }
    ]

    columns = _attach_odds_spreads(preds, odds)

    assert "odds_draftkings_home_moneyline" in columns
    assert "odds_draftkings_away_moneyline" in columns
    assert preds.loc[0, "odds_draftkings_home_moneyline"] == -130
    assert preds.loc[0, "odds_draftkings_away_moneyline"] == 110


def test_attach_market_policy_fields_marks_moneyline_and_blocks_spread(tmp_path):
    policy_path = tmp_path / "market_edge_policy.csv"
    pd.DataFrame(
        [
            {
                "signal": "moneyline_disagreement",
                "recommended": True,
                "recommendation_tier": "strong",
                "threshold": 0.02,
            },
            {
                "signal": "spread_threshold",
                "recommended": False,
                "recommendation_tier": "no_bet",
                "threshold": 2.0,
            },
            {
                "signal": "spread_edge_bin",
                "recommended": True,
                "recommendation_tier": "lean",
                "edge_bin": "2-3",
            },
        ]
    ).to_csv(policy_path, index=False)
    preds = pd.DataFrame(
        [
            {
                    "home_team": "DAL",
                    "away_team": "TB",
                    "home_win_prob": 0.52,
                "pred_home_margin": 6.0,
                "odds_draftkings_home_moneyline": 120,
                "odds_draftkings_away_moneyline": -140,
                "odds_draftkings_home_spread": 1.0,
                "odds_draftkings_away_spread": -1.0,
            },
                {
                    "home_team": "JAX",
                    "away_team": "PHI",
                    "home_win_prob": 0.51,
                    "pred_home_margin": 1.0,
                    "odds_draftkings_home_moneyline": -105,
                    "odds_draftkings_away_moneyline": -105,
                    "odds_draftkings_home_spread": -3.5,
                    "odds_draftkings_away_spread": 3.5,
                },
                {
                    "home_team": "DEN",
                    "away_team": "KC",
                    "home_win_prob": 0.49,
                    "pred_home_margin": -12.0,
                    "odds_draftkings_home_moneyline": 200,
                    "odds_draftkings_away_moneyline": -230,
                    "odds_draftkings_home_spread": 6.0,
                "odds_draftkings_away_spread": -6.0,
            },
        ]
    )

    columns = _attach_market_policy_fields(preds, policy_path=policy_path)

    assert "market_policy_status" in columns
    assert bool(preds.loc[0, "market_moneyline_policy_recommendation_flag"]) is True
    assert preds.loc[0, "market_moneyline_policy_status"] == "strong_moneyline_disagreement"
    assert preds.loc[0, "market_policy_primary_signal"] == "moneyline_disagreement"
    assert bool(preds.loc[1, "market_spread_policy_recommendation_flag"]) is True
    assert preds.loc[1, "market_spread_policy_status"] == "lean_spread_research"
    assert preds.loc[1, "market_policy_primary_signal"] == "spread_edge_bin"
    assert bool(preds.loc[2, "market_spread_policy_recommendation_flag"]) is False
    assert preds.loc[2, "market_spread_policy_status"] == "blocked_overconfident_spread_edge"
    assert preds.loc[2, "market_policy_tier"] == "blocked"


def test_attach_market_policy_fields_surfaces_clv_blocked_moneyline_policy(tmp_path):
    policy_path = tmp_path / "market_edge_policy.csv"
    pd.DataFrame(
        [
            {
                "signal": "moneyline_disagreement",
                "recommended": False,
                "recommendation_tier": "review",
                "threshold": 0.02,
                "clv_status": "insufficient_tracked_games",
            },
        ]
    ).to_csv(policy_path, index=False)
    preds = pd.DataFrame(
        [
            {
                "home_team": "DAL",
                "away_team": "TB",
                "home_win_prob": 0.52,
                "pred_home_margin": 6.0,
                "odds_draftkings_home_moneyline": 120,
                "odds_draftkings_away_moneyline": -140,
                "odds_draftkings_home_spread": 1.0,
                "odds_draftkings_away_spread": -1.0,
            },
        ]
    )

    _attach_market_policy_fields(preds, policy_path=policy_path)

    assert bool(preds.loc[0, "market_moneyline_policy_recommendation_flag"]) is False
    assert preds.loc[0, "market_moneyline_policy_tier"] == "review"
    assert preds.loc[0, "market_moneyline_policy_status"] == "insufficient_tracked_games"
    assert preds.loc[0, "market_policy_primary_signal"] == "moneyline_review"
    assert preds.loc[0, "market_policy_status"] == "insufficient_tracked_games"


def test_attach_market_policy_fields_blocks_spread_when_calibration_policy_fails(tmp_path):
    policy_path = tmp_path / "market_edge_policy.csv"
    pd.DataFrame(
        [
            {
                "signal": "spread_edge_bin",
                "recommended": False,
                "recommendation_tier": "review",
                "edge_bin": "2-3",
                "spread_cover_calibration_status": "blocked_spread_cover_overconfidence",
            }
        ]
    ).to_csv(policy_path, index=False)
    preds = pd.DataFrame(
        [
            {
                "home_team": "JAX",
                "away_team": "PHI",
                "home_win_prob": 0.51,
                "pred_home_margin": 1.0,
                "odds_draftkings_home_moneyline": -105,
                "odds_draftkings_away_moneyline": -105,
                "odds_draftkings_home_spread": -3.5,
                "odds_draftkings_away_spread": 3.5,
            },
        ]
    )

    _attach_market_policy_fields(preds, policy_path=policy_path)

    assert bool(preds.loc[0, "market_spread_policy_recommendation_flag"]) is False
    assert preds.loc[0, "market_spread_policy_tier"] == "review"
    assert preds.loc[0, "market_spread_policy_status"] == "blocked_spread_cover_overconfidence"
    assert preds.loc[0, "market_policy_primary_signal"] == "blocked_spread"
