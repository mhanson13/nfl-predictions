from __future__ import annotations

from pathlib import Path

import pytest

from tools.publish_mtb_csvs import publish_csvs


def _write_csv(path: Path, text: str = "col\nvalue\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_publish_csvs_copies_current_diagnostics_and_week_snapshot(tmp_path):
    source_root = tmp_path / "nfl-predictions"
    mtb_repo = tmp_path / "mtb"
    mtb_repo.mkdir()

    _write_csv(source_root / "predictions" / "predictions.csv", "game_id\nabc\n")
    _write_csv(source_root / "predictions" / "predictions_full.csv", "game_id,feature\nabc,1\n")
    _write_csv(source_root / "predictions" / "player_props_qb.csv", "player_id,projection\nqb,250\n")
    _write_csv(source_root / "predictions" / "player_props_offense.csv", "player_id,projection\nrb,75\n")
    _write_csv(source_root / "predictions" / "player_props_defense.csv", "player_id,projection\ndl,0.5\n")
    _write_csv(source_root / "predictions" / "player_props_novelty.csv", "player_id,projection\n")
    _write_csv(source_root / "predictions" / "evaluation" / "overall_metrics.csv", "metric,value\nauc,0.7\n")
    _write_csv(
        source_root / "predictions" / "evaluation" / "feature_family_ablation.csv",
        "target,feature_set,primary_value\nwin_prob,baseline_plus_bdl,0.7\n",
    )
    _write_csv(source_root / "analysis" / "market_roi_spread.csv", "market,roi\nspread,0.1\n")
    _write_csv(source_root / "analysis" / "market_benchmark_summary.csv", "segment,n_games\nall,10\n")
    _write_csv(source_root / "analysis" / "market_clv_detail.csv", "game_id,moneyline_clv_prob_delta\nabc,0.01\n")
    _write_csv(source_root / "analysis" / "market_clv_summary.csv", "scope,segment,n_games\noverall,all,1\n")
    _write_csv(source_root / "analysis" / "market_disagreement_edge_bins.csv", "edge_bin,roi\n2-5%,0.1\n")
    _write_csv(
        source_root / "analysis" / "market_disagreement_segments.csv",
        "segment_family,segment,n_bets,roi\nmodel_pick_side,home,10,0.1\n",
    )
    _write_csv(
        source_root / "analysis" / "market_disagreement_season_stability.csv",
        "threshold,stability_pass\n0.02,true\n",
    )
    _write_csv(
        source_root / "analysis" / "market_probability_edge_calibration.csv",
        "bin_family,bin_label,model_brier,vegas_brier\nall,all,0.22,0.21\n",
    )
    _write_csv(
        source_root / "analysis" / "market_model_confidence_calibration.csv",
        "agreement_segment,confidence_bin,brier,log_loss\nall,70%+,0.22,0.63\n",
    )
    _write_csv(source_root / "analysis" / "market_blend_backtest.csv", "game_id,blend_home_win_prob\nabc,0.55\n")
    _write_csv(source_root / "analysis" / "market_blend_summary.csv", "scope,segment,blend_brier\noverall,all,0.21\n")
    _write_csv(source_root / "analysis" / "market_blend_policy.csv", "signal,recommended\nmarket_blend_probability,false\n")
    _write_csv(source_root / "analysis" / "market_spread_key_numbers.csv", "key_number,edge_bucket,roi\n3,2-3,0.01\n")
    _write_csv(source_root / "analysis" / "market_spread_calibration.csv", "cover_prob_bin,calibration_error\n80%+,0.8\n")
    _write_csv(
        source_root / "analysis" / "market_spread_cover_recalibration.csv",
        "cover_prob_bin,calibration_status\n80%+,overconfident\n",
    )
    _write_csv(source_root / "analysis" / "market_utility_segments.csv", "segment,n_bets\nall,10\n")
    _write_csv(source_root / "analysis" / "market_edge_policy.csv", "signal,recommended\nspread_threshold,true\n")
    _write_csv(source_root / "analysis" / "model_metrics_week_02.csv", "metric_key,value\naccuracy,0.65\n")
    _write_csv(
        source_root / "predictions" / "evaluation" / "player_props" / "prop_quality_week_02.csv",
        "market,rows\nqb_passing_yards,1\n",
    )
    _write_csv(
        source_root / "predictions" / "evaluation" / "player_props" / "prop_quality_summary_week_02.csv",
        "scope,rows\noverall,1\n",
    )
    _write_csv(
        source_root / "predictions" / "evaluation" / "player_props" / "prop_line_coverage_week_02.csv",
        "scope,raw_rows\noverall,3\n",
    )
    _write_csv(
        source_root / "predictions" / "evaluation" / "player_props" / "prop_roster_line_audit_week_02.csv",
        "player_id,line_availability_status\nrb,unknown\n",
    )
    _write_csv(
        source_root / "predictions" / "evaluation" / "player_props" / "confidence_tier_metrics.csv",
        "evaluation_scope,scope,n_rows\nhistorical_lines,overall,100\n",
    )

    published = publish_csvs(
        source_root=source_root,
        repo_dir=mtb_repo,
        season=2026,
        week=2,
        strict=False,
    )

    destination_root = mtb_repo / "data" / "prediction-csvs"
    assert (destination_root / "current" / "predictions.csv").read_text(encoding="utf-8") == "game_id\nabc\n"
    assert (destination_root / "current" / "predictions_full.csv").exists()
    assert (destination_root / "current" / "player_props_qb.csv").exists()
    assert (destination_root / "current" / "player_props_offense.csv").exists()
    assert (destination_root / "current" / "player_props_defense.csv").exists()
    assert (destination_root / "current" / "player_props_novelty.csv").exists()
    assert (destination_root / "evaluation" / "overall_metrics.csv").exists()
    assert (destination_root / "evaluation" / "feature_family_ablation.csv").exists()
    assert (destination_root / "analysis" / "market_roi_spread.csv").exists()
    assert (destination_root / "analysis" / "market_benchmark_summary.csv").exists()
    assert (destination_root / "analysis" / "market_clv_detail.csv").exists()
    assert (destination_root / "analysis" / "market_clv_summary.csv").exists()
    assert (destination_root / "analysis" / "market_disagreement_edge_bins.csv").exists()
    assert (destination_root / "analysis" / "market_disagreement_segments.csv").exists()
    assert (destination_root / "analysis" / "market_disagreement_season_stability.csv").exists()
    assert (destination_root / "analysis" / "market_probability_edge_calibration.csv").exists()
    assert (destination_root / "analysis" / "market_model_confidence_calibration.csv").exists()
    assert (destination_root / "analysis" / "market_blend_backtest.csv").exists()
    assert (destination_root / "analysis" / "market_blend_summary.csv").exists()
    assert (destination_root / "analysis" / "market_blend_policy.csv").exists()
    assert (destination_root / "analysis" / "market_spread_key_numbers.csv").exists()
    assert (destination_root / "analysis" / "market_spread_calibration.csv").exists()
    assert (destination_root / "analysis" / "market_spread_cover_recalibration.csv").exists()
    assert (destination_root / "analysis" / "market_utility_segments.csv").exists()
    assert (destination_root / "analysis" / "market_edge_policy.csv").exists()
    assert (destination_root / "analysis" / "model_metrics_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "prop_quality_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "prop_quality_summary_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "prop_line_coverage_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "prop_roster_line_audit_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "confidence_tier_metrics.csv").exists()
    assert (destination_root / "seasons" / "2026" / "week_02" / "predictions.csv").exists()
    assert (destination_root / "seasons" / "2026" / "week_02" / "player_props_qb.csv").exists()
    assert len(published) == 37


def test_publish_csvs_requires_predictions_csv(tmp_path):
    source_root = tmp_path / "nfl-predictions"
    mtb_repo = tmp_path / "mtb"
    mtb_repo.mkdir()

    with pytest.raises(FileNotFoundError, match="predictions.csv"):
        publish_csvs(source_root=source_root, repo_dir=mtb_repo)


def test_publish_csvs_strict_requires_optional_selected_csvs(tmp_path):
    source_root = tmp_path / "nfl-predictions"
    mtb_repo = tmp_path / "mtb"
    mtb_repo.mkdir()
    _write_csv(source_root / "predictions" / "predictions.csv")

    with pytest.raises(FileNotFoundError, match="predictions_full.csv"):
        publish_csvs(
            source_root=source_root,
            repo_dir=mtb_repo,
            include_diagnostics=False,
            strict=True,
        )

    assert not (mtb_repo / "data" / "prediction-csvs" / "current" / "predictions.csv").exists()


def test_publish_csvs_rejects_destination_outside_repo(tmp_path):
    source_root = tmp_path / "nfl-predictions"
    mtb_repo = tmp_path / "mtb"
    outside_dir = tmp_path / "outside"
    mtb_repo.mkdir()
    _write_csv(source_root / "predictions" / "predictions.csv")

    with pytest.raises(ValueError, match="inside the MattyTheBookie repo"):
        publish_csvs(source_root=source_root, repo_dir=mtb_repo, dest_dir=outside_dir)
