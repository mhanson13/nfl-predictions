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
    _write_csv(source_root / "analysis" / "market_roi_spread.csv", "market,roi\nspread,0.1\n")
    _write_csv(source_root / "analysis" / "market_benchmark_summary.csv", "segment,n_games\nall,10\n")
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
    assert (destination_root / "analysis" / "market_roi_spread.csv").exists()
    assert (destination_root / "analysis" / "market_benchmark_summary.csv").exists()
    assert (destination_root / "analysis" / "model_metrics_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "prop_quality_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "prop_quality_summary_week_02.csv").exists()
    assert (destination_root / "evaluation" / "player_props" / "prop_line_coverage_week_02.csv").exists()
    assert (destination_root / "seasons" / "2026" / "week_02" / "predictions.csv").exists()
    assert (destination_root / "seasons" / "2026" / "week_02" / "player_props_qb.csv").exists()
    assert len(published) == 19


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
