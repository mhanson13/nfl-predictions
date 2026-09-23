# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Publish CSV pipeline artifacts into the local MattyTheBookie repository."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_MTB_REPO_DIR = Path(os.environ.get("MTB_REPO_DIR", Path("..") / "mtb"))
DEFAULT_MTB_CSV_DIR = Path("data") / "prediction-csvs"


@dataclass(frozen=True)
class CsvArtifact:
    source: Path
    destination: Path
    required: bool = False


CURRENT_CSV_ARTIFACTS = [
    CsvArtifact(Path("predictions/predictions.csv"), Path("current/predictions.csv"), required=True),
    CsvArtifact(Path("predictions/predictions_full.csv"), Path("current/predictions_full.csv")),
    CsvArtifact(Path("predictions/predictions_players_qb.csv"), Path("current/predictions_players_qb.csv")),
    CsvArtifact(Path("predictions/predictions_players_offense.csv"), Path("current/predictions_players_offense.csv")),
    CsvArtifact(Path("predictions/predictions_players_defense.csv"), Path("current/predictions_players_defense.csv")),
    CsvArtifact(Path("predictions/player_props_qb.csv"), Path("current/player_props_qb.csv")),
    CsvArtifact(Path("predictions/player_props_offense.csv"), Path("current/player_props_offense.csv")),
    CsvArtifact(Path("predictions/player_props_defense.csv"), Path("current/player_props_defense.csv")),
    CsvArtifact(Path("predictions/player_props_novelty.csv"), Path("current/player_props_novelty.csv")),
]


DIAGNOSTIC_CSV_ARTIFACTS = [
    CsvArtifact(Path("predictions/evaluation/overall_metrics.csv"), Path("evaluation/overall_metrics.csv")),
    CsvArtifact(Path("predictions/evaluation/weekly_metrics.csv"), Path("evaluation/weekly_metrics.csv")),
    CsvArtifact(
        Path("predictions/evaluation/calibration_by_prob_bin.csv"),
        Path("evaluation/calibration_by_prob_bin.csv"),
    ),
    CsvArtifact(Path("predictions/evaluation/team_error_summary.csv"), Path("evaluation/team_error_summary.csv")),
    CsvArtifact(
        Path("predictions/evaluation/high_confidence_misclassifications.csv"),
        Path("evaluation/high_confidence_misclassifications.csv"),
    ),
    CsvArtifact(Path("predictions/evaluation/largest_margin_errors.csv"), Path("evaluation/largest_margin_errors.csv")),
    CsvArtifact(Path("predictions/evaluation/feature_quality.csv"), Path("evaluation/feature_quality.csv")),
    CsvArtifact(
        Path("predictions/evaluation/merged_predictions_actuals.csv"),
        Path("evaluation/merged_predictions_actuals.csv"),
    ),
    CsvArtifact(Path("analysis/market_roi_moneyline.csv"), Path("analysis/market_roi_moneyline.csv")),
    CsvArtifact(Path("analysis/market_roi_spread.csv"), Path("analysis/market_roi_spread.csv")),
    CsvArtifact(Path("analysis/market_benchmark.csv"), Path("analysis/market_benchmark.csv")),
    CsvArtifact(Path("analysis/market_benchmark_summary.csv"), Path("analysis/market_benchmark_summary.csv")),
    CsvArtifact(Path("analysis/market_disagreement_roi.csv"), Path("analysis/market_disagreement_roi.csv")),
    CsvArtifact(Path("analysis/market_roi_spread_edge_bins.csv"), Path("analysis/market_roi_spread_edge_bins.csv")),
    CsvArtifact(
        Path("analysis/volatility_classifier_importance.csv"),
        Path("analysis/volatility_classifier_importance.csv"),
    ),
    CsvArtifact(
        Path("analysis/volatility_classifier_dataset.csv"),
        Path("analysis/volatility_classifier_dataset.csv"),
    ),
]


@dataclass(frozen=True)
class PublishedCsv:
    source: Path
    destination: Path
    bytes_copied: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy selected CSV outputs into the local MattyTheBookie repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("."),
        help="Root of the nfl-predictions repository.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_MTB_REPO_DIR,
        help="Local MattyTheBookie repository directory.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_MTB_CSV_DIR,
        help="Destination directory inside the MattyTheBookie repo.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Optional season for archiving current prediction CSVs.",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Optional week for archiving current prediction CSVs.",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Publish only the current prediction CSVs, skipping evaluation/analysis CSVs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected CSV is missing. By default only predictions.csv is required.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args(argv)


def build_artifact_plan(
    *,
    include_diagnostics: bool = True,
    season: int | None = None,
    week: int | None = None,
) -> list[CsvArtifact]:
    artifacts = list(CURRENT_CSV_ARTIFACTS)
    if include_diagnostics:
        artifacts.extend(DIAGNOSTIC_CSV_ARTIFACTS)
        if week is not None:
            artifacts.append(
                CsvArtifact(
                    Path("analysis") / f"model_metrics_week_{week:02d}.csv",
                    Path("analysis") / f"model_metrics_week_{week:02d}.csv",
                )
            )
            artifacts.extend(
                [
                    CsvArtifact(
                        Path("predictions")
                        / "evaluation"
                        / "player_props"
                        / f"prop_quality_week_{week:02d}.csv",
                        Path("evaluation") / "player_props" / f"prop_quality_week_{week:02d}.csv",
                    ),
                    CsvArtifact(
                        Path("predictions")
                        / "evaluation"
                        / "player_props"
                        / f"prop_quality_summary_week_{week:02d}.csv",
                        Path("evaluation") / "player_props" / f"prop_quality_summary_week_{week:02d}.csv",
                    ),
                    CsvArtifact(
                        Path("predictions")
                        / "evaluation"
                        / "player_props"
                        / f"prop_line_coverage_week_{week:02d}.csv",
                        Path("evaluation") / "player_props" / f"prop_line_coverage_week_{week:02d}.csv",
                    ),
                ]
            )

    if season is not None and week is not None:
        week_root = Path("seasons") / str(season) / f"week_{week:02d}"
        artifacts.extend(
            CsvArtifact(artifact.source, week_root / artifact.destination.name, artifact.required)
            for artifact in CURRENT_CSV_ARTIFACTS
        )

    return artifacts


def _resolve_destination_root(repo_dir: Path, dest_dir: Path) -> Path:
    repo_root = repo_dir.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"MattyTheBookie repo not found: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"MattyTheBookie repo path is not a directory: {repo_root}")

    destination_root = dest_dir if dest_dir.is_absolute() else repo_root / dest_dir
    destination_root = destination_root.resolve()
    try:
        destination_root.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Destination must be inside the MattyTheBookie repo: {destination_root}") from exc

    return destination_root


def publish_csvs(
    *,
    source_root: Path,
    repo_dir: Path,
    dest_dir: Path = DEFAULT_MTB_CSV_DIR,
    include_diagnostics: bool = True,
    season: int | None = None,
    week: int | None = None,
    strict: bool = False,
    dry_run: bool = False,
) -> list[PublishedCsv]:
    source_root = source_root.resolve()
    destination_root = _resolve_destination_root(repo_dir, dest_dir)
    artifacts = build_artifact_plan(include_diagnostics=include_diagnostics, season=season, week=week)
    published: list[PublishedCsv] = []
    missing: list[Path] = []
    copy_plan: list[tuple[CsvArtifact, Path, Path]] = []

    for artifact in artifacts:
        source_path = source_root / artifact.source
        destination_path = destination_root / artifact.destination
        if not source_path.exists():
            if strict or artifact.required:
                missing.append(artifact.source)
            else:
                print(f"[publish_mtb] Skipping missing optional CSV: {artifact.source}")
            continue

        if not source_path.is_file():
            raise ValueError(f"CSV artifact path is not a file: {source_path}")

        copy_plan.append((artifact, source_path, destination_path))

    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required CSV artifact(s): {missing_list}")

    for artifact, source_path, destination_path in copy_plan:
        bytes_copied = source_path.stat().st_size
        if dry_run:
            print(f"[publish_mtb] Would copy {source_path} -> {destination_path}")
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            print(f"[publish_mtb] Copied {artifact.source} -> {destination_path}")

        published.append(PublishedCsv(source_path, destination_path, bytes_copied))

    return published


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    published = publish_csvs(
        source_root=args.source_root,
        repo_dir=args.repo_dir,
        dest_dir=args.dest_dir,
        include_diagnostics=not args.current_only,
        season=args.season,
        week=args.week,
        strict=args.strict,
        dry_run=args.dry_run,
    )
    action = "Planned" if args.dry_run else "Published"
    print(f"[publish_mtb] {action} {len(published)} CSV artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
