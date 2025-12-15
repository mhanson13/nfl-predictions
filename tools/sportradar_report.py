# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.build_features import load_sportradar_team_features
from src.utils.io import PROC_DIR, read_df, write_df

logger = logging.getLogger("sportradar_report")

PROC_SPORTRADAR = PROC_DIR / "sportradar"
EVAL_DIR = Path("predictions") / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostics for Sportradar ingests.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_DIR,
        help="Directory to store summary outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log intended outputs without writing files.",
    )
    return parser.parse_args()


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_df(path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def summarize_change_log(proc_dir: Path) -> pd.DataFrame:
    df = _read_parquet_if_exists(proc_dir / "change_log.parquet")
    if df.empty:
        return pd.DataFrame(columns=["team_id", "type", "changes"])
    summary = (
        df.groupby(["team_id", "type"], dropna=False)
        .size()
        .reset_index(name="changes")
        .sort_values("changes", ascending=False)
    )
    return summary


def summarize_transactions(proc_dir: Path) -> pd.DataFrame:
    df = _read_parquet_if_exists(proc_dir / "transactions.parquet")
    if df.empty:
        return pd.DataFrame(columns=["team_id", "transaction_type", "count"])
    summary = (
        df.groupby(["team_id", "transaction_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return summary


def generate_team_feature_snapshot() -> pd.DataFrame:
    df = load_sportradar_team_features()
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df
    # Ensure consistent column order
    cols = ["season", "abbr"]
    metric_cols = [c for c in df.columns if c not in cols]
    return df[cols + metric_cols]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[sportradar_report] %(message)s",
    )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    change_summary = summarize_change_log(PROC_SPORTRADAR)
    txn_summary = summarize_transactions(PROC_SPORTRADAR)
    team_snapshot = generate_team_feature_snapshot()

    if args.dry_run:
        logger.info("[dry-run] change summary rows: %s", len(change_summary))
        logger.info("[dry-run] transaction summary rows: %s", len(txn_summary))
        logger.info("[dry-run] team feature snapshot rows: %s", len(team_snapshot))
        return

    if not change_summary.empty:
        write_df(change_summary, args.output_dir / "sportradar_change_summary.parquet")
        logger.info("Wrote change summary (%s rows)", len(change_summary))
    else:
        logger.info("No change log entries available.")

    if not txn_summary.empty:
        write_df(txn_summary, args.output_dir / "sportradar_transactions_summary.parquet")
        logger.info("Wrote transactions summary (%s rows)", len(txn_summary))
    else:
        logger.info("No transaction entries available.")

    if not team_snapshot.empty:
        write_df(team_snapshot, args.output_dir / "sportradar_team_features.parquet")
        logger.info("Wrote team feature snapshot (%s rows)", len(team_snapshot))
    else:
        logger.info("Team feature snapshot empty (no Sportradar stats aggregated).")


if __name__ == "__main__":
    main()

