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

import json
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd

from tools import sportradar_report as report
from src.utils.io import write_df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_df(df, path)


def test_report_functions_produce_expected_summaries():
    with tempfile.TemporaryDirectory() as tmpdir:
        proc_dir = Path(tmpdir) / "proc"
        sr_dir = proc_dir / "sportradar"
        sr_dir.mkdir(parents=True, exist_ok=True)

        # Create change log and transactions samples
        change_df = pd.DataFrame(
            [
                {"change_id": "c1", "team_id": "team-home", "type": "depth_chart"},
                {"change_id": "c2", "team_id": "team-home", "type": "depth_chart"},
                {"change_id": "c3", "team_id": "team-away", "type": "injury"},
            ]
        )
        txn_df = pd.DataFrame(
            [
                {"transaction_id": "t1", "team_id": "team-home", "transaction_type": "signing"},
                {"transaction_id": "t2", "team_id": "team-home", "transaction_type": "signing"},
                {"transaction_id": "t3", "team_id": "team-away", "transaction_type": "waived"},
            ]
        )
        _write_parquet(change_df, sr_dir / "change_log.parquet")
        _write_parquet(txn_df, sr_dir / "transactions.parquet")

        # Sample game stats + schedule for snapshot
        schedule = pd.DataFrame(
            [
                {
                    "game_id": "game-1",
                    "season_year": 2025,
                    "week": 1,
                    "home_id": "team-home",
                    "home_alias": "PHI",
                    "home_name": "Philadelphia Eagles",
                    "away_id": "team-away",
                    "away_alias": "DAL",
                    "away_name": "Dallas Cowboys",
                }
            ]
        )
        team_stats = pd.DataFrame(
            [
                {
                    "scope": "team",
                    "game_id": "game-1",
                    "team_id": "team-home",
                    "team_role": "home",
                    "stat_category": "passing",
                    "stats": {"yards": 250, "touchdowns": 2},
                },
                {
                    "scope": "team",
                    "game_id": "game-1",
                    "team_id": "team-away",
                    "team_role": "away",
                    "stat_category": "rushing",
                    "stats": {"yards": 110, "touchdowns": 1},
                },
            ]
        )
        _write_parquet(schedule, sr_dir / "schedule.parquet")
        _write_parquet(team_stats, sr_dir / "game_team_stats.parquet")

        with mock.patch("tools.sportradar_report.PROC_SPORTRADAR", sr_dir), mock.patch(
            "src.features.build_features.PROC_DIR", proc_dir
        ):
            change_summary = report.summarize_change_log(sr_dir)
            txn_summary = report.summarize_transactions(sr_dir)
            snapshot = report.generate_team_feature_snapshot()

        assert not change_summary.empty
        assert change_summary.loc[change_summary["team_id"] == "team-home", "changes"].iloc[0] == 2

        assert not txn_summary.empty
        assert txn_summary.loc[
            (txn_summary["team_id"] == "team-home")
            & (txn_summary["transaction_type"] == "signing")
        ]["count"].iloc[0] == 2

        assert not snapshot.empty
        assert set(snapshot["abbr"]) == {"PHI", "DAL"}
