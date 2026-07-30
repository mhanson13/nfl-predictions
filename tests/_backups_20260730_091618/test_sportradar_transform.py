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
import unittest

import pandas as pd

from src.data import sportradar_transform as st


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SportradarTransformTests(unittest.TestCase):
    def test_transform_schedule_creates_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "raw"
            out_root = Path(tmpdir) / "proc"
            sample_schedule = {
                "id": "season-2025",
                "year": 2025,
                "type": {"code": "REG"},
                "weeks": [
                    {
                        "sequence": 1,
                        "title": "Week 1",
                        "games": [
                            {
                                "id": "game-1",
                                "status": "scheduled",
                                "scheduled": "2025-09-01T17:00:00Z",
                                "attendance": 60000,
                                "game_type": "regular",
                                "conference_game": False,
                                "duration": "03:12",
                                "summary": {
                                    "venue": {"id": "venue-1", "name": "Sample Stadium"},
                                    "home": {"id": "team-home", "name": "Home Team", "points": 0},
                                    "away": {"id": "team-away", "name": "Away Team", "points": 0},
                                },
                            }
                        ],
                    }
                ],
            }
            _write_json(raw_root / "season_schedule" / "2025" / "schedule.json", sample_schedule)

            st.transform_schedule(raw_root, out_root, dry_run=False)
            out_path = out_root / "schedule.parquet"
            self.assertTrue(out_path.exists())
            df = pd.read_parquet(out_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "game_id"], "game-1")
            self.assertEqual(df.loc[0, "home_id"], "team-home")

    def test_transform_rosters_creates_players(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "raw"
            out_root = Path(tmpdir) / "proc"
            roster_payload = {
                "id": "team-home",
                "name": "Home Team",
                "market": "Home",
                "alias": "HT",
                "players": [
                    {
                        "id": "player-1",
                        "name": "Player One",
                        "position": "QB",
                        "status": "ACT",
                        "jersey": "12",
                    }
                ],
            }
            _write_json(
                raw_root / "team_roster" / "team-home" / "roster_20251023T000000Z.json",
                roster_payload,
            )

            st.transform_rosters(raw_root, out_root, dry_run=False)
            out_path = out_root / "roster_players.parquet"
            self.assertTrue(out_path.exists())
            df = pd.read_parquet(out_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "player_id"], "player-1")
            self.assertEqual(df.loc[0, "team_id"], "team-home")

            summary_path = out_root / "roster_status_summary.parquet"
            self.assertTrue(summary_path.exists())
            summary_df = pd.read_parquet(summary_path)
            self.assertEqual(summary_df.loc[0, "season"], 2025)
            self.assertEqual(summary_df.loc[0, "roster_total_players"], 1)
            self.assertEqual(summary_df.loc[0, "roster_status_act_count"], 1)
            self.assertEqual(summary_df.loc[0, "roster_injured_count"], 0)

    def test_transform_game_stats_outputs_team_and_player_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "raw"
            out_root = Path(tmpdir) / "proc"
            stats_payload = {
                "id": "game-1",
                "statistics": {
                    "home": {
                        "id": "team-home",
                        "passing": {
                            "totals": {"yards": 250, "touchdowns": 2},
                            "players": [
                                {
                                    "id": "player-1",
                                    "name": "QB One",
                                    "position": "QB",
                                    "yards": 250,
                                    "touchdowns": 2,
                                }
                            ],
                        },
                    },
                    "away": {
                        "id": "team-away",
                        "rushing": {
                            "totals": {"yards": 110},
                            "players": [
                                {
                                    "id": "player-2",
                                    "name": "RB Two",
                                    "position": "RB",
                                    "yards": 110,
                                }
                            ],
                        },
                    },
                },
            }
            _write_json(raw_root / "game_statistics" / "game-1" / "stats.json", stats_payload)

            st.transform_game_stats(raw_root, out_root, dry_run=False)
            team_path = out_root / "game_team_stats.parquet"
            player_path = out_root / "game_player_stats.parquet"
            self.assertTrue(team_path.exists())
            self.assertTrue(player_path.exists())
            team_df = pd.read_parquet(team_path)
            player_df = pd.read_parquet(player_path)
            self.assertSetEqual(set(team_df["stat_category"]), {"passing", "rushing"})
            self.assertSetEqual(set(player_df["player_id"]), {"player-1", "player-2"})

    def test_transform_transactions_and_change_log_handle_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "raw"
            out_root = Path(tmpdir) / "proc"
            tx_payload = {
                "transactions": [
                    {
                        "id": "tx-1",
                        "transaction_type": "signing",
                        "effective_on": "2025-10-01",
                        "team": {"id": "team-home", "name": "Home Team"},
                        "player": {"id": "player-1", "name": "QB One"},
                    }
                ]
            }
            change_payload = {
                "changes": [
                    {
                        "id": "chg-1",
                        "type": "depth_chart",
                        "effective": "2025-10-01T08:15:00Z",
                        "team": {"id": "team-home", "name": "Home Team"},
                        "player": {"id": "player-1", "name": "QB One"},
                        "game": {"id": "game-1"},
                    }
                ]
            }
            _write_json(raw_root / "daily_transactions" / "tx.json", tx_payload)
            _write_json(raw_root / "daily_change_log" / "change.json", change_payload)

            st.transform_transactions(raw_root, out_root, dry_run=False)
            st.transform_change_log(raw_root, out_root, dry_run=False)

            tx_df = pd.read_parquet(out_root / "transactions.parquet")
            change_df = pd.read_parquet(out_root / "change_log.parquet")
            self.assertEqual(tx_df.shape[0], 1)
            self.assertEqual(tx_df.loc[0, "transaction_id"], "tx-1")
            self.assertEqual(change_df.shape[0], 1)
            self.assertEqual(change_df.loc[0, "change_id"], "chg-1")

    def test_transform_seasonal_stats_outputs_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "raw"
            out_root = Path(tmpdir) / "proc"
            payload = {
                "season": {"year": 2025},
                "teams": [
                    {
                        "id": "team-home",
                        "name": "Home Team",
                        "alias": "HT",
                        "statistics": {
                            "offense": {"points": 100, "yards": 450},
                            "defense": {"points_allowed": 80},
                        },
                    }
                ],
            }
            _write_json(
                raw_root / "seasonal_statistics" / "2025" / "seasonal_stats_20251023T000000Z.json",
                payload,
            )

            st.transform_seasonal_stats(raw_root, out_root, dry_run=False)
            out_path = out_root / "seasonal_team_stats.parquet"
            self.assertTrue(out_path.exists())
            df = pd.read_parquet(out_path)
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["season"], 2025)
            self.assertEqual(row["team_alias"], "HT")
            self.assertEqual(row["sr_season_offense_points"], 100.0)
            self.assertEqual(row["sr_season_defense_points_allowed"], 80.0)


if __name__ == "__main__":
    unittest.main()
