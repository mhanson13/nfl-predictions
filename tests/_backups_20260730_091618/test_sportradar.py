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
import shutil
import unittest
from pathlib import Path

from src.data.sportradar import (
    FeedRequest,
    build_output_path,
    build_requests,
    dedupe_preserve_order,
    extract_game_ids_from_schedule,
)


class SportradarHelpersTestCase(unittest.TestCase):
    def test_dedupe_preserve_order(self):
        values = ["a", "b", "a", "c"]
        self.assertEqual(dedupe_preserve_order(values), ["a", "b", "c"])

    def test_build_requests_static(self):
        requests = build_requests(["league_hierarchy"], [], [], [], [], [])
        self.assertEqual(len(requests), 1)
        self.assertIsInstance(requests[0], FeedRequest)
        self.assertTrue(requests[0].path.endswith("league/hierarchy.json"))

    def test_build_requests_change_feed(self):
        requests = build_requests(["daily_change_log"], [], [], [], [], [])
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0].path.endswith("daily_change_log.json"))

    def test_build_requests_requires_season(self):
        with self.assertRaises(ValueError):
            build_requests(["season_schedule"], [], [], [], [], [])

    def test_build_requests_weekly_requires_weeks(self):
        with self.assertRaises(ValueError):
            build_requests(["weekly_schedule"], [2024], [], [], [], [])

    def test_build_output_path(self):
        tmp_path = Path("build/test_sportradar")
        try:
            target = build_output_path(
                tmp_path,
                "season_schedule",
                {"season": "2025", "week": "01"},
            )
            self.assertEqual(target.parent.name, "01")
            self.assertEqual(target.parent.parent.name, "2025")
            self.assertEqual(target.parent.parent.parent.name, "season_schedule")
            self.assertFalse(target.exists())
        finally:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)

    def test_extract_game_ids(self):
        tmp_file = Path("build/test_sportradar_schedule.json")
        try:
            sample = {
                "weeks": [
                    {"sequence": 1, "games": [{"id": "game-1"}, {"id": "game-2"}]},
                    {"sequence": 2, "games": [{"id": "game-3"}]},
                ]
            }
            tmp_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file.write_text(json.dumps(sample), encoding="utf-8")
            ids = extract_game_ids_from_schedule(tmp_file, {1})
            self.assertEqual(ids, ["game-1", "game-2"])
        finally:
            if tmp_file.exists():
                tmp_file.unlink()


if __name__ == "__main__":
    unittest.main()
