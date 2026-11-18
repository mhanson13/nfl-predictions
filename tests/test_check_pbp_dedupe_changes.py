import pandas as pd

from tools.check_pbp_dedupe_changes import check


def _frames():
    base = pd.DataFrame(
        {
            "game_id": ["2025_01_A_B", "2026_01_A_B", "2026_02_A_B"],
            "season": [2025, 2026, 2026],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "plays_home": [60.0, 120.0, 130.0],
            "sack_yards_lost_home": [-10.0, -20.0, -30.0],
            "sr_games_sampled_home": [2.0, 2.0, 2.0],
        }
    ).set_index("game_id")
    expected = pd.DataFrame(
        {"column": ["plays_home", "sack_yards_lost_home", "sr_games_sampled_home"], "model_feature": [False] * 3}
    )
    return base, expected


def test_counts_falling_on_2026_rows_is_ok():
    before, expected = _frames()
    after = before.copy()
    after.loc[after.season == 2026, "plays_home"] = [60.0, 65.0]
    after.loc["2026_01_A_B", "sack_yards_lost_home"] = -10.0  # signed: rise is allowed
    assert check(before, after, expected) == []


def test_pre_2026_change_count_rise_and_unexpected_column_alert():
    before, expected = _frames()
    before["other_home"] = 1.0
    after = before.copy()
    after.loc["2025_01_A_B", "plays_home"] = 59.0
    after.loc["2026_02_A_B", "plays_home"] = 131.0
    after.loc["2026_01_A_B", "other_home"] = 2.0
    alerts = check(before, after, expected)
    assert any("pre-2026 rows changed: plays_home" in a for a in alerts)
    assert any("count/sum rose after dedupe: plays_home" in a for a in alerts)
    assert any("unexpected column changed: other_home" in a for a in alerts)


def test_rise_explained_by_new_games():
    before, expected = _frames()
    after = before.copy()
    after.loc["2026_02_A_B", ["plays_home", "sr_games_sampled_home"]] = [190.0, 3.0]
    assert check(before, after, expected) == []
