from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.team_strength import ELO_MEAN, add_team_strength_features
from src.models.feature_columns import make_feature_diffs, select_feature_columns


def _games() -> pd.DataFrame:
    rows = []
    teams = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    for season in (2020, 2021):
        for week in range(1, 7):
            pairs = [("A", "B"), ("C", "D")] if week % 2 else [("A", "C"), ("B", "D")]
            for home, away in pairs:
                hs, as_ = (int(x) for x in rng.integers(0, 35, size=2))
                rows.append(
                    {
                        "season": season, "week": week, "game_id": f"{season}_{week}_{home}{away}",
                        "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
                        "pass_epa_per_db_home": (hs - as_) / 50, "pass_epa_per_db_away": (as_ - hs) / 50,
                    }
                )
    assert teams
    return pd.DataFrame(rows)


def test_features_never_see_same_week_or_later_results():
    games = _games()
    base = add_team_strength_features(games)
    changed = games.copy()
    later = changed.index[(changed.season == 2021) & (changed.week >= 3)]
    changed.loc[later, ["home_score", "away_score"]] = [[60, 0]] * len(later)
    changed.loc[later, "pass_epa_per_db_home"] = 5.0
    after = add_team_strength_features(changed)

    cols = [c for c in base.columns if c.startswith("strength_")]
    rows_through_week3 = base.index[(base.season < 2021) | (base.week <= 3)]
    pd.testing.assert_frame_equal(base.loc[rows_through_week3, cols], after.loc[rows_through_week3, cols])


def test_first_game_starts_at_mean_and_upcoming_rows_get_ratings():
    games = _games()
    games.loc[games.index[-2:], ["home_score", "away_score"]] = np.nan
    out = add_team_strength_features(games)
    first = out.iloc[0]
    assert first["strength_elo_home"] == ELO_MEAN and first["strength_season_games_home"] == 0
    assert out.iloc[-1][["strength_elo_home", "strength_pdiff_ewm_home"]].notna().all()
    # winners gain Elo
    wk1 = out[(out.season == 2020) & (out.week == 2)]
    assert (wk1["strength_elo_home"] != ELO_MEAN).all()


def test_offseason_regresses_elo_toward_mean():
    games = pd.DataFrame(
        {
            "season": [2020, 2020, 2021],
            "week": [1, 2, 1],
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
            "home_score": [30, 0, np.nan],
            "away_score": [0, 30, np.nan],
        }
    )
    out = add_team_strength_features(games)
    # pre-week-2 rating of A reflects only the week-1 win
    after_week1 = out.loc[1, "strength_elo_away"]
    assert after_week1 > ELO_MEAN
    assert out.loc[0, "strength_elo_home"] == ELO_MEAN
    final_2020 = 2 * after_week1 - ELO_MEAN  # symmetric second win of equal size is not guaranteed; compute directly
    pre_2021 = out.loc[2, "strength_elo_home"]
    assert ELO_MEAN < pre_2021 < final_2020


def test_strength_diffs_are_selected_as_model_features():
    feats = make_feature_diffs(add_team_strength_features(_games()))
    feats = pd.concat([feats] * 5, ignore_index=True)  # selection needs >= 50 non-null rows
    selected = select_feature_columns(feats)
    assert {"strength_elo_diff", "strength_pdiff_ewm_diff", "strength_pass_epa_def_ewm_diff"}.issubset(selected)


def test_season_aggregate_sources_are_not_model_features():
    from src.models.feature_columns import _is_safe_model_diff

    assert not _is_safe_model_diff("pressures_allowed_per_db_rolling3_diff")
    assert not _is_safe_model_diff("pressures_generated_per_db_rolling3_diff")
    assert _is_safe_model_diff("pass_epa_per_db_rolling3_diff")
    assert _is_safe_model_diff("sack_rate_allowed_rolling3_diff")
