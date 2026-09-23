from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.predict import predict_history as predict_history_module
from src.predict.predict_history import _clear_history_files, _walk_forward_split_frames


def test_walk_forward_split_uses_only_prior_seasons_for_training():
    df = pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2024, 2025],
            "week": [1, 1, 1, 2, 1],
            "game_id": ["a", "b", "c", "d", "e"],
            "home_margin": [3.0, -4.0, 7.0, 1.0, 2.0],
        }
    )

    train_df, score_df = _walk_forward_split_frames(
        df,
        score_season=2024,
        train_start_year=2023,
        weeks=[2],
    )

    assert train_df["season"].tolist() == [2023]
    assert score_df["game_id"].tolist() == ["d"]
    assert not train_df["season"].eq(2024).any()


def test_clear_history_files_removes_requested_seasons_only(tmp_path):
    keep = tmp_path / "w01_predictions_history_2022.csv"
    remove = tmp_path / "w01_predictions_history_2024.csv"
    unrelated = tmp_path / "predictions.csv"
    keep.write_text("keep", encoding="utf-8")
    remove.write_text("remove", encoding="utf-8")
    unrelated.write_text("other", encoding="utf-8")

    removed = _clear_history_files(tmp_path, [2024], logger=type("Logger", (), {"debug": lambda *a, **k: None})())

    assert removed == 1
    assert keep.exists()
    assert not remove.exists()
    assert unrelated.exists()


def test_walk_forward_history_preserves_pre_cap_raw_probability(monkeypatch):
    df = pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "week": [1, 2, 1, 2],
            "game_id": ["train_a", "train_b", "score_a", "score_b"],
            "home_team": ["AAA", "BBB", "CCC", "DDD"],
            "away_team": ["EEE", "FFF", "GGG", "HHH"],
            "home_score": [24, 17, 21, 14],
            "away_score": [17, 24, 17, 21],
            "home_margin": [7.0, -7.0, 4.0, -7.0],
            "feat_diff": [0.1, -0.1, 0.2, -0.2],
        }
    )
    args = SimpleNamespace(
        seasons=[2024],
        weeks=None,
        train_start_year=2023,
        min_train_games=1,
        use_gpu=False,
        skip_logit=True,
        winprob_param_config=None,
        spread_param_config=None,
        disable_volatility=True,
        volatility_artifact="missing.pkl",
        volatility_threshold=None,
        volatility_strength=0.35,
        volatility_margin_strength=0.45,
    )
    logger = type("Logger", (), {"warning": lambda *a, **k: None, "info": lambda *a, **k: None})()

    monkeypatch.setattr(predict_history_module, "make_feature_diffs", lambda frame: frame)
    monkeypatch.setattr(predict_history_module, "select_feature_columns", lambda frame: ["feat_diff"])
    monkeypatch.setattr(predict_history_module, "_fit_walk_forward_win_model", lambda *a, **k: {})
    monkeypatch.setattr(predict_history_module, "_fit_walk_forward_spread_model", lambda *a, **k: {})
    monkeypatch.setattr(
        predict_history_module,
        "_predict_walk_forward_win",
        lambda *a, **k: np.array([0.01, 0.99], dtype=float),
    )
    monkeypatch.setattr(
        predict_history_module,
        "_predict_walk_forward_spread",
        lambda *a, **k: np.array([1.0, -1.0], dtype=float),
    )

    result = predict_history_module._build_walk_forward_history(df, args, logger)

    assert result["home_win_prob_raw"].tolist() == [0.01, 0.99]
    assert result["home_win_prob_model_raw"].tolist() == [0.01, 0.99]
    assert np.allclose(result["home_win_prob_capped"].to_numpy(), [0.15, 0.85])
    assert np.allclose(result["home_win_prob"].to_numpy(), [0.15, 0.85])
