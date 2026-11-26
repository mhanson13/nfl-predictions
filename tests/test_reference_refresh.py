"""nfl_reference_refresh must not fail a live run when Wikipedia rate-limits us."""

import pandas as pd
import pytest
import requests

from src.data.reference import update_regular_season as urs


def _http_429():
    resp = requests.Response()
    resp.status_code = 429
    resp.url = urs.SEASON_HISTORY_URL
    return requests.HTTPError("429 Client Error: Too Many Requests", response=resp)


def _raise_429():
    raise _http_429()


def test_fetch_error_keeps_cached_csv(tmp_path, monkeypatch, capsys):
    regular = tmp_path / "regular.csv"
    history = tmp_path / "history.csv"
    pd.DataFrame({"a": [1]}).to_csv(regular, index=False)
    pd.DataFrame({"season": [2025]}).to_csv(history, index=False)
    before = (regular.read_bytes(), history.read_bytes())
    monkeypatch.setattr(urs, "REGULAR_SEASON_PATH", regular)
    monkeypatch.setattr(urs, "SEASON_HISTORY_PATH", history)
    monkeypatch.setattr(urs, "fetch_regular_season_games", _raise_429)
    monkeypatch.setattr(urs, "fetch_season_history", _raise_429)

    urs.main([])  # must not raise

    assert (regular.read_bytes(), history.read_bytes()) == before
    assert capsys.readouterr().out.count("keeping cached") == 2


def test_fetch_error_without_cache_still_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(urs, "REGULAR_SEASON_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(urs, "fetch_regular_season_games", _raise_429)
    with pytest.raises(requests.HTTPError):
        urs.main(["--skip-history"])


def test_skip_history_does_not_fetch_history(tmp_path, monkeypatch):
    regular = tmp_path / "regular.csv"
    monkeypatch.setattr(urs, "REGULAR_SEASON_PATH", regular)
    monkeypatch.setattr(urs, "fetch_regular_season_games", lambda: pd.DataFrame({"a": [1]}))

    def _fail():
        raise AssertionError("history must not be fetched with --skip-history")

    monkeypatch.setattr(urs, "fetch_season_history", _fail)
    urs.main(["--skip-history"])
    assert regular.exists()
