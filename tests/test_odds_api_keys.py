from __future__ import annotations

import pandas as pd

from src.data import odds_api_historical
from src.utils import odds


def test_odds_api_key_helpers_keep_free_and_paid_separate(monkeypatch):
    secrets = {
        odds.ODDS_API_FREE_KEY_SECRET: " free-key ",
        odds.ODDS_API_PAID_KEY_SECRET: " paid-key ",
    }
    monkeypatch.setattr(odds, "get_secret", lambda name: secrets.get(name))

    assert odds.get_odds_api_free_key() == "free-key"
    assert odds.get_odds_api_paid_key() == "paid-key"


def test_odds_api_key_helpers_support_uppercase_aliases(monkeypatch):
    secrets = {
        "ODDS_API_FREE_KEY": " free-key ",
        "ODDS_API_PAID_KEY": " paid-key ",
    }
    monkeypatch.setattr(odds, "get_secret", lambda name: secrets.get(name))

    assert odds.get_odds_api_free_key() == "free-key"
    assert odds.get_odds_api_paid_key() == "paid-key"


def test_odds_api_free_key_does_not_fall_back_to_paid(monkeypatch):
    monkeypatch.setattr(odds, "get_secret", lambda name: "paid-key" if name in odds.ODDS_API_PAID_KEY_ALIASES else None)

    assert odds.get_odds_api_free_key() is None


def test_odds_api_paid_key_does_not_fall_back_to_free(monkeypatch):
    monkeypatch.setattr(odds, "get_secret", lambda name: "free-key" if name in odds.ODDS_API_FREE_KEY_ALIASES else None)

    assert odds.get_odds_api_paid_key() is None


def test_historical_cli_uses_paid_key_only(monkeypatch, tmp_path):
    captured: dict[str, str] = {}

    def fake_fetch_historical_events(**kwargs):
        captured["api_key"] = kwargs["api_key"]
        return pd.DataFrame()

    monkeypatch.setattr(odds_api_historical, "get_odds_api_paid_key", lambda: "paid-key")
    monkeypatch.setattr(odds_api_historical, "fetch_historical_events", fake_fetch_historical_events)

    odds_api_historical.main(
        [
            "--mode",
            "events",
            "--date",
            "2024-09-08T16:55:00Z",
            "--output",
            str(tmp_path / "events.csv"),
        ]
    )

    assert captured["api_key"] == "paid-key"
