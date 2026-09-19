from __future__ import annotations

from tools.yahoo_oauth import _redact_tokens, _write_env_updates


def test_write_env_updates_replaces_and_appends(tmp_path):
    env_path = tmp_path / "secrets.env"
    env_path.write_text(
        "\n".join(
            [
                "YAHOO_ACCESS_TOKEN=old-access",
                "OTHER_KEY=value",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_env_updates(
        env_path,
        {
            "YAHOO_ACCESS_TOKEN": "new-access",
            "YAHOO_REFRESH_TOKEN": "new-refresh",
        },
    )

    assert env_path.read_text(encoding="utf-8").splitlines() == [
        "YAHOO_ACCESS_TOKEN=new-access",
        "OTHER_KEY=value",
        "",
        "YAHOO_REFRESH_TOKEN=new-refresh",
    ]


def test_redact_tokens_hides_oauth_values():
    payload = {
        "access_token": "secret-access",
        "nested": {"refresh_token": "secret-refresh", "description": "visible"},
    }

    assert _redact_tokens(payload) == {
        "access_token": "***REDACTED***",
        "nested": {"refresh_token": "***REDACTED***", "description": "visible"},
    }
