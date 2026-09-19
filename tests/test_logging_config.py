from __future__ import annotations

from src.utils.logging_config import redact_secrets


def test_redact_secrets_masks_query_credentials():
    url = "https://example.test/path?unitGroup=us&key=abc123&token=xyz789&safe=value"

    assert redact_secrets(url) == (
        "https://example.test/path?unitGroup=us&key=***REDACTED***"
        "&token=***REDACTED***&safe=value"
    )
