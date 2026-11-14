"""Pipeline alerts: always written to a local outbox, emailed when configured.

Email goes through SendGrid's v3 API from ``ALERT_EMAIL_FROM`` (e.g.
alerts@mattythebookie.com) to ``ALERT_EMAIL_TO`` (comma-separated). All three
values come from ``secrets.env`` / the environment:

    SENDGRID_API_KEY   restricted key with Mail Send only
    ALERT_EMAIL_FROM   sender on the authenticated alerts subdomain
    ALERT_EMAIL_TO     recipient(s)

If any are missing, or sending fails, the alert is still saved under
``logs/alerts/`` and the failure reason is recorded there. Sending never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.utils.secrets import get_secret

OUTBOX_DIR = Path("logs/alerts")
SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"
FROM_NAME = "MattyTheBookie Alerts"


@dataclass
class AlertResult:
    outbox_path: Path
    emailed: bool
    detail: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "alert"


def _post_sendgrid(api_key: str, payload: dict) -> tuple[int, str]:
    import requests

    resp = requests.post(
        SENDGRID_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    return resp.status_code, resp.text[:300]


def build_payload(sender: str, recipients: list[str], subject: str, body: str) -> dict:
    return {
        "personalizations": [{"to": [{"email": r} for r in recipients]}],
        "from": {"email": sender, "name": FROM_NAME},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
        # Alerts carry local paths and links; never rewrite them or add pixels.
        "tracking_settings": {"click_tracking": {"enable": False}, "open_tracking": {"enable": False}},
    }


def send_alert(
    subject: str,
    body: str,
    *,
    outbox_dir: Path = OUTBOX_DIR,
    now: Optional[datetime] = None,
    secret: Callable[[str], Optional[str]] = get_secret,
    post: Callable[[str, dict], tuple[int, str]] = _post_sendgrid,
) -> AlertResult:
    now = now or datetime.now()
    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / f"{now:%Y%m%d-%H%M%S}_{_slug(subject)}.md"

    api_key, sender, to = secret("SENDGRID_API_KEY"), secret("ALERT_EMAIL_FROM"), secret("ALERT_EMAIL_TO")
    recipients = [r.strip() for r in (to or "").split(",") if r.strip()]
    if not (api_key and sender and recipients):
        missing = [n for n, v in (("SENDGRID_API_KEY", api_key), ("ALERT_EMAIL_FROM", sender),
                                  ("ALERT_EMAIL_TO", recipients)) if not v]
        emailed, detail = False, f"email not configured (missing {', '.join(missing)}); outbox only"
    else:
        try:
            status, text = post(api_key, build_payload(sender, recipients, subject, body))
            emailed = status in (200, 202)
            detail = f"sendgrid HTTP {status}" + ("" if emailed else f": {text}")
        except Exception as exc:  # network etc. -- alerts must never break a run
            emailed, detail = False, f"sendgrid error: {exc.__class__.__name__}: {exc}"

    path.write_text(f"# {subject}\n\n_{now:%Y-%m-%d %H:%M:%S} | {detail}_\n\n{body}\n", encoding="utf-8")
    print(f"[alerts] {subject} -> {path} ({detail})")
    return AlertResult(path, emailed, detail)
