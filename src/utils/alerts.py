"""Pipeline alerts: always written to a local outbox, emailed when configured.

Email is sent from ``ALERT_EMAIL_FROM`` (alerts@mattythebookie.com) to
``ALERT_EMAIL_TO`` (comma-separated) through the first configured provider:

1. Bird SMTP  -- ``BIRD_ACCESS_KEY`` (alias ``BIRD_API_KEY``; API key with the emails scope; it is the SMTP
   password). us1.smtp.bird.com:587, STARTTLS required, username ``bird``.
2. SendGrid   -- ``SENDGRID_API_KEY`` (restricted, Mail Send only), v3 HTTPS API.

Values come from ``secrets.env`` / the environment. If nothing is configured or
sending fails, the alert is still saved under ``logs/alerts/`` with the reason.
Sending never raises.
"""

from __future__ import annotations

import re
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Callable, Optional

from src.utils.secrets import get_secret

OUTBOX_DIR = Path("logs/alerts")
FROM_NAME = "MattyTheBookie Alerts"
BIRD_SMTP_HOST = "us1.smtp.bird.com"
BIRD_SMTP_PORT = 587
BIRD_SMTP_USER = "bird"
SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"


@dataclass
class AlertResult:
    outbox_path: Path
    emailed: bool
    detail: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "alert"


def build_message(sender: str, recipients: list[str], subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = f"{FROM_NAME} <{sender}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1])
    msg.set_content(body)
    return msg


def _send_bird_smtp(api_key: str, msg: EmailMessage, timeout: float = 30) -> str:
    """Send over SMTP with mandatory, verified STARTTLS (never plaintext auth)."""
    context = ssl.create_default_context()
    with smtplib.SMTP(BIRD_SMTP_HOST, BIRD_SMTP_PORT, timeout=timeout) as smtp:
        smtp.ehlo()
        smtp.starttls(context=context)
        smtp.ehlo()
        smtp.login(BIRD_SMTP_USER, api_key)
        refused = smtp.send_message(msg)
    if refused:
        raise RuntimeError(f"recipients refused: {sorted(refused)}")
    return "bird smtp accepted"


def build_sendgrid_payload(sender: str, recipients: list[str], subject: str, body: str) -> dict:
    return {
        "personalizations": [{"to": [{"email": r} for r in recipients]}],
        "from": {"email": sender, "name": FROM_NAME},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
        # Alerts carry local paths and links; never rewrite them or add pixels.
        "tracking_settings": {"click_tracking": {"enable": False}, "open_tracking": {"enable": False}},
    }


def _send_sendgrid(api_key: str, payload: dict) -> str:
    import requests

    resp = requests.post(
        SENDGRID_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"sendgrid HTTP {resp.status_code}: {resp.text[:300]}")
    return f"sendgrid HTTP {resp.status_code}"


def send_alert(
    subject: str,
    body: str,
    *,
    outbox_dir: Path = OUTBOX_DIR,
    now: Optional[datetime] = None,
    secret: Callable[[str], Optional[str]] = get_secret,
    bird_send: Callable[[str, EmailMessage], str] = _send_bird_smtp,
    sendgrid_send: Callable[[str, dict], str] = _send_sendgrid,
) -> AlertResult:
    now = now or datetime.now()
    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / f"{now:%Y%m%d-%H%M%S}_{_slug(subject)}.md"

    sender, to = secret("ALERT_EMAIL_FROM"), secret("ALERT_EMAIL_TO")
    recipients = [r.strip() for r in (to or "").split(",") if r.strip()]
    bird_key = secret("BIRD_ACCESS_KEY") or secret("BIRD_API_KEY")
    sg_key = secret("SENDGRID_API_KEY")
    emailed = False
    if not (sender and recipients):
        missing = [n for n, v in (("ALERT_EMAIL_FROM", sender), ("ALERT_EMAIL_TO", recipients)) if not v]
        detail = f"email not configured (missing {', '.join(missing)}); outbox only"
    elif not (bird_key or sg_key):
        detail = "email not configured (no BIRD_ACCESS_KEY or SENDGRID_API_KEY); outbox only"
    else:
        try:
            if bird_key:
                detail = bird_send(bird_key, build_message(sender, recipients, subject, body))
            else:
                detail = sendgrid_send(sg_key, build_sendgrid_payload(sender, recipients, subject, body))
            emailed = True
        except Exception as exc:  # network, auth, refusal -- alerts must never break a run
            provider = "bird smtp" if bird_key else "sendgrid"
            detail = f"{provider} error: {exc.__class__.__name__}: {str(exc)[:300]}"

    path.write_text(f"# {subject}\n\n_{now:%Y-%m-%d %H:%M:%S} | {detail}_\n\n{body}\n", encoding="utf-8")
    print(f"[alerts] {subject} -> {path} ({detail})")
    return AlertResult(path, emailed, detail)
