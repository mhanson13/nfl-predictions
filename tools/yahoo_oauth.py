from __future__ import annotations

import argparse
import json
import secrets
import sys
import webbrowser
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from requests.auth import HTTPBasicAuth

from src.utils.secrets import get_secret


REQUEST_AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
FANTASY_GAME_URL = "https://fantasysports.yahooapis.com/fantasy/v2/game/nfl"
TOKEN_KEYS = {"access_token", "refresh_token"}
ENV_TOKEN_KEYS = {
    "access_token": "YAHOO_ACCESS_TOKEN",
    "refresh_token": "YAHOO_REFRESH_TOKEN",
}


def _require_secret(name: str) -> str:
    value = get_secret(name)
    if not value:
        raise RuntimeError(f"Missing required secret: {name}")
    return value


def _redirect_uri(value: str | None) -> str:
    redirect_uri = value or get_secret("YAHOO_REDIRECT_URI")
    if not redirect_uri:
        raise RuntimeError("Missing redirect URI. Set YAHOO_REDIRECT_URI or pass --redirect-uri.")
    return redirect_uri


def _refresh_token(value: str | None) -> str:
    refresh_token = value or get_secret("YAHOO_REFRESH_TOKEN") or get_secret("YAHOO_ACCESS_TOKEN_SECRET")
    if not refresh_token:
        raise RuntimeError("Missing refresh token. Set YAHOO_REFRESH_TOKEN or pass --refresh-token.")
    return refresh_token


def _safe_response_text(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:1000]
    return json.dumps(_redact_tokens(payload), indent=2)


def _redact_tokens(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if key in TOKEN_KEYS else _redact_tokens(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_tokens(item) for item in value]
    return value


def _post_token(payload: Mapping[str, str], *, timeout: float) -> dict[str, Any]:
    client_id = _require_secret("YAHOO_CLIENT_ID")
    client_secret = _require_secret("YAHOO_CLIENT_SECRET")
    response = requests.post(
        TOKEN_URL,
        data=dict(payload),
        auth=HTTPBasicAuth(client_id, client_secret),
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"Yahoo token request failed ({response.status_code}): {_safe_response_text(response)}")
    return response.json()


def _token_updates(data: Mapping[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    for source_key, env_key in ENV_TOKEN_KEYS.items():
        value = data.get(source_key)
        if isinstance(value, str) and value:
            updates[env_key] = value
    return updates


def _write_env_updates(path: Path, updates: Mapping[str, str]) -> None:
    if not updates:
        return
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    next_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            next_lines.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key in updates:
            next_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            next_lines.append(line)
    missing = [key for key in updates if key not in seen]
    if missing and next_lines and next_lines[-1].strip():
        next_lines.append("")
    for key in missing:
        next_lines.append(f"{key}={updates[key]}")
    path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def _print_token_summary(data: Mapping[str, Any], *, wrote_secrets: bool) -> None:
    expires_in = data.get("expires_in")
    token_type = data.get("token_type")
    refresh_returned = bool(data.get("refresh_token"))
    print(f"Yahoo token response: token_type={token_type!r}, expires_in={expires_in!r}, refresh_token_returned={refresh_returned}")
    if wrote_secrets:
        print("Updated secrets.env with refreshed Yahoo token fields.")
    else:
        print("Token values were not printed. Re-run with --write-secrets to update secrets.env.")


def cmd_auth_url(args: argparse.Namespace) -> int:
    client_id = _require_secret("YAHOO_CLIENT_ID")
    redirect_uri = _redirect_uri(args.redirect_uri)
    state = args.state
    if not args.no_state and not state:
        state = secrets.token_urlsafe(18)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "language": args.language,
    }
    if state:
        params["state"] = state
    if args.prompt_consent:
        params["prompt"] = "consent"
    url = f"{REQUEST_AUTH_URL}?{urlencode(params)}"
    print(url)
    if state:
        print(f"state={state}")
    if args.open:
        webbrowser.open(url)
    return 0


def cmd_exchange_code(args: argparse.Namespace) -> int:
    redirect_uri = _redirect_uri(args.redirect_uri)
    code = args.code
    if args.redirected_url:
        parsed = parse_qs(urlparse(args.redirected_url).query)
        code = parsed.get("code", [None])[0]
        returned_state = parsed.get("state", [None])[0]
        if args.expected_state and returned_state != args.expected_state:
            raise RuntimeError("Returned OAuth state did not match --expected-state.")
    if not code:
        raise RuntimeError("Missing authorization code. Pass --code or --redirected-url.")
    data = _post_token(
        {
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code": code,
        },
        timeout=args.timeout,
    )
    wrote = False
    if args.write_secrets:
        _write_env_updates(args.secrets_file, _token_updates(data))
        wrote = True
    _print_token_summary(data, wrote_secrets=wrote)
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    redirect_uri = _redirect_uri(args.redirect_uri)
    data = _post_token(
        {
            "grant_type": "refresh_token",
            "redirect_uri": redirect_uri,
            "refresh_token": _refresh_token(args.refresh_token),
        },
        timeout=args.timeout,
    )
    wrote = False
    if args.write_secrets:
        _write_env_updates(args.secrets_file, _token_updates(data))
        wrote = True
    _print_token_summary(data, wrote_secrets=wrote)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    redirect_uri = _redirect_uri(args.redirect_uri)
    token_data = _post_token(
        {
            "grant_type": "refresh_token",
            "redirect_uri": redirect_uri,
            "refresh_token": _refresh_token(args.refresh_token),
        },
        timeout=args.timeout,
    )
    access_token = token_data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Yahoo refresh response did not include an access token.")
    if args.write_secrets:
        _write_env_updates(args.secrets_file, _token_updates(token_data))
    response = requests.get(
        FANTASY_GAME_URL,
        params={"format": "json"},
        headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"},
        timeout=args.timeout,
    )
    if response.ok:
        payload = response.json()
        game_key = None
        try:
            game_key = payload["fantasy_content"]["game"][0]["game_key"]
        except (KeyError, IndexError, TypeError):
            pass
        print(f"Yahoo Fantasy game/nfl check succeeded. game_key={game_key!r}")
        return 0
    print(f"Yahoo Fantasy game/nfl check failed ({response.status_code}): {_safe_response_text(response)}")
    if response.status_code == 403:
        print("The token refreshed, but the app/user is not authorized for the Fantasy Sports API.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yahoo OAuth helper for the Fantasy Sports API.")
    parser.add_argument("--redirect-uri", help="Registered Yahoo redirect URI. Defaults to YAHOO_REDIRECT_URI.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--secrets-file", type=Path, default=Path("secrets.env"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_url = subparsers.add_parser("auth-url", help="Print the Yahoo user-consent URL.")
    auth_url.add_argument("--language", default="en-us")
    auth_url.add_argument("--state")
    auth_url.add_argument("--no-state", action="store_true")
    auth_url.add_argument("--prompt-consent", action="store_true", help="Ask Yahoo to show a fresh consent prompt.")
    auth_url.add_argument("--open", action="store_true", help="Open the authorization URL in a browser.")
    auth_url.set_defaults(func=cmd_auth_url)

    exchange_code = subparsers.add_parser("exchange-code", help="Exchange an authorization code for tokens.")
    exchange_code.add_argument("--code")
    exchange_code.add_argument("--redirected-url", help="Full redirected URL containing ?code=...")
    exchange_code.add_argument("--expected-state")
    exchange_code.add_argument("--write-secrets", action="store_true")
    exchange_code.set_defaults(func=cmd_exchange_code)

    refresh = subparsers.add_parser("refresh", help="Refresh Yahoo OAuth tokens.")
    refresh.add_argument("--refresh-token")
    refresh.add_argument("--write-secrets", action="store_true")
    refresh.set_defaults(func=cmd_refresh)

    check = subparsers.add_parser("check", help="Refresh token and call fantasy/v2/game/nfl.")
    check.add_argument("--refresh-token")
    check.add_argument("--write-secrets", action="store_true")
    check.set_defaults(func=cmd_check)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
