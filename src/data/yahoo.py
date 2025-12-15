# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import requests
from requests.auth import HTTPBasicAuth

from src.utils.io import RAW_DIR
from src.utils.logging import configure as configure_logging
from src.utils.secrets import get_secret

logger = logging.getLogger("yahoo")


def ensure_credentials() -> tuple[str, str, str, str, str]:
    app_id = get_secret("YAHOO_APP_ID")
    client_id = get_secret("YAHOO_CLIENT_ID")
    client_secret = get_secret("YAHOO_CLIENT_SECRET")
    access_token = get_secret("YAHOO_ACCESS_TOKEN")
    refresh_token = get_secret("YAHOO_REFRESH_TOKEN") or get_secret("YAHOO_ACCESS_TOKEN_SECRET")
    missing = [
        key
        for key, value in [
            ("YAHOO_APP_ID", app_id),
            ("YAHOO_CLIENT_ID", client_id),
            ("YAHOO_CLIENT_SECRET", client_secret),
            ("YAHOO_ACCESS_TOKEN", access_token),
            ("YAHOO_REFRESH_TOKEN", refresh_token),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing Yahoo credentials: {', '.join(missing)}")
    return app_id, client_id, client_secret, access_token, refresh_token


class YahooClient:
    BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2"
    TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        refresh_token: str,
        *,
        timeout: float = 20.0,
        retries: int = 4,
        backoff: float = 0.5,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at: Optional[float] = None
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = requests.Session()

    def close(self) -> None:  # pragma: no cover - cleanup only
        try:
            self.session.close()
        except Exception:
            pass

    def _sleep_backoff(self, attempt: int) -> None:
        wait = min(10.0, self.backoff * (2**attempt))
        time.sleep(wait)

    def _maybe_refresh(self) -> None:
        if self.expires_at and time.time() < self.expires_at:
            return
        self._refresh_token()

    def _refresh_token(self) -> None:
        logger.debug("Refreshing Yahoo OAuth2 token.")
        auth = HTTPBasicAuth(self.client_id, self.client_secret)
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        response = requests.post(self.TOKEN_URL, data=payload, auth=auth, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = f" ({response.text})"
            except Exception:
                detail = ""
            raise RuntimeError(f"Failed to refresh Yahoo access token: {exc}{detail}") from exc
        data = response.json()
        self.access_token = data.get("access_token", self.access_token)
        self.refresh_token = data.get("refresh_token", self.refresh_token)
        expires_in = data.get("expires_in")
        if expires_in:
            self.expires_at = time.time() + float(expires_in) - 30
        else:
            self.expires_at = None

    def request(self, path: str, params: Optional[Mapping[str, object]] = None) -> requests.Response:
        params = dict(params or {})
        params.setdefault("format", "json")
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.access_token}"}
        last_exc: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                if attempt == 0:
                    self._maybe_refresh()
                response = self.session.get(
                    f"{self.BASE_URL}/{path}",
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                status = response.status_code
                if status == 429:
                    logger.warning(
                        "Rate limited (429) for %s; remaining=%s",
                        path,
                        response.headers.get("X-RateLimit-Remaining"),
                    )
                    last_exc = requests.HTTPError(f"429 Too Many Requests: {response.text}", response=response)
                    self._sleep_backoff(attempt)
                    continue
                if status >= 500:
                    last_exc = requests.HTTPError(f"Server error {status}: {response.text}", response=response)
                    self._sleep_backoff(attempt)
                    continue
                if status in (401, 403):
                    if attempt == self.retries:
                        raise requests.HTTPError(f"Unauthorized {status}: {response.text}", response=response)
                    self._refresh_token()
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self.retries:
                    break
                self._sleep_backoff(attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown Yahoo API error")


def build_output_path(base_dir: Path, feed: str, suffix: Optional[str] = None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target_dir = base_dir / feed
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{feed}_{suffix}_{timestamp}.json" if suffix else f"{feed}_{timestamp}.json"
    return target_dir / filename


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)


def fetch_game_metadata(client: YahooClient, save_dir: Path) -> Optional[str]:
    try:
        response = client.request("game/nfl")
        data = response.json()
        write_json(build_output_path(save_dir, "game"), data)
        try:
            game_key = data["fantasy_content"]["game"][0]["game_key"]
            logger.debug("Yahoo game_key resolved to %s", game_key)
            return str(game_key)
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Unable to parse Yahoo game key: %s", exc)
            return None
    except requests.HTTPError as exc:
        detail = getattr(exc.response, "text", "")
        logger.warning("Failed to fetch game metadata: %s %s", exc, detail)
    except Exception as exc:
        logger.warning("Failed to fetch game metadata: %s", exc)
    return None


def fetch_teams(client: YahooClient, save_dir: Path, game_key: str) -> None:
    try:
        response = client.request(f"game/{game_key}/teams")
        write_json(build_output_path(save_dir, "teams"), response.json())
    except requests.HTTPError as exc:
        detail = getattr(exc.response, "text", "")
        logger.warning("Failed to fetch teams: %s %s", exc, detail)
    except Exception as exc:
        logger.warning("Failed to fetch teams: %s", exc)


def fetch_injuries(client: YahooClient, save_dir: Path, game_key: str, *, page_size: int = 25, max_pages: int = 40) -> None:
    start = 0
    for page in range(max_pages):
        path = f"game/{game_key}/players;status=INJ;start={start}"
        try:
            response = client.request(path)
            data = response.json()
            write_json(build_output_path(save_dir, "injuries", f"start{start}"), data)
            players = data.get("fantasy_content", {}).get("players", {})
            count = int(players.get("count", 0)) if isinstance(players, dict) else 0
            if count < page_size:
                break
        except requests.HTTPError as exc:
            detail = getattr(exc.response, "text", "")
            logger.warning("Failed to fetch injuries page start=%s: %s %s", start, exc, detail)
            break
        except Exception as exc:
            logger.warning("Failed to fetch injuries at start=%s (%s)", start, exc)
            break
        start += page_size


def fetch_scoreboard(
    client: YahooClient,
    save_dir: Path,
    game_key: str,
    *,
    week: Optional[int] = None,
) -> None:
    path = f"game/{game_key}/scoreboard"
    if week:
        path = f"{path};week={week}"
    try:
        response = client.request(path)
        suffix = f"week{week}" if week else None
        write_json(build_output_path(save_dir, "scoreboard", suffix), response.json())
    except requests.HTTPError as exc:
        detail = getattr(exc.response, "text", "")
        logger.warning("Failed to fetch scoreboard: %s %s", exc, detail)
    except Exception as exc:
        logger.warning("Failed to fetch scoreboard: %s", exc)


def fetch_players(
    client: YahooClient,
    save_dir: Path,
    game_key: str,
    *,
    page_size: int = 25,
    max_pages: int = 200,
) -> None:
    start = 0
    for page in range(max_pages):
        path = f"game/{game_key}/players;start={start}"
        try:
            response = client.request(path)
            data = response.json()
            write_json(build_output_path(save_dir, "players", f"start{start}"), data)
            players = data.get("fantasy_content", {}).get("players", {})
            count = int(players.get("count", 0)) if isinstance(players, dict) else 0
            if count < page_size:
                break
        except requests.HTTPError as exc:
            detail = getattr(exc.response, "text", "")
            logger.warning("Failed to fetch players page start=%s: %s %s", start, exc, detail)
            break
        except Exception as exc:
            logger.warning("Failed to fetch players at start=%s (%s)", start, exc)
            break
        start += page_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch static NFL data from Yahoo Sports API.")
    parser.add_argument(
        "--feeds",
        nargs="+",
        default=["game", "players"],
        choices=["game", "teams", "players", "injuries", "scoreboard"],
        help="Yahoo feeds to fetch.",
    )
    parser.add_argument("--season", type=int, help="Optional season filter for metadata/scoreboard context.")
    parser.add_argument("--week", type=int, help="Optional week for the scoreboard feed.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=RAW_DIR / "yahoo",
        help="Directory to store raw Yahoo responses.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log requests without executing them.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    _, client_id, client_secret, access_token, refresh_token = ensure_credentials()
    if args.dry_run:
        logger.info("Dry-run mode: Yahoo requests will be skipped.")
        return

    client = YahooClient(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    feeds = list(dict.fromkeys(args.feeds))
    try:
        game_key: Optional[str] = fetch_game_metadata(client, args.save_dir)
        for feed in feeds:
            if feed == "game":
                continue  # already handled above
            if feed in {"teams", "players", "injuries", "scoreboard"} and not game_key:
                logger.warning("Skipping Yahoo feed '%s' because game key is unavailable.", feed)
                continue
            if feed == "teams":
                logger.warning("Yahoo API does not expose a game-level teams feed; skipping.")
                continue
            if feed == "injuries":
                logger.warning("Yahoo API does not expose a standalone injuries feed; derive from players output instead.")
                continue
            if feed == "scoreboard":
                logger.warning("Yahoo API does not expose a game-level scoreboard feed; skipping.")
                continue
            if feed == "players":
                fetch_players(client, args.save_dir, game_key)
    finally:
        client.close()


if __name__ == "__main__":
    main()
