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

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

try:
    import httpx
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency 'httpx'. Install with: pip install httpx"
    ) from exc

from src.utils.rate_limit import SlidingWindowLimiter, TokenBucket


logger = logging.getLogger("http")


@dataclass
class RetryConfig:
    retries: int = 4
    backoff_factor: float = 0.5
    max_backoff: float = 10.0
    jitter: float = 0.2


class SportradarClient:
    BASE_URL = "https://api.sportradar.us/nfl/official/trial/v7/en"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 15.0,
        max_requests_per_sec: float = 1.0,
        burst: int = 2,
        retry_config: RetryConfig | None = None,
        session: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.client = session or httpx.Client(base_url=self.BASE_URL, timeout=timeout)
        self.token_bucket = TokenBucket(rate=max_requests_per_sec, capacity=burst)
        self.window_limiter = SlidingWindowLimiter(max_calls=burst, window=1.0)
        self.retry_config = retry_config or RetryConfig()

    def close(self) -> None:
        self.client.close()

    def _sleep_backoff(self, attempt: int) -> None:
        backoff = min(
            self.retry_config.max_backoff,
            self.retry_config.backoff_factor * (2 ** attempt),
        )
        jitter = random.uniform(0, self.retry_config.jitter)
        time.sleep(backoff + jitter)

    def _prepare_params(
        self,
        params: Mapping[str, Any] | None,
    ) -> MutableMapping[str, Any]:
        merged: MutableMapping[str, Any] = {}
        if params:
            merged.update(params)
        merged["api_key"] = self.api_key
        return merged

    def request(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        prepared_params = self._prepare_params(params)
        for attempt in range(self.retry_config.retries + 1):
            self.token_bucket.consume()
            self.window_limiter.acquire()
            try:
                response = self.client.get(path, params=prepared_params)
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait_seconds = float(retry_after)
                        except ValueError:
                            wait_seconds = self.retry_config.backoff_factor
                        time.sleep(wait_seconds)
                    else:
                        self._sleep_backoff(attempt)
                    last_exc = httpx.HTTPStatusError(
                        "Too Many Requests",
                        request=response.request,
                        response=response,
                    )
                    continue
                if response.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    self._sleep_backoff(attempt)
                    continue
                response.raise_for_status()
                return response
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt == self.retry_config.retries:
                    break
                self._sleep_backoff(attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown error during Sportradar request")
