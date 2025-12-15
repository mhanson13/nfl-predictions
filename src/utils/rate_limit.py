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

import threading
import time
from collections import deque


class TokenBucket:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> None:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            needed = tokens - self.tokens
            wait_time = needed / self.rate if self.rate > 0 else float("inf")
        if wait_time > 0:
            time.sleep(wait_time)
        with self.lock:
            self.timestamp = time.monotonic()
            self.tokens = max(0.0, self.tokens - tokens)


class SlidingWindowLimiter:
    """Rate limiter that ensures at most N events per window seconds."""

    def __init__(self, max_calls: int, window: float) -> None:
        self.max_calls = max_calls
        self.window = window
        self.events = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            while self.events and now - self.events[0] > self.window:
                self.events.popleft()
            if len(self.events) < self.max_calls:
                self.events.append(now)
                return
            sleep_time = self.window - (now - self.events[0])
        if sleep_time > 0:
            time.sleep(sleep_time)
        with self.lock:
            now = time.monotonic()
            while self.events and now - self.events[0] > self.window:
                self.events.popleft()
            self.events.append(now)

