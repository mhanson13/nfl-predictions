"""Shared MTB publish lock, vendored from MattyTheBookie.

Source: mhanson13/mattythebookie PR #22, scripts/publish_lock.py at e7df1ea
(see that repo's docs/publish-lock.md). Copied verbatim below this note so the
dispatcher does not import from the MTB checkout; keep it in sync by hand if MTB
changes the lock format or rules. (Not lint-fixed, so a re-sync stays a plain copy.)
"""
# ruff: noqa

from __future__ import annotations

import json
import os
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

LOCK_FILE_NAME = ".publish.lock"
# EX_TEMPFAIL: the lock is held by another publish; try again later.
LOCK_HELD_EXIT_CODE = 75
# EX_NOPERM: a caller passed a lock token that doesn't match the lock file.
LOCK_TOKEN_MISMATCH_EXIT_CODE = 77
DEFAULT_STALE_MINUTES = 90
POLL_SECONDS = 5


class LockHeld(RuntimeError):
    def __init__(self, path: Path, holder: dict | None) -> None:
        self.path = path
        self.holder = holder or {}
        who = ", ".join(
            f"{key}={self.holder[key]}"
            for key in ("owner", "sport", "pid", "started_at")
            if key in self.holder
        )
        super().__init__(f"Publish lock {path} is held ({who or 'unreadable lock file'}).")


def _utc_now_iso(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: object) -> float | None:
    if not isinstance(value, str):
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_lock(path: Path) -> dict | None:
    """The holder's JSON, or None when the file is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    return data if isinstance(data, dict) else None


def is_stale(path: Path, holder: dict | None, now: float, default_minutes: float) -> bool:
    started = _parse_iso(holder.get("started_at")) if holder else None

    if started is None:
        try:
            started = path.stat().st_mtime
        except OSError:
            return True

    minutes = holder.get("stale_after_minutes") if holder else None

    if not isinstance(minutes, (int, float)) or minutes <= 0:
        minutes = default_minutes

    return now - started > minutes * 60


def _try_create(path: Path, record: dict) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.write("\n")

    return True


def _take_over_stale(path: Path, seen: dict | None) -> None:
    """Move a stale lock aside, unless someone replaced it since we read it."""
    aside = path.with_name(f"{path.name}.stale-{secrets.token_hex(4)}")

    try:
        os.rename(path, aside)
    except OSError:
        return

    moved = read_lock(aside)

    if moved is not None and seen is not None and moved.get("token") != seen.get("token"):
        # Another process took the lock between our read and the rename: put it back.
        if not path.exists():
            os.rename(aside, path)
            return

    try:
        os.remove(aside)
    except OSError:
        pass


def acquire(
    path: Path,
    *,
    owner: str,
    sport: str,
    stale_minutes: float = DEFAULT_STALE_MINUTES,
    wait_seconds: float = 0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict:
    """Take the lock or raise LockHeld after waiting up to `wait_seconds`."""
    deadline = clock() + max(wait_seconds, 0)
    announced = False

    while True:
        now = clock()
        record = {
            "owner": owner,
            "pid": os.getpid(),
            "sport": sport,
            "stale_after_minutes": stale_minutes,
            "started_at": _utc_now_iso(now),
            "token": secrets.token_hex(8),
        }

        if _try_create(path, record):
            return record

        holder = read_lock(path)

        if is_stale(path, holder, now, stale_minutes):
            log(f"Publish lock {path} is stale ({holder or 'unreadable'}); taking it over.")
            _take_over_stale(path, holder)
            continue

        if now >= deadline:
            raise LockHeld(path, holder)

        if not announced:
            log(f"{LockHeld(path, holder)} Waiting up to {int(deadline - now)}s.")
            announced = True

        sleep(min(POLL_SECONDS, max(deadline - now, 0.1)))


class LockTokenMismatch(RuntimeError):
    pass


def check_held_by(
    path: Path,
    token: str,
    *,
    default_stale_minutes: float = DEFAULT_STALE_MINUTES,
    clock: Callable[[], float] = time.time,
) -> dict:
    """For a caller that already holds the lock: confirm `token` is the live holder's."""
    holder = read_lock(path)

    if holder is None:
        raise LockTokenMismatch(f"Publish lock {path} is missing or unreadable; the caller does not hold it.")

    if holder.get("token") != token:
        raise LockTokenMismatch(
            f"Publish lock {path} is held by someone else "
            f"(owner={holder.get('owner')}, pid={holder.get('pid')}); the token doesn't match."
        )

    if is_stale(path, holder, clock(), default_stale_minutes):
        raise LockTokenMismatch(
            f"Publish lock {path} is stale (started {holder.get('started_at')}), so another publish "
            "could take it over. Take a fresh lock with a stale_after_minutes that covers the whole publish."
        )

    return holder


def release(path: Path, record: dict) -> None:
    """Remove the lock only if it is still ours."""
    holder = read_lock(path)

    if holder is not None and holder.get("token") == record.get("token"):
        try:
            os.remove(path)
        except OSError:
            pass


@contextmanager
def publish_lock(path: Path, **kwargs) -> Iterator[dict]:
    record = acquire(path, **kwargs)

    try:
        yield record
    finally:
        release(path, record)
