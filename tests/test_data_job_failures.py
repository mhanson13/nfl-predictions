"""A data job or the run-log wrapper dying must leave a clear error in the debug log."""
from __future__ import annotations

import asyncio
import sys
import time

import pytest

from src.utils import run_log
from tools import run_pipeline as rp


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(rp, "DATA_JOB_RETRY_DELAY", 0.0)


def _job(name: str, code: str) -> rp.Job:
    return rp.Job(name, [sys.executable, "-c", code])


def _run_async(jobs):
    asyncio.run(rp._async_run_jobs_parallel(jobs, max_workers=2, launch_delay=0.0, env=None))


def test_describe_exit_decodes_windows_crash_codes():
    assert rp.describe_exit(3221226505) == "exit code 3221226505 (0xC0000409: stack buffer overrun / fail-fast)"
    assert rp.describe_exit(-1073741819) == "exit code -1073741819 (0xC0000005: access violation)"
    assert rp.describe_exit(2) == "exit code 2"
    assert rp.describe_exit(-9) == "killed by signal 9"


def test_failed_data_job_is_logged_and_retried(tmp_path, capsys):
    marker = tmp_path / "first_attempt_done"
    flaky = _job("flaky", f"import pathlib,sys; p=pathlib.Path(r'{marker}'); "
                          "sys.exit(0) if p.exists() else (p.touch(), sys.exit(3))")
    _run_async([flaky, _job("steady", "pass")])
    out = capsys.readouterr().out
    assert "[pipeline] ERROR: data job flaky died: exit code 3 (attempt 1/2)" in out
    assert "[pipeline] Retrying data job flaky" in out
    assert "[pipeline] Starting flaky (async) (retry 1/1)" in out
    assert "[pipeline] Completed flaky (async)" in out


def test_every_failed_required_job_is_reported(capsys):
    with pytest.raises(RuntimeError) as err:
        _run_async([_job("espn_players", "import sys; sys.exit(5)"),
                    _job("espn_player_news", "import sys; sys.exit(7)"),
                    _job("ok_job", "pass")])
    out = capsys.readouterr().out
    assert "[pipeline] ERROR: 2 required data job(s) failed; stopping the run" in out
    assert "data job espn_players failed: exit code 5 after 2 attempt(s)" in str(err.value)
    assert "data job espn_player_news failed: exit code 7 after 2 attempt(s)" in str(err.value)


def test_single_failure_raises_data_job_error(capsys):
    with pytest.raises(rp.DataJobError) as err:
        _run_async([_job("nflverse", "import sys; sys.exit(9)"), _job("ok_job", "pass")])
    assert err.value.job_name == "nflverse" and err.value.returncode == 9 and err.value.attempts == 2


def test_optional_job_failure_warns_without_retry(capsys):
    _run_async([_job("noaa_weather", "import sys; sys.exit(4)"), _job("ok_job", "pass")])
    out = capsys.readouterr().out
    assert "optional async job noaa_weather failed (exit code 4); continuing." in out
    assert "Retrying data job noaa_weather" not in out


def test_threaded_runner_uses_the_same_retry_policy(tmp_path, capsys):
    marker = tmp_path / "done"
    flaky = _job("flaky", f"import pathlib,sys; p=pathlib.Path(r'{marker}'); "
                          "sys.exit(0) if p.exists() else (p.touch(), sys.exit(3))")
    rp.run_jobs_parallel([flaky, _job("steady", "pass")], max_workers=2, launch_delay=0.0,
                         env=None, dry_run=False, use_async=False)
    out = capsys.readouterr().out
    assert "[pipeline] ERROR: data job flaky died: exit code 3 (attempt 1/2)" in out
    with pytest.raises(RuntimeError, match="data job broken failed: exit code 6 after 2 attempt"):
        rp.run_jobs_parallel([_job("broken", "import sys; sys.exit(6)"), _job("steady", "pass")],
                             max_workers=2, launch_delay=0.0, env=None, dry_run=False, use_async=False)


def test_wrapper_failure_is_logged_and_stops_the_child(tmp_path, monkeypatch):
    """If the logger itself fails, the debug log says why and the pipeline is not left running."""
    original = run_log.write

    def flaky_write(text, streams):
        if text.startswith("boom"):
            raise OSError("console gone")
        return original(text, streams)

    monkeypatch.setattr(run_log, "write", flaky_write)
    log = tmp_path / "debug.txt"
    started = time.time()
    rc = run_log.run_logged([sys.executable, "-u", "-c", "print('boom'); import time; time.sleep(60)"],
                            "hdr\n", [log])
    assert time.time() - started < 30, "child pipeline was left running"
    text = log.read_text(encoding="utf-8")
    assert rc != 0
    assert "[run_log] ERROR: run_log wrapper failed (OSError: console gone); pipeline stopped" in text
    assert "Traceback (most recent call last)" in text
    assert "exit code   :" in text and "FAILED" in text


def test_stage_markers_after_a_progress_bar_carriage_return(tmp_path):
    code = ("import sys; sys.stdout.write('Category=rushing: 100%|\\u2588\\u2588|\\r[pipeline] Starting espn_player_news (async): x\\n'); "
            "print('[pipeline] Completed espn_player_news (async)')")
    log = tmp_path / "debug.txt"
    assert run_log.run_logged([sys.executable, "-u", "-c", code], "hdr\n", [log]) == 0
    text = log.read_text(encoding="utf-8")
    footer_lines = [ln for ln in text.splitlines() if ln.strip().endswith("espn_player_news")]
    assert footer_lines and "open" not in footer_lines[-1]
