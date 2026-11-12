from __future__ import annotations

from types import SimpleNamespace

from tools import import_mtb_predictions


def test_import_predictions_sets_archive_overwrite_env(monkeypatch, tmp_path):
    repo_dir = tmp_path / "mtb"
    repo_dir.mkdir()
    (repo_dir / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(import_mtb_predictions.shutil, "which", lambda _name: "npm")
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, check, env):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(import_mtb_predictions.subprocess, "run", fake_run)

    import_mtb_predictions.import_predictions(repo_dir, overwrite_archive=True)

    assert captured["command"] == ["npm", "run", "import:predictions"]
    assert captured["cwd"] == repo_dir.resolve()
    assert captured["check"] is False
    assert captured["env"]["PREDICTIONS_OVERWRITE_ARCHIVE"] == "1"
