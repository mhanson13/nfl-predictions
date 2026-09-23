from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from tools.import_mtb_predictions import import_predictions, parse_args


def test_parse_args_defaults_repo_dir():
    args = parse_args([])

    assert args.script == "import:predictions"
    assert isinstance(args.repo_dir, Path)


def test_import_predictions_runs_npm_script(tmp_path, monkeypatch):
    repo = tmp_path / "mtb"
    repo.mkdir()
    (repo / "package.json").write_text('{"scripts":{"import:predictions":"echo ok"}}', encoding="utf-8")

    monkeypatch.setattr("tools.import_mtb_predictions.shutil.which", lambda _: "npm")
    run_mock = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr("tools.import_mtb_predictions.subprocess.run", run_mock)

    import_predictions(repo)

    run_mock.assert_called_once_with(["npm", "run", "import:predictions"], cwd=repo.resolve(), check=False)


def test_import_predictions_requires_package_json(tmp_path):
    repo = tmp_path / "mtb"
    repo.mkdir()

    with pytest.raises(FileNotFoundError, match="package.json"):
        import_predictions(repo)


def test_import_predictions_raises_on_failed_script(tmp_path, monkeypatch):
    repo = tmp_path / "mtb"
    repo.mkdir()
    (repo / "package.json").write_text('{"scripts":{"import:predictions":"exit 1"}}', encoding="utf-8")

    monkeypatch.setattr("tools.import_mtb_predictions.shutil.which", lambda _: "npm")
    monkeypatch.setattr("tools.import_mtb_predictions.subprocess.run", Mock(return_value=Mock(returncode=1)))

    with pytest.raises(RuntimeError, match="failed"):
        import_predictions(repo)
