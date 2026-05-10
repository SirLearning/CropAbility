"""Unit tests for evo2 CLI commands."""

from __future__ import annotations

import argparse
import sys
from types import ModuleType

from cropability.cli.main import _COMMANDS, build_parser


def _install_fake_evo2_module(monkeypatch, *, check_ok: bool = True) -> None:
    module = ModuleType("cropability.models.evo2")

    def check_model_downloadable(repo_id: str, token: str | None = None):
        _ = (repo_id, token)
        return check_ok, "ok" if check_ok else "not ok"

    def download_model(repo_id: str, local_dir: str, token: str | None = None):
        _ = (repo_id, local_dir, token)
        return local_dir

    def run_evo2(
        prompt: str,
        repo_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        token: str | None = None,
        local_files_only: bool = False,
    ):
        _ = (
            prompt,
            repo_id,
            max_new_tokens,
            temperature,
            top_p,
            token,
            local_files_only,
        )
        return "fake-output"

    module.check_model_downloadable = check_model_downloadable
    module.download_model = download_model
    module.run_evo2 = run_evo2
    monkeypatch.setitem(sys.modules, "cropability.models.evo2", module)


def test_parser_has_evo2_subcommand():
    parser = build_parser()
    args = parser.parse_args(["evo2", "--check"])
    assert args.command == "evo2"
    assert args.check is True


def test_cmd_evo2_check_success(monkeypatch):
    _install_fake_evo2_module(monkeypatch, check_ok=True)
    args = argparse.Namespace(
        repo_id="arcinstitute/evo2_7b",
        token=None,
        check=True,
        download_dir=None,
        prompt=None,
        max_new_tokens=64,
        temperature=0.8,
        top_p=0.95,
        local_files_only=False,
    )
    rc = _COMMANDS["evo2"](args)
    assert rc == 0


def test_cmd_evo2_check_fail(monkeypatch):
    _install_fake_evo2_module(monkeypatch, check_ok=False)
    args = argparse.Namespace(
        repo_id="arcinstitute/evo2_7b",
        token=None,
        check=True,
        download_dir=None,
        prompt=None,
        max_new_tokens=64,
        temperature=0.8,
        top_p=0.95,
        local_files_only=False,
    )
    rc = _COMMANDS["evo2"](args)
    assert rc == 1
