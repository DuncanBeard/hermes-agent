"""Verification policy is enforced at the helpers, not by cooperating callers."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import main, main_install_repair as ir, managed_uv, update_cmd


CUSTOM_INDEX = "https://example.invalid/simple"
WINDOWS_SHIMS = pytest.mark.skipif(sys.platform != "win32", reason="Windows console shims")
VERIFIERS = ["core", pytest.param("shim", marks=WINDOWS_SHIMS)]


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "test"))
    for name in ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "PIP_INDEX_URL", "VIRTUAL_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["missing-dep>=1"]\n'
        '[project.scripts]\nhermes = "hermes_cli.main:main"\n'
    )
    scripts = tmp_path / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    python = scripts / ("python.exe" if sys.platform == "win32" else "python")
    python.touch()
    shim = scripts / "hermes.exe"
    shim.touch()
    # Only process boundaries are mocked; policy reads, target resolution,
    # metadata reads, shim checks and the verifier call graph remain real.
    child = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    install = Mock()
    monkeypatch.setattr(ir.subprocess, "run", child)
    monkeypatch.setattr(ir, "_run_install_with_heartbeat", install)
    return SimpleNamespace(root=tmp_path, python=python, shim=shim, child=child, install=install)


def missing_target(target, verifier):
    if verifier == "core":
        target.child.return_value.stdout = "missing-dep\n"
        return ir._verify_core_dependencies_installed
    target.shim.unlink()
    return ir._verify_console_scripts_installed


def assert_read_only(target):
    target.install.assert_not_called()
    assert all(call.args[0][1] == "-c" for call in target.child.call_args_list)


@pytest.mark.parametrize("verifier", VERIFIERS)
@pytest.mark.parametrize("source", ["explicit", "inherited", "shared-policy", "shared-policy-explicit"])
@pytest.mark.parametrize("repair_kwargs", [{}, {"allow_repair": True}], ids=["default", "requested"])
def test_custom_feed_verification_cannot_enable_repair(target, monkeypatch, verifier, source, repair_kwargs):
    env = {"VIRTUAL_ENV": str(target.root / "venv")}
    if source == "explicit":
        env["UV_DEFAULT_INDEX"] = CUSTOM_INDEX
    elif source == "shared-policy-explicit":
        env["UV_DEFAULT_INDEX"] = "https://pypi.org/simple"
        (target.root / "package-sources.json").write_text(json.dumps({"python_index_url": CUSTOM_INDEX}))
    else:
        monkeypatch.setenv("VIRTUAL_ENV", env["VIRTUAL_ENV"])
        env = None
        if source == "inherited":
            monkeypatch.setenv("UV_DEFAULT_INDEX", CUSTOM_INDEX)
        else:
            (target.root / "package-sources.json").write_text(json.dumps({"python_index_url": CUSTOM_INDEX}))
    original = dict(env) if env is not None else dict(os.environ)
    verify = missing_target(target, verifier)

    with pytest.raises(RuntimeError, match="verification failed: missing .*repair disabled"):
        verify(["uv", "pip"], env=env, **repair_kwargs)

    assert_read_only(target)
    assert (env if env is not None else dict(os.environ)) == original
    if verifier == "core":
        target.child.assert_called_once()
        assert target.child.call_args.args[0][0] == str(target.python)
        assert target.child.call_args.kwargs["env"]["UV_DEFAULT_INDEX"] == CUSTOM_INDEX


@pytest.mark.parametrize("verifier", VERIFIERS)
@pytest.mark.parametrize("source", ["inherited", "shared-policy"])
def test_skipped_install_verification_remains_read_only(target, monkeypatch, capsys, verifier, source):
    if source == "inherited":
        monkeypatch.setenv("UV_DEFAULT_INDEX", CUSTOM_INDEX)
    else:
        (target.root / "package-sources.json").write_text(json.dumps({"python_index_url": CUSTOM_INDEX}))
    missing_target(target, verifier)
    probe_result = target.child.return_value
    target.child.side_effect = lambda cmd, **kw: (
        subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"] else probe_result
    )
    monkeypatch.setattr(main, "_abort_dependency_sync_if_self_locked", Mock())
    monkeypatch.setattr(managed_uv, "update_managed_uv", Mock())
    monkeypatch.setattr(managed_uv, "ensure_uv", Mock(return_value="uv"))
    # Stop before unrelated lazy refresh if the broken verifier returns success.
    clear_marker = Mock(side_effect=AssertionError("verification incorrectly reached marker cleanup"))
    monkeypatch.setattr(main, "_clear_update_incomplete_marker", clear_marker)

    with pytest.raises(RuntimeError, match="verification failed: missing .*repair disabled"):
        update_cmd._sync_python_dependencies_after_pull(
            ["git"], "main", "before-pull", active_lazy_features=[],
            active_tool_dependencies=[], _windows_gateway_resume=None,
        )

    assert "skipping reinstall" in capsys.readouterr().out
    target.install.assert_not_called()
    assert target.child.call_args_list[0].args[0][:2] == ["git", "diff"]
    assert all(call.args[0][1] == "-c" for call in target.child.call_args_list[1:])
    clear_marker.assert_not_called()
    assert (target.root / ".update-incomplete").is_file()


@pytest.mark.parametrize("verifier", VERIFIERS)
@pytest.mark.parametrize("source", ["default", "public"])
@pytest.mark.parametrize("explicit_env", [False, True], ids=["env-none", "explicit-env"])
@pytest.mark.parametrize("allow_repair", [False, True], ids=["read-only", "repair"])
def test_public_or_default_feed_preserves_repair_choice(
    target, monkeypatch, verifier, source, explicit_env, allow_repair,
):
    env = {"VIRTUAL_ENV": str(target.root / "venv")}
    if source == "public":
        env["UV_DEFAULT_INDEX"] = "https://pypi.org/simple/"
    if not explicit_env:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        env = None
    verify = missing_target(target, verifier)
    prefix = [str(target.python), "-m", "pip"]
    if not allow_repair:
        with pytest.raises(RuntimeError, match="repair disabled"):
            verify(prefix, env=env, allow_repair=False)
        assert_read_only(target)
    else:
        verify(prefix, env=env)
        commands = [call.args[0] for call in target.install.call_args_list]
        expected = [prefix + ["install", "--reinstall", "-e", "."]]
        if verifier == "core":
            expected.append(prefix + ["install", "--reinstall", "missing-dep>=1"])
        assert commands == expected
        assert all(call.kwargs["env"] is env for call in target.install.call_args_list)


@pytest.mark.parametrize("verifier", VERIFIERS)
def test_custom_feed_healthy_verification_is_read_only(target, verifier):
    verify = (ir._verify_core_dependencies_installed if verifier == "core"
              else ir._verify_console_scripts_installed)
    verify(["uv", "pip"], env={"VIRTUAL_ENV": str(target.root / "venv"),
                               "UV_DEFAULT_INDEX": CUSTOM_INDEX})
    assert_read_only(target)
