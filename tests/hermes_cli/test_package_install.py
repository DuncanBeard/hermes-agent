import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import os
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class LockedFeedTests(unittest.TestCase):
    def test_export_install_retains_hashes_and_isolates_build_dependencies(self):
        from hermes_cli import package_install as module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uv.lock").write_text(
                'version = 1\n[[package]]\nname="setuptools"\nversion="83.0.0"\nsource={registry="https://pypi.org/simple"}\nwheels=[{url="https://files.pythonhosted.org/a.whl",hash="sha256:'
                + "a" * 64
                + '"}]\n'
            )
            (root / "pyproject.toml").write_text(
                '[build-system]\nrequires=["setuptools==83.0.0"]\n'
            )
            commands = []

            def run(cmd, **kw):
                commands.append((cmd, kw))
                if "export" in cmd:
                    Path(cmd[cmd.index("--output-file") + 1]).write_text(
                        "example==1.0 --hash=sha256:" + "b" * 64 + "\n"
                    )

            with patch.object(module.subprocess, "run", side_effect=run):
                module.install_locked_from_feed(
                    "uv",
                    root,
                    root / "venv/python",
                    {"UV_DEFAULT_INDEX": "https://example.invalid/simple"},
                    group="all",
                )
            self.assertEqual(len(commands), 4)
            self.assertIn("--frozen", commands[0][0])
            self.assertIn("--no-emit-project", commands[0][0])
            for cmd, kw in commands[1:3]:
                self.assertIn("--require-hashes", cmd)
                self.assertEqual(
                    kw["env"]["UV_DEFAULT_INDEX"], "https://example.invalid/simple"
                )
            for cmd, kwargs in commands[1:]:
                self.assertEqual(
                    kwargs.get("stderr"),
                    module.subprocess.STDOUT,
                    "candidate installs must stream diagnostics on stdout",
                )
            self.assertIn("--no-build-isolation", commands[3][0])
            self.assertIn("--no-deps", commands[3][0])
            self.assertNotIn("--locked", commands[0][0])

    def test_unlocked_build_tool_gets_approved_feed_hash_snapshot(self):
        from hermes_cli import package_install as module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uv.lock").write_text("version=1\npackage=[]\n")
            (root / "pyproject.toml").write_text('[build-system]\nrequires=["wheel"]\n')
            commands = []

            def run(cmd, **kw):
                commands.append(cmd)
                if "--output-file" in cmd:
                    Path(cmd[cmd.index("--output-file") + 1]).write_text(
                        "wheel==0.1 --hash=sha256:" + "b" * 64 + "\n"
                    )

            with patch.object(module.subprocess, "run", side_effect=run):
                module.install_locked_from_feed(
                    "uv",
                    root,
                    root / "python",
                    {"UV_DEFAULT_INDEX": "https://example.invalid/simple"},
                )
            compile_cmd = next(c for c in commands if "compile" in c)
            self.assertIn("--generate-hashes", compile_cmd)
            self.assertIn("--constraint", compile_cmd)

    def test_real_offline_export_preserves_registry_pins_and_hashes(self):
        import os
        import shutil
        import subprocess
        import tomllib
        from hermes_cli import package_install as module

        uv = shutil.which("uv")
        if uv is None:
            self.skipTest("uv is required for the real offline export probe")
        real_run = subprocess.run
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("uv.lock", "pyproject.toml"):
                shutil.copyfile(repo / name, root / name)
            original_lock = (root / "uv.lock").read_bytes()
            lock = tomllib.loads(original_lock.decode())
            known_hashes = {
                a["hash"]
                for p in lock["package"]
                for a in [
                    *p.get("wheels", []),
                    *([p["sdist"]] if p.get("sdist") else []),
                ]
            }
            exported = []

            class ExportObserved(Exception):
                pass

            def run(cmd, **kwargs):
                if "export" in cmd:
                    result = real_run(cmd, **kwargs, timeout=30)
                    text = Path(cmd[cmd.index("--output-file") + 1]).read_text(
                        encoding="utf-8"
                    )
                    exported.append(text)
                    return result
                # Deliberately stop before any resolver/install operation: this
                # probe exercises real uv export, not a live package install.
                raise ExportObserved

            env = {
                k: v
                for k, v in os.environ.items()
                if k.upper()
                in {"SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "PATH"}
            }
            env.update(
                HERMES_HOME=str(root / "home"),
                UV_OFFLINE="1",
                UV_PYTHON_DOWNLOADS="never",
                UV_DEFAULT_INDEX="https://example.invalid/simple",
                UV_CACHE_DIR=str(root / "cache"),
            )
            with patch.object(module.subprocess, "run", side_effect=run):
                with self.assertRaises(ExportObserved):
                    module.install_locked_from_feed(uv, root, Path(sys.executable), env)
            self.assertEqual(len(exported), 1)
            text = exported[0]
            hashes = re.findall(r"--hash=(sha256:[a-f0-9]{64})", text)
            self.assertTrue(hashes)
            self.assertTrue(set(hashes).issubset(known_hashes))
            self.assertNotIn("://", text)
            self.assertNotIn(" @ ", text)
            pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)", text, flags=re.M)
            self.assertTrue(pins)
            self.assertTrue(
                set(pins).issubset({(p["name"], p["version"]) for p in lock["package"]})
            )
            self.assertEqual((root / "uv.lock").read_bytes(), original_lock)

    def test_direct_export_source_fails_before_install(self):
        from hermes_cli import package_install as module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uv.lock").write_text("version=1\npackage=[]\n")
            (root / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
            for line in (
                "example @ https://example.invalid/example.whl",
                "--extra-index-url https://example.invalid/simple",
                "-e ../other",
            ):
                commands = []

                def run(cmd, **kwargs):
                    commands.append(cmd)
                    Path(cmd[cmd.index("--output-file") + 1]).write_text(line)

                with (
                    self.subTest(line=line),
                    patch.object(module.subprocess, "run", side_effect=run),
                ):
                    with self.assertRaisesRegex(ValueError, "direct source"):
                        module.install_locked_from_feed(
                            "uv", root, Path(sys.executable), {}
                        )
                self.assertEqual(len(commands), 1)


@pytest.fixture(params=["early", "normal", "normal-stale", "normal-no-env"])
def live_feed_install(request, tmp_path, monkeypatch):
    from hermes_cli import _early_recovery as er, _install_repair as early
    from hermes_cli import main_install_repair as normal, package_install

    root = tmp_path
    fallback = request.param in ("normal-stale", "normal-no-env")
    scripts = root / ("runtime" if fallback else "venv") / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.touch()
    if fallback:
        monkeypatch.setattr(sys, "executable", str(python))
    shims = [scripts / f"{name}.exe" for name in ("hermes", "hermes-gateway")]
    for shim in shims:
        shim.write_bytes(b"original shim")
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools==83.0.0"]\n'
        '[project.scripts]\nhermes="hermes_cli.main:main"\n'
    )
    (root / "uv.lock").write_text(
        'version=1\n[[package]]\nname="setuptools"\nversion="83.0.0"\n'
        'source={registry="https://pypi.org/simple"}\n'
        'wheels=[{hash="sha256:' + "a" * 64 + '"}]\n'
    )
    env = {
        "HERMES_HOME": str(root),
        "VIRTUAL_ENV": str(root / "venv"),
        "UV_DEFAULT_INDEX": "https://example.invalid/simple",
    }
    if request.param == "normal-no-env":
        env.pop("VIRTUAL_ENV")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("UV_DEFAULT_INDEX", env["UV_DEFAULT_INDEX"])
    monkeypatch.setattr(Path, "home", lambda: root)
    monkeypatch.setitem(sys.modules, "hermes_cli.main", SimpleNamespace(PROJECT_ROOT=root))
    monkeypatch.setattr(er, "_find_uv_binary", lambda: "uv")
    monkeypatch.setattr(early, "_stdout_to_stderr", nullcontext)
    monkeypatch.setattr(normal, "_verify_core_dependencies_installed", lambda *a, **kw: None)
    monkeypatch.setattr(normal, "_verify_console_scripts_installed", lambda *a, **kw: None)
    monkeypatch.setattr(normal, "_QUARANTINE_BACKOFF_MS", (0,))
    state = SimpleNamespace(installs=[], blocked_after=None, fail_at=None)
    real_rename = os.rename

    def rename(source, target, *args, **kwargs):
        if (Path(source) == shims[0] and state.blocked_after is not None
                and len(state.installs) >= state.blocked_after):
            raise PermissionError("shim held open")
        return real_rename(source, target, *args, **kwargs)

    def run(cmd, **kwargs):
        if "export" in cmd:
            Path(cmd[cmd.index("--output-file") + 1]).write_text(
                "example==1.0 --hash=sha256:" + "b" * 64 + "\n"
            )
        else:
            assert cmd[:3] == ["uv", "pip", "install"]
            protected = all(not shim.exists() for shim in shims)
            state.installs.append((cmd, kwargs, protected))
            if state.fail_at == len(state.installs) - 1:
                raise subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(package_install.subprocess, "run", run)
    if request.param == "early":
        state.install = lambda: early.run_core_install(root)
        state.error = early.ShimQuarantineError
    else:
        state.install = lambda: normal._install_python_dependencies_with_optional_fallback(
            ["uv", "pip"], env=env
        )
        state.error = normal.ShimQuarantineError
    expected_env = dict(env)
    if fallback:
        expected_env.pop("VIRTUAL_ENV", None)
    state.python, state.env, state.shims = python, expected_env, shims
    return state


@pytest.mark.windows_only
@pytest.mark.parametrize("blocked_after", [0, 1, 2])
def test_live_feed_refuses_before_each_contended_mutation(live_feed_install, blocked_after):
    state = live_feed_install
    state.blocked_after = blocked_after
    with pytest.raises(state.error):
        state.install()
    # Initial contention permits ZERO dependency mutations. Contention acquired
    # between steps must also be checked before the next command, not just once.
    assert len(state.installs) == blocked_after
    assert all(protected for _, _, protected in state.installs)
    assert all(shim.read_bytes() == b"original shim" for shim in state.shims)


@pytest.mark.windows_only
@pytest.mark.parametrize("fail_at", [None, 0, 1, 2])
def test_live_feed_protects_target_and_restores_shims(live_feed_install, fail_at):
    state = live_feed_install
    state.fail_at = fail_at
    with pytest.raises(subprocess.CalledProcessError) if fail_at is not None else nullcontext():
        state.install()
    assert len(state.installs) == (3 if fail_at is None else fail_at + 1)
    for cmd, kwargs, protected in state.installs:
        assert protected, "every live-venv mutation must run with shims quarantined"
        assert cmd[cmd.index("--python") + 1] == str(state.python)
        assert kwargs["env"].get("VIRTUAL_ENV") == state.env.get("VIRTUAL_ENV")
        assert kwargs["env"]["UV_DEFAULT_INDEX"] == state.env["UV_DEFAULT_INDEX"]
        assert kwargs["env"]["UV_NO_CONFIG"] == "1"
    # Runtime/build installs and already-satisfied editable installs do not
    # replace shims; success and failure must both restore the original files.
    assert all(shim.read_bytes() == b"original shim" for shim in state.shims)
    assert not list(state.shims[0].parent.glob("*.old.*"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
