"""Stdlib-only regression tests; runnable without the Hermes venv or pytest."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hermes_cli import _early_recovery as er
import hermes_constants


class IsolatedHomeTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"HERMES_HOME": tmp}))


class ManagedUvTests(IsolatedHomeTests):
    def test_managed_uv_precedes_legacy_user_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            managed = home / "managed"
            exe = "uv.exe" if sys.platform == "win32" else "uv"
            new_uv = managed / "bin" / exe
            old_uv = home / ".local" / "bin" / exe
            for p in (new_uv, old_uv):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            with (
                patch.object(
                    hermes_constants, "get_default_hermes_root", return_value=managed
                ),
                patch.object(Path, "home", return_value=home),
            ):
                self.assertEqual(er._find_uv_binary(), str(new_uv))


class ReadOnlyVerificationTests(IsolatedHomeTests):
    def setUp(self):
        super().setUp()
        from types import SimpleNamespace
        import importlib
        from hermes_cli import main_install_repair as ir

        # Load before patch.dict snapshots sys.modules; do not leave a stale
        # package attribute pointing at a module removed by fixture cleanup.
        importlib.import_module("hermes_cli.package_install")
        self.ir = ir
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "pyproject.toml").write_text(
            '[project]\ndependencies = ["missing-dep>=1"]\n'
            '[project.scripts]\nhermes = "hermes_cli.main:main"\n'
        )
        self.scripts = self.root / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
        self.scripts.mkdir(parents=True)
        (self.scripts / ("python.exe" if sys.platform == "win32" else "python")).touch()
        self.shim = self.scripts / "hermes.exe"
        self.shim.touch()
        self.env = {"VIRTUAL_ENV": str(self.root / "venv"),
                    "UV_DEFAULT_INDEX": "https://example.invalid/simple"}
        self.enterContext(patch.dict(sys.modules, {
            "hermes_cli.main": SimpleNamespace(PROJECT_ROOT=self.root)}))
        self.probe = self.enterContext(patch.object(
            ir, "_venv_probe", return_value=SimpleNamespace(stdout="", returncode=0)))
        self.installs = self.enterContext(patch.object(ir, "_run_install_with_heartbeat"))

    def verify_custom_feed(self):
        import importlib
        package_install = importlib.import_module("hermes_cli.package_install")

        # Skip only the preceding locked install; execute BOTH real verifiers.
        with patch.object(package_install, "install_locked_from_feed"):
            self.ir._install_python_dependencies_with_optional_fallback(["uv", "pip"], env=self.env)

    def test_custom_feed_missing_dependencies_never_repairs(self):
        self.probe.return_value.stdout = "missing-dep\n"
        with self.assertRaisesRegex(RuntimeError, "missing-dep.*repair disabled"):
            self.verify_custom_feed()
        self.installs.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows console shims")
    def test_custom_feed_missing_shim_never_repairs(self):
        self.shim.unlink()
        with self.assertRaisesRegex(RuntimeError, "hermes.*repair disabled"):
            self.verify_custom_feed()
        self.installs.assert_not_called()

    def test_custom_feed_success_is_read_only(self):
        self.verify_custom_feed()
        self.probe.assert_called_once()
        self.installs.assert_not_called()

    def test_custom_feed_probe_failure_is_not_success(self):
        for failure in (OSError("probe unavailable"), None):
            with self.subTest(failure=failure):
                self.probe.side_effect = failure
                self.probe.return_value.returncode = 1
                with self.assertRaisesRegex(RuntimeError, "dependency verification failed"):
                    self.verify_custom_feed()
                self.installs.assert_not_called()

    def test_default_dependency_repair_remains_enabled(self):
        self.probe.return_value.stdout = "missing-dep\n"
        self.ir._verify_core_dependencies_installed(
            ["uv", "pip"], env={"VIRTUAL_ENV": self.env["VIRTUAL_ENV"]})
        commands = [call.args[0] for call in self.installs.call_args_list]
        self.assertIn(["uv", "pip", "install", "--reinstall", "-e", "."], commands)
        self.assertIn(["uv", "pip", "install", "--reinstall", "missing-dep>=1"], commands)

    @unittest.skipUnless(sys.platform == "win32", "Windows console shims")
    def test_default_shim_repair_remains_enabled(self):
        self.shim.unlink()
        self.ir._verify_console_scripts_installed(
            ["uv", "pip"], env={"VIRTUAL_ENV": self.env["VIRTUAL_ENV"]})
        self.assertEqual(self.installs.call_args.args[0],
                         ["uv", "pip", "install", "--reinstall", "-e", "."])


class PackageSourceTests(IsolatedHomeTests):
    def test_policy_overrides_public_indexes_without_mutating_input(self):
        fn = getattr(er, "package_source_env", None)
        self.assertTrue(
            callable(fn), "early startup must load the persistent package feed policy"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package-sources.json").write_text(
                json.dumps({"python_index_url": "https://example.invalid/simple"})
            )
            original = {
                "UV_DEFAULT_INDEX": "https://pypi.org/simple",
                "UV_EXTRA_INDEX_URL": "https://pypi.org/simple",
                "PIP_EXTRA_INDEX_URL": "https://pypi.org/simple",
                "KEEP": "yes",
            }
            env = fn(original, root=root)
            for key in ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "PIP_INDEX_URL"):
                self.assertEqual(env[key], "https://example.invalid/simple")
            self.assertNotIn("UV_EXTRA_INDEX_URL", env)
            self.assertNotIn("PIP_EXTRA_INDEX_URL", env)
            self.assertEqual(env["KEEP"], "yes")
            self.assertEqual(original["UV_DEFAULT_INDEX"], "https://pypi.org/simple")

    def test_managed_python_env_keeps_persistent_feed(self):
        from hermes_cli.managed_uv import managed_python_env

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package-sources.json").write_text(
                json.dumps({"python_index_url": "https://example.invalid/simple"})
            )
            with patch.object(
                hermes_constants, "get_default_hermes_root", return_value=root
            ):
                env = managed_python_env(
                    root, base_env={"UV_DEFAULT_INDEX": "https://pypi.org/simple"}
                )
            self.assertEqual(env["UV_DEFAULT_INDEX"], "https://example.invalid/simple")
            self.assertEqual(env["UV_NO_CONFIG"], "1")

    def test_cli_applies_feed_before_early_recovery(self):
        import subprocess
        import textwrap

        # -S removes site packages: package policy must work even when YAML and
        # the installed dependency graph are unavailable. Stop at the repair
        # boundary, never repair this interpreter or import the heavy CLI graph.
        probe = textwrap.dedent("""
            import json, os, sys
            from hermes_cli import _early_recovery
            def observe():
                keys = ('UV_DEFAULT_INDEX', 'UV_EXTRA_INDEX_URL')
                print(json.dumps({k: os.environ[k] for k in keys if k in os.environ}))
                raise SystemExit(0)
            _early_recovery.recover_if_needed = observe
            import hermes_cli.main
        """)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "package-sources.json"
            env = {
                k: v
                for k, v in os.environ.items()
                if k.upper()
                in {"SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "PATH"}
            }
            env.update(
                HERMES_HOME=str(root / "profiles" / "a"),
                HOME=str(root),
                USERPROFILE=str(root),
                LOCALAPPDATA=str(root / "local"),
                UV_DEFAULT_INDEX="https://pypi.org/simple",
                UV_EXTRA_INDEX_URL="https://pypi.org/simple",
            )
            for valid in (True, False):
                policy.write_text(
                    json.dumps({
                        "python_index_url": "https://example.invalid/simple"
                        if valid
                        else "http://example.invalid/simple"
                    })
                )
                result = subprocess.run(
                    [sys.executable, "-S", "-c", probe],
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if valid:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    observed = json.loads(result.stdout)
                    self.assertEqual(
                        observed["UV_DEFAULT_INDEX"], "https://example.invalid/simple"
                    )
                    self.assertNotIn("UV_EXTRA_INDEX_URL", observed)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("HTTPS URL", result.stderr)

    def test_early_core_install_uses_locked_feed_path(self):
        from hermes_cli import _install_repair as ir, package_install
        import contextlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "UV_DEFAULT_INDEX": "https://example.invalid/simple",
                "VIRTUAL_ENV": str(root / "venv"),
            }
            with (
                patch.object(
                    ir, "_resolve_install_target", return_value=(["uv", "pip"], env)
                ),
                patch.object(
                    ir, "_stdout_to_stderr", side_effect=contextlib.nullcontext
                ),
                patch.object(er, "_run_ensurepip"),
                patch.object(ir, "_run_install_cmd") as unlocked,
                patch.object(package_install, "install_locked_from_feed") as locked,
            ):
                ir.run_core_install(root)
            locked.assert_called_once()
            unlocked.assert_not_called()

    def test_regular_update_uses_locked_feed_path(self):
        from hermes_cli import main_install_repair as ir, package_install
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = (
                root
                / "venv"
                / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            )
            python.parent.mkdir(parents=True)
            python.touch()
            env = {
                "UV_DEFAULT_INDEX": "https://example.invalid/simple",
                "VIRTUAL_ENV": str(root / "venv"),
            }
            with (
                patch.dict(
                    sys.modules, {"hermes_cli.main": SimpleNamespace(PROJECT_ROOT=root)}
                ),
                patch.object(ir, "_venv_scripts_dir", return_value=None),
                patch.object(ir, "_verify_core_dependencies_installed"),
                patch.object(ir, "_verify_console_scripts_installed"),
                patch.object(package_install, "install_locked_from_feed") as locked,
            ):
                ir._install_python_dependencies_with_optional_fallback(
                    ["uv", "pip"], env=env
                )
            self.assertEqual(locked.call_args.args[1], root)
            self.assertEqual(locked.call_args.args[2], python)

    def test_custom_feed_without_env_target_uses_running_python(self):
        from types import SimpleNamespace
        from hermes_cli import main_install_repair as ir, package_install

        for stale in (False, True):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env = {
                    "HERMES_HOME": str(root),
                    "UV_DEFAULT_INDEX": "https://example.invalid/simple",
                }
                if stale:
                    env["VIRTUAL_ENV"] = str(root / "missing-venv")
                original = dict(env)
                with (
                    patch.dict(sys.modules, {"hermes_cli.main": SimpleNamespace(PROJECT_ROOT=root)}),
                    patch.object(ir, "_venv_scripts_dir", return_value=None),
                    patch.object(ir, "_verify_core_dependencies_installed"),
                    patch.object(ir, "_verify_console_scripts_installed"),
                    patch.object(package_install, "install_locked_from_feed") as locked,
                ):
                    ir._install_python_dependencies_with_optional_fallback(
                        ["uv", "pip"], env=env
                    )
                self.assertEqual(locked.call_args.args[2], Path(sys.executable))
                self.assertNotIn("VIRTUAL_ENV", locked.call_args.args[3])
                self.assertEqual(env, original)

    def test_early_core_install_requires_uv_for_custom_feed(self):
        import contextlib
        from hermes_cli import _install_repair as ir, package_install

        for source in ("environment", "policy", "default", "public"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env = {"HERMES_HOME": str(root)}
                if source == "policy":
                    (root / "package-sources.json").write_text(json.dumps({
                        "python_index_url": "https://example.invalid/simple"
                    }))
                elif source != "default":
                    env["UV_DEFAULT_INDEX"] = (
                        "https://pypi.org/simple/" if source == "public"
                        else "https://example.invalid/simple"
                    )
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(Path, "home", return_value=root),
                    patch.object(er, "_find_uv_binary", return_value=None),
                    patch.object(ir, "_stdout_to_stderr", side_effect=contextlib.nullcontext),
                    patch.object(er, "_run_ensurepip") as bootstrap,
                    patch.object(ir.subprocess, "run") as child,
                    patch.object(ir, "_run_install_cmd") as unlocked,
                    patch.object(package_install, "install_locked_from_feed") as locked,
                ):
                    if source in ("environment", "policy"):
                        with self.assertRaisesRegex(RuntimeError, "requires uv"):
                            ir.run_core_install(root)
                        bootstrap.assert_not_called()
                        unlocked.assert_not_called()
                    else:
                        ir.run_core_install(root)
                        bootstrap.assert_called_once_with(root)
                        self.assertEqual(
                            unlocked.call_args.args[0],
                            [sys.executable, "-m", "pip", "install", "-e", ".[all]"],
                        )
                        self.assertIsNone(unlocked.call_args.kwargs["env"])
                    locked.assert_not_called()
                    child.assert_not_called()

    def test_regular_update_requires_uv_for_custom_feed(self):
        from hermes_cli import main_install_repair as ir, package_install, update_cmd

        for source in ("environment", "policy", "explicit", "default", "public"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env = {"HERMES_HOME": str(root)}
                if source == "policy":
                    (root / "package-sources.json").write_text(json.dumps({
                        "python_index_url": "https://example.invalid/simple"
                    }))
                elif source not in ("default", "explicit"):
                    env["UV_DEFAULT_INDEX"] = (
                        "https://pypi.org/simple/" if source == "public"
                        else "https://example.invalid/simple"
                    )
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(Path, "home", return_value=root),
                    patch.object(ir, "_venv_scripts_dir", return_value=None),
                    patch.object(ir, "_verify_core_dependencies_installed"),
                    patch.object(ir, "_verify_console_scripts_installed"),
                    patch.object(ir.subprocess, "run") as child,
                    patch.object(ir, "_run_quarantined_install") as unlocked,
                    patch.object(package_install, "install_locked_from_feed") as locked,
                ):
                    prefix, install_env = update_cmd._pip_install_prefix(None)
                    prefixes = [prefix]
                    if source == "explicit":
                        install_env = {"UV_DEFAULT_INDEX": "https://example.invalid/simple"}
                        # A pip binary (even one whose name contains uv) is not uv.
                        prefixes += [[str(root / name)] for name in ("pip", "pip.exe", "uv-pip.exe")]
                    for prefix in prefixes:
                        with self.subTest(prefix=prefix):
                            if source in ("environment", "policy", "explicit"):
                                with self.assertRaisesRegex(RuntimeError, "requires uv"):
                                    ir._install_python_dependencies_with_optional_fallback(
                                        prefix, env=install_env
                                    )
                                unlocked.assert_not_called()
                            else:
                                ir._install_python_dependencies_with_optional_fallback(
                                    prefix, env=install_env
                                )
                                self.assertEqual(
                                    unlocked.call_args.args[0],
                                    prefix + ["install", "-e", ".[all]"],
                                )
                                self.assertIsNone(unlocked.call_args.kwargs["env"])
                            locked.assert_not_called()
                            child.assert_not_called()

    def test_npm_policy_pins_registry_and_managed_child_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "node" / ("node.exe" if sys.platform == "win32" else "node")
            node.parent.mkdir()
            node.touch()
            (root / "package-sources.json").write_text(
                json.dumps({"npm_registry": "https://example.invalid/npm/"})
            )
            env = er.package_source_env(
                {
                    "PATH": "system-path",
                    "npm_config_registry": "https://registry.npmjs.org/",
                },
                root=root,
            )
            self.assertEqual(env["NPM_CONFIG_REGISTRY"], "https://example.invalid/npm/")
            self.assertNotIn("npm_config_registry", env)
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(node.parent))
            self.assertNotIn("UV_DEFAULT_INDEX", env)

    def test_shared_policy_is_profile_independent_and_not_cached(self):
        from hermes_cli.managed_uv import managed_python_env

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = {"python_index_url": "https://example.invalid/simple"}
            (root / "package-sources.json").write_text(json.dumps(policy))
            for name in ("a", "b", "a"):
                home = root / "profiles" / name
                home.mkdir(parents=True, exist_ok=True)
                # Profile-local policy is deliberately ignored: there is one
                # shared installation and independent per-profile app configs.
                (home / "package-sources.json").write_text("{invalid profile policy")
                with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
                    self.assertEqual(hermes_constants.get_default_hermes_root(), root)
                    env = managed_python_env(root / "checkout", base_env={})
                    self.assertEqual(
                        env["UV_DEFAULT_INDEX"], policy["python_index_url"]
                    )
            (root / "package-sources.json").write_text("{}")
            with patch.dict(os.environ, {"HERMES_HOME": str(root)}):
                self.assertNotIn(
                    "UV_DEFAULT_INDEX", managed_python_env(root, base_env={})
                )

    def test_candidate_install_uses_feed_without_public_sync_fallback(self):
        import subprocess
        from types import SimpleNamespace
        from hermes_cli import managed_uv, package_install

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uv.lock").touch()
            (root / "package-sources.json").write_text(
                json.dumps({"python_index_url": "https://example.invalid/simple"})
            )
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(root)}),
                patch.object(
                    managed_uv.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                patch.object(managed_uv, "_stream_sync") as public_sync,
                patch.object(
                    managed_uv, "_smoke_candidate_venv", return_value=(True, "", None)
                ),
                patch.object(managed_uv, "_reject"),
                patch.object(package_install, "install_locked_from_feed") as locked,
            ):
                candidate = managed_uv._stage_candidate_venv(
                    "uv",
                    project_root=root,
                    generation=root / "generation",
                    python=root / "python",
                )
                self.assertEqual(
                    locked.call_args.args[2], managed_uv._venv_python(candidate)
                )
                self.assertEqual(
                    locked.call_args.args[3]["UV_DEFAULT_INDEX"],
                    "https://example.invalid/simple",
                )
                locked.side_effect = subprocess.CalledProcessError(
                    1, ["uv", "pip", "install"]
                )
                with self.assertRaises(managed_uv._CandidateStageError):
                    managed_uv._stage_candidate_venv(
                        "uv",
                        project_root=root,
                        generation=root / "generation",
                        python=root / "python",
                    )
                public_sync.assert_not_called()

    def test_missing_policy_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {
                "KEEP": "yes",
                "UV_DEFAULT_INDEX": "https://example.invalid/existing",
            }
            self.assertEqual(er.package_source_env(original, root=root), original)
            for policy in ({}, {"python_index_url": None, "npm_registry": None}):
                (root / "package-sources.json").write_text(json.dumps(policy))
                self.assertEqual(er.package_source_env(original, root=root), original)

    def test_invalid_policy_cannot_fall_back_to_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for value in (
                "http://example.invalid/simple",
                "https://user:secret@example.invalid/simple",
                "https://example.invalid/simple?token=secret",
                "https://example.invalid/simple\n",
                "https://example.invalid/simple#fragment",
                "https://example.invalid:notaport/simple",
                "https://example.invalid:70000/simple",
                "",
                42,
                True,
            ):
                for key in ("python_index_url", "npm_registry"):
                    (root / "package-sources.json").write_text(json.dumps({key: value}))
                    with (
                        self.subTest(key=key, value=value),
                        self.assertRaises(ValueError),
                    ):
                        er.package_source_env({}, root=root)
            for raw in ("not json", "[]", '{"unknown": "https://example.invalid"}'):
                (root / "package-sources.json").write_text(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    er.package_source_env({}, root=root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
