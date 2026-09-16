"""Hash-preserving installation through an explicitly configured package index."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib


def install_locked_from_feed(
    uv: str,
    root: Path,
    python: Path,
    env: dict,
    *,
    group: str = "all",
    run_install=None,
) -> None:
    """Keep the shipped graph/hashes, but let the selected index supply artifacts.

    Do not rewrite the public lockfile or fall back to an unlocked/public install.
    Reuse locked build tools when available; otherwise create a build-only hash
    snapshot through the approved feed, constrained by the locked runtime graph.

    Live-venv callers supply ``run_install`` for EVERY mutating command, not
    just the editable install. Their wrappers quarantine and restore per command,
    rechecking contention between steps; fresh candidates need no quarantine.
    """
    root = Path(root)
    with (root / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)
    with (root / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)
    build_lines = []
    snapshot_build = False
    for requirement in project.get("build-system", {}).get("requires", []):
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:==([^;\s]+))?", requirement)
        if not match:
            raise ValueError(f"Unsupported locked build requirement: {requirement}")
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        candidates = [
            p
            for p in lock["package"]
            if p["name"] == normalized and (version is None or p["version"] == version)
        ]
        if not candidates:
            snapshot_build = True
            continue
        if len(candidates) != 1 or "registry" not in candidates[0].get("source", {}):
            raise ValueError(
                f"Build requirement must have one registry entry in uv.lock: {requirement}"
            )
        package = candidates[0]
        artifacts = list(package.get("wheels", []))
        if package.get("sdist"):
            artifacts.append(package["sdist"])
        hashes = sorted({
            a["hash"]
            for a in artifacts
            if re.fullmatch(r"sha256:[a-f0-9]{64}", a.get("hash", ""))
        })
        if not hashes:
            raise ValueError(f"Missing lockfile hashes for build requirement: {name}")
        build_lines.append(
            f"{name}=={package['version']} " + " ".join("--hash=" + h for h in hashes)
        )

    install_env = dict(env)
    # The immutable exported graph replaces rolling resolver cutoff settings.
    # The approved feed still applies its own server-side package security gates.
    install_env["UV_NO_CONFIG"] = "1"

    def install(cmd: list[str]) -> None:
        if run_install is not None:
            run_install(cmd, install_env)
        else:
            # Legacy desktop hand-offs drain stdout only.
            subprocess.run(
                cmd, cwd=root, env=install_env, check=True, stderr=subprocess.STDOUT
            )

    with tempfile.TemporaryDirectory(prefix="hermes-feed-lock-") as tmp:
        requirements = Path(tmp) / "requirements.txt"
        build_requirements = Path(tmp) / "build.txt"
        subprocess.run(
            [
                uv,
                "export",
                "--frozen",
                "--no-dev",
                "--extra",
                group,
                "--no-emit-project",
                "--no-header",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements),
            ],
            cwd=root,
            env=install_env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        text = requirements.read_text(encoding="utf-8")
        # Direct URLs and index directives would bypass the selected registry.
        for line in text.splitlines():
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and (
                    "://" in stripped
                    or " @ " in stripped
                    or stripped.startswith((
                        "-e ",
                        "--index",
                        "--extra-index",
                        "--find-links",
                    ))
                )
            ):
                raise ValueError(
                    "Exported dependency has a direct source; refusing configured-feed bypass"
                )
        build_requirements.write_text("\n".join(build_lines) + "\n", encoding="utf-8")
        if snapshot_build:
            # Build-only requirements are not necessarily represented in uv.lock
            # (e.g. wheel). Resolve these through the approved feed and immediately
            # freeze hashes; never change the runtime dependency graph.
            subprocess.run(
                [
                    uv,
                    "pip",
                    "compile",
                    "--python",
                    str(python),
                    "--generate-hashes",
                    "--constraint",
                    str(requirements),
                    "--no-header",
                    "--output-file",
                    str(build_requirements),
                    "-",
                ],
                input="\n".join(project["build-system"]["requires"]),
                text=True,
                cwd=root,
                env=install_env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        print(
            "  -> Installing locked packages through the configured feed (SHA-256 verified)",
            flush=True,
        )
        prefix = [uv, "pip", "install", "--python", str(python)]
        for req in (requirements, build_requirements):
            if req == build_requirements and not build_lines and not snapshot_build:
                continue
            cmd = prefix + ["--no-deps", "--require-hashes", "-r", str(req)]
            if sys.platform == "win32":
                cmd += ["--only-binary", ":all:"]
            install(cmd)
        # Build using the verified build tools; do not re-resolve dependencies.
        project_cmd = prefix + ["--no-deps", "--no-build-isolation", "-e", "."]
        install(project_cmd)
