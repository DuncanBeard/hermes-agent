"""Packaging boundaries for the reduced inference-provider build."""
from pathlib import Path
import tomllib


def test_retired_cloud_inference_extras_are_not_installable():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert not ({"bedrock", "vertex", "azure-identity"} & extras.keys())


def test_shared_protocol_and_tool_dependencies_remain_available():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert any(dep.startswith("openai==") for dep in project["dependencies"])
    assert any(dep.startswith("anthropic==") for dep in extras["anthropic"])
    assert any(dep.startswith("agent-client-protocol==") for dep in extras["acp"])
    assert any(dep.startswith("google-auth==") for dep in extras["google"])
