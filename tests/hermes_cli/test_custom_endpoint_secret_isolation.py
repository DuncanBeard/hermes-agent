"""Endpoint overrides must not export credentials from the original config."""
from unittest.mock import Mock

import pytest

from hermes_cli import runtime_provider as rp

ORIGINAL = "https://original.example/v1"


@pytest.fixture
def isolated(monkeypatch):
    config = {"model": {"provider": "custom", "default": "test-model", "base_url": ORIGINAL}}
    monkeypatch.setattr(rp, "load_config", lambda: config)
    monkeypatch.setattr(rp._config_mod, "load_config", lambda: config)
    monkeypatch.setattr(rp, "load_pool", lambda *a, **k: None)
    monkeypatch.setattr(rp, "_host_gated_env_key_candidates", lambda *a, **k: [])
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    command = Mock(return_value=lambda: "SYNTHETIC_COMMAND_KEY")
    monkeypatch.setattr("agent.command_token_source.build_command_token_provider", command)
    monkeypatch.setattr(rp, "_auto_detect_local_model", Mock(side_effect=AssertionError("no discovery")))
    return config, command


@pytest.mark.parametrize("source", ["api_key", "key_env", "api_key_env", "key_cmd", "extra_headers"])
@pytest.mark.parametrize("destination", ["https://different.example/v1", "http://original.example/v1", "https://original.example/v2", "https://original.example/V1"])
@pytest.mark.parametrize("override", ["explicit", "env", "named"])
def test_changed_destination_does_not_inherit_secrets(isolated, monkeypatch, source, destination, override):
    config, command = isolated
    cfg = config["model"]
    if source in {"key_env", "api_key_env"}:
        cfg[source] = "SYNTHETIC_ORIGINAL_KEY_ENV"
        monkeypatch.setenv("SYNTHETIC_ORIGINAL_KEY_ENV", "SYNTHETIC_ORIGINAL_KEY")
    elif source == "extra_headers":
        cfg[source] = {"Authorization": "Bearer SYNTHETIC_ORIGINAL_KEY", "X-Private-Token": "SYNTHETIC_HEADER_KEY"}
    else:
        cfg[source] = "SYNTHETIC_ORIGINAL_KEY"
    cfg["capabilities"] = {"vision": True}
    cfg["extra_body"] = {"private_token": "SYNTHETIC_BODY_KEY"}
    kwargs = {"requested": "custom"}
    if override == "env":
        monkeypatch.setenv("CUSTOM_BASE_URL", destination)
    else:
        kwargs["explicit_base_url"] = destination
    if override == "named":
        config["providers"] = {"original": dict(cfg)}
        kwargs["requested"] = "custom:original"
    result = rp.resolve_runtime_provider(**kwargs)
    assert result["base_url"] == destination
    assert result["api_key"] == "no-key-required"
    assert not result.get("extra_headers")
    assert not result.get("request_overrides")
    assert result["capabilities"] == {"vision": True}
    command.assert_not_called()


@pytest.mark.parametrize("override", ["explicit", "env"])
def test_same_endpoint_inherits_config(isolated, monkeypatch, override):
    config, _ = isolated
    config["model"].update(api_key="SYNTHETIC_ORIGINAL_KEY", extra_headers={"Authorization": "Bearer SYNTHETIC_HEADER_KEY"})
    kwargs = {"requested": "custom"}
    if override == "env":
        monkeypatch.setenv("CUSTOM_BASE_URL", ORIGINAL + "/")
    else:
        kwargs["explicit_base_url"] = ORIGINAL + "/"
    result = rp.resolve_runtime_provider(**kwargs)
    assert result["api_key"] == "SYNTHETIC_ORIGINAL_KEY"
    assert result["extra_headers"] == config["model"]["extra_headers"]


def test_explicit_key_wins_at_new_destination(isolated):
    config, command = isolated
    config["model"].update(api_key="SYNTHETIC_ORIGINAL_KEY", key_cmd="original-command", extra_headers={"Authorization": "original"})
    result = rp.resolve_runtime_provider(requested="custom", explicit_base_url="https://different.example/v1", explicit_api_key="SYNTHETIC_EXPLICIT_KEY")
    assert result["api_key"] == "SYNTHETIC_EXPLICIT_KEY"
    assert not result.get("extra_headers")
    command.assert_not_called()


@pytest.mark.parametrize("override", ["explicit", "env"])
def test_destination_owned_named_credentials(isolated, monkeypatch, override):
    config, _ = isolated
    config["model"]["api_key"] = "SYNTHETIC_ORIGINAL_KEY"
    destination = "https://different.example/v1"
    config["providers"] = {"destination": {"base_url": destination, "api_key": "SYNTHETIC_DESTINATION_KEY", "extra_headers": {"X-Key": "SYNTHETIC_DESTINATION_HEADER"}}}
    kwargs = {"requested": "custom"}
    if override == "env":
        monkeypatch.setenv("CUSTOM_BASE_URL", destination)
    else:
        kwargs["explicit_base_url"] = destination
    result = rp.resolve_runtime_provider(**kwargs)
    assert result["api_key"] == "SYNTHETIC_DESTINATION_KEY"
    assert result["extra_headers"] == {"X-Key": "SYNTHETIC_DESTINATION_HEADER"}
