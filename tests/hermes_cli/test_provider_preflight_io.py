"""Provider pins are policy decisions, never invitations to bootstrap another account."""
import json
from unittest.mock import Mock

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    return tmp_path


@pytest.mark.parametrize("source", ["config", "env", "env_with_nous"])
def test_bootstrap_rejects_retired_pin_without_mint(home, monkeypatch, source):
    from hermes_cli import anon_auth, free_tier_bootstrap as boot
    config = {"model": {"provider": "openrouter" if source == "config" else "auto"},
              "free_tier": {"enabled": True}}
    if source == "env_with_nous":
        config["model"]["provider"] = "nous"
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    if source.startswith("env"):
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    mint = Mock(return_value={"guest": True})
    monkeypatch.setattr(anon_auth, "ensure_portal_identity", mint)
    monkeypatch.setattr(anon_auth, "guest_enabled", lambda: True)
    boot.reset_for_tests()
    try:
        result = boot.run_bootstrap(announce=False)
        mint.assert_not_called()
        assert not result.provider_configured
        assert "Unsupported model provider" in result.error
        assert not result.failure.get("retryable")
        result = boot.retry_bootstrap_mint(force=True, announce=False)
        mint.assert_not_called()
        assert not result.provider_configured
    finally:
        boot.reset_for_tests()


@pytest.mark.parametrize("source", ["config", "env"])
def test_retired_pin_never_probes_local_model(home, monkeypatch, source):
    from hermes_cli import runtime_provider as rp
    from hermes_cli.provider_policy import UnsupportedProviderError
    config = {"model": {"provider": "openrouter" if source == "config" else "auto",
                        "base_url": "http://localhost:9999/v1", "default": ""}}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    if source == "env":
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    probe = Mock(return_value="local-model")
    monkeypatch.setattr(rp, "_auto_detect_local_model", probe)
    with pytest.raises(UnsupportedProviderError):
        rp.resolve_runtime_provider()
    probe.assert_not_called()


def test_cli_retired_override_stops_before_bootstrap(home, monkeypatch, capsys):
    from types import SimpleNamespace
    from hermes_cli import main, free_tier_bootstrap as boot
    (home / "config.yaml").write_text('model:\n  provider: nous\n', encoding="utf-8")
    bootstrap = Mock(side_effect=AssertionError("bootstrap reached"))
    monkeypatch.setattr(boot, "run_bootstrap", bootstrap)
    monkeypatch.setattr(main, "_resolve_use_tui", lambda args: False)
    monkeypatch.setattr(main, "_resolve_chat_session_args", lambda *args: None)
    monkeypatch.setattr(main, "_warn_retired_xai_models", lambda: None)
    with pytest.raises(SystemExit) as error:
        main.cmd_chat(SimpleNamespace(provider="openrouter", query=None))
    assert error.value.code == 1
    bootstrap.assert_not_called()
    assert "Unsupported model provider" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["custom", "lab"])
def test_valid_custom_local_model_is_still_discovered(home, monkeypatch, provider):
    from hermes_cli import runtime_provider as rp
    url = "http://localhost:9999/v1"
    config = {"model": {"provider": provider, "default": "", "base_url": url},
              "providers": {"lab": {"base_url": url}}}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    probe = Mock(return_value="local-model")
    monkeypatch.setattr(rp, "_auto_detect_local_model", probe)
    runtime = rp.resolve_runtime_provider(explicit_api_key="test-key")
    assert runtime["provider"] in {"custom", "custom:lab"}
    assert runtime["base_url"] == url
    assert rp._get_model_config()["default"] == "local-model"
    probe.assert_called_with(url)


@pytest.mark.parametrize("provider", [None, "nous"])
def test_blank_and_nous_bootstrap_still_create_guest(home, monkeypatch, provider):
    import httpx
    from hermes_cli import anon_auth, free_tier_bootstrap as boot
    from tests.hermes_cli.anon_portal import FakePortal
    config = {"model": {"provider": provider}} if provider else {}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(home / "shared"))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    monkeypatch.setenv("HERMES_ANON_API_SECRET", "test-secret")
    fake = FakePortal()
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(
        *a, **{**kw, "transport": httpx.MockTransport(fake.handler)}))
    anon_auth.reset_mint_memo_for_tests()
    boot.reset_for_tests()
    try:
        result = boot.run_bootstrap(announce=False)
        assert result.provider_configured and result.free_tier and result.has_identity
        assert result.inference_provider == "nous"
        assert fake.creates() == 1
        assert boot.run_bootstrap(announce=False) is result
        assert fake.creates() == 1
    finally:
        boot.reset_for_tests()
        anon_auth.reset_mint_memo_for_tests()
