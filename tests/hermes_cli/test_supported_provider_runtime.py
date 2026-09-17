"""Exercise the actual resolver with isolated config, without account/network access."""
import json
import pytest
from hermes_cli.provider_policy import UnsupportedProviderError


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  provider: auto\n  default: test-model\nfree_tier:\n  enabled: false\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("provider", ["openrouter", "anthropic", "openai-codex", "gemini", "vertex", "bedrock", "copilot-acp", "moa", "openai", "xai-oauth"])
def test_explicit_retired_provider_fails_before_credentials(home, monkeypatch, provider):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import resolve_provider
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "load_pool", lambda *a, **k: pytest.fail("credential pool consulted for retired account"))
    for resolve in (lambda: resolve_provider(provider), lambda: resolve_runtime_provider(requested=provider, explicit_base_url="https://example.test/v1")):
        with pytest.raises(UnsupportedProviderError):
            resolve()


def test_retired_env_and_config_are_not_silently_rerouted(home, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import resolve_provider, AuthError
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    with pytest.raises(UnsupportedProviderError):
        resolve_runtime_provider()
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-retired-key")
    with pytest.raises(AuthError) as error:
        resolve_provider("auto", skip_free_tier=True)
    assert error.value.code == "no_provider_configured"
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n", encoding="utf-8")
    with pytest.raises(UnsupportedProviderError):
        resolve_provider("auto")


@pytest.mark.parametrize("url", [None, "", "not-a-url", "file:///tmp/socket", "https://"])
def test_custom_requires_valid_http_endpoint(home, url):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    with pytest.raises(ValueError, match="endpoint"):
        resolve_runtime_provider(requested="custom", explicit_api_key="test-key", explicit_base_url=url)


def test_explicit_url_means_custom_not_openrouter(home):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider(requested="auto", explicit_base_url="https://example.test/v1", explicit_api_key="test-key")
    assert runtime["provider"] == "custom"
    assert runtime["base_url"] == "https://example.test/v1"
    assert runtime["api_key"] == "test-key"


def test_named_custom_profiles_are_isolated_a_b_a(home, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import build_profile_secret_scope, set_secret_scope, reset_secret_scope, set_multiplex_active
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import resolve_provider
    homes = []
    for label, mode in [("a", "responses"), ("b", "anthropic_messages")]:
        directory = home / label
        directory.mkdir()
        config = {"model": {"provider": "lab", "default": "test-model"}, "providers": {
            "lab": {"base_url": f"https://{label}.example.test/v1", "key_env": "LAB_KEY", "api_mode": mode,
                    "extra_headers": {"X-Project": label}, "extra_body": {"region": label}}
        }}
        (directory / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        (directory / ".env").write_text(f"LAB_KEY=test-key-{label}\n", encoding="utf-8")
        homes.append(directory)
    for directory, mode in [(homes[0], "codex_responses"), (homes[1], "anthropic_messages"), (homes[0], "codex_responses")]:
        token = set_hermes_home_override(str(directory))
        set_multiplex_active(True)
        secret_token = set_secret_scope(build_profile_secret_scope(directory))
        try:
            assert resolve_provider("auto") == "custom:lab"
            runtime = resolve_runtime_provider()
            assert runtime["base_url"] == f"https://{directory.name}.example.test/v1"
            assert runtime["api_key"] == f"test-key-{directory.name}"
            assert runtime["api_mode"] == mode
            assert runtime["extra_headers"] == {"X-Project": directory.name}
            assert runtime["request_overrides"]["extra_body"] == {"region": directory.name}
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(token)
            set_multiplex_active(False)


def test_bare_custom_explicit_endpoint_preserves_selected_protocol(home):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    (home / "config.yaml").write_text("model:\n  provider: custom\n  default: test-model\n  api_mode: responses\n", encoding="utf-8")
    result = resolve_runtime_provider(requested="custom", explicit_base_url="https://example.test/v1", explicit_api_key="test-key")
    assert result["api_mode"] == "codex_responses"
