"""Configured custom storage keys are not inference provider identities."""
import json

import pytest
import yaml

from agent.credential_pool import custom_provider_pool_key_candidates, load_pool


def _home(root, name, url, pools):
    home = root / name
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "providers": {"b-ai": {"name": "B.AI", "base_url": url}},
    }))
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": pools}))
    return home


def _row(token, url=""):
    return {"id": token, "source": "manual", "auth_type": "api_key", "priority": 0,
            "access_token": token, "base_url": url}


@pytest.mark.parametrize("storage_key", ["b-ai", "custom:b.ai", "custom:b-ai"])
@pytest.mark.parametrize("same_endpoint", [True, False])
def test_configured_storage_is_profile_and_endpoint_scoped(tmp_path, monkeypatch, storage_key, same_endpoint):
    url_a = "https://a.example/v1"
    url_b = url_a if same_endpoint else "https://b.example/v1"
    a = _home(tmp_path, "a", url_a, {storage_key: [_row("token-a", url_a)]})
    b = _home(tmp_path, "b", url_b, {storage_key: [_row("token-b", url_b), _row("wrong-endpoint", "https://other.example/v1")]})
    for home, token, url in [(a, "token-a", url_a), (b, "token-b", url_b), (a, "token-a", url_a)]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        before = (home / "auth.json").read_bytes()
        assert storage_key in custom_provider_pool_key_candidates(url, "b-ai")
        assert [entry.access_token for entry in load_pool(storage_key).entries()] == [token]
        assert (home / "auth.json").read_bytes() == before


def test_unknown_raw_pool_does_not_become_a_provider(tmp_path, monkeypatch):
    home = _home(tmp_path, "a", "https://a.example/v1", {"unknown": [_row("saved-token")]})
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = (home / "auth.json").read_bytes()
    assert not load_pool("unknown").has_credentials()
    assert (home / "auth.json").read_bytes() == before


def test_named_candidates_require_the_configured_endpoint(tmp_path, monkeypatch):
    home = _home(tmp_path, "a", "https://a.example/v1", {"b-ai": [_row("saved-token")]})
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert custom_provider_pool_key_candidates("https://other.example/v1", "b-ai") == []
    assert load_pool("b-ai").select().access_token == "saved-token"


@pytest.mark.parametrize("reserved", ["nous", "copilot", "custom", "openai-codex", "xai-oauth"])
def test_custom_name_cannot_borrow_builtin_or_service_storage(tmp_path, monkeypatch, reserved):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "providers": {reserved: {"name": "Explicit endpoint", "base_url": "https://custom.example/v1"}},
    }))
    assert custom_provider_pool_key_candidates("https://custom.example/v1", reserved) == ["custom:explicit-endpoint"]


def test_retired_selection_stays_rejected_with_explicit_custom_config(tmp_path, monkeypatch):
    home = tmp_path / "retired"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "providers": {"openrouter": {"base_url": "https://explicit.example/v1"}},
    }))
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {},
        "credential_pool": {"openrouter": [_row("custom-token")]}}))
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.provider_policy import UnsupportedProviderError
    # Existing resolver semantics allow a configured bare name, but it must
    # resolve as custom, never revive the retired built-in implementation.
    runtime = resolve_runtime_provider(requested="openrouter")
    assert runtime["provider"] == "custom"
    runtime = resolve_runtime_provider(requested="custom:openrouter")
    assert runtime["api_key"] == "custom-token"
    assert runtime["base_url"] == "https://explicit.example/v1"
    (home / "config.yaml").write_text("{}")
    with pytest.raises(UnsupportedProviderError):
        resolve_runtime_provider(requested="openrouter")
    assert not load_pool("openrouter").has_credentials()


@pytest.mark.parametrize("same_url", [True, False])
def test_auth_add_custom_identities_do_not_share_labels_or_endpoints(tmp_path, monkeypatch, same_url):
    from types import SimpleNamespace
    from hermes_cli.auth_commands import auth_add_command
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent.auxiliary_client import resolve_provider_client
    from agent.credential_pool import credential_pool_matches_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    urls = {"first": "https://first.example/v1",
            "second": "https://first.example/v1" if same_url else "https://second.example/v1"}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"providers": {
        key: {"name": "Shared Label", "base_url": url} for key, url in urls.items()
    }}))
    (tmp_path / "auth.json").write_text(json.dumps({"version": 1, "providers": {},
        "credential_pool": {"custom:shared-label": [_row("ambiguous-token")]}}))
    for key in urls:
        auth_add_command(SimpleNamespace(provider=key, auth_type="api-key", api_key=f"token-{key}", label=key))
    before = (tmp_path / "auth.json").read_bytes()
    for key, url in urls.items():
        candidates = custom_provider_pool_key_candidates(url, key)
        assert f"custom:{key}" in candidates
        assert "custom:shared-label" not in candidates
        runtime = resolve_runtime_provider(requested=f"custom:{key}")
        assert runtime["api_key"] == f"token-{key}"
        client, model = resolve_provider_client(f"custom:{key}", "test-model")
        assert client.api_key == f"token-{key}"
        assert str(client.base_url).rstrip("/") == url
        assert credential_pool_matches_provider(f"custom:{key}", key, base_url=url)
        sibling = "second" if key == "first" else "first"
        assert not credential_pool_matches_provider(f"custom:{key}", sibling, base_url=urls[sibling])
    assert (tmp_path / "auth.json").read_bytes() == before
    if same_url:
        assert custom_provider_pool_key_candidates(urls["first"]) == []
