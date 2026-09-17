"""Credential ownership survives delegation config -> child constructor kwargs."""
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from tools.delegate_tool_config import _resolve_child_runtime, _resolve_delegation_credentials


@pytest.fixture
def parent():
    return SimpleNamespace(model="parent-model", provider="copilot", api_mode="codex_responses",
                           base_url="https://api.githubcopilot.com", api_key="parent-secret")


def child_kwargs(parent, cfg):
    bundle = _resolve_delegation_credentials(cfg, parent)
    return _resolve_child_runtime(
        parent, cfg, parent.api_key, model=bundle["model"],
        override_provider=bundle["provider"], override_base_url=bundle["base_url"],
        override_api_key=bundle["api_key"], override_api_mode=bundle["api_mode"],
        override_acp_command=None, override_acp_args=None,
    )


@pytest.mark.parametrize("key", [None, "child-owned-key"])
def test_direct_child_endpoint_never_receives_parent_secret(parent, key):
    cfg = {"base_url": "https://child.example.test/v1", "model": "child-model"}
    if key is not None:
        cfg["api_key"] = key
    kwargs = child_kwargs(parent, cfg)
    assert kwargs["api_key"] != parent.api_key
    if key:
        assert kwargs["api_key"] == key
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "child-model", "choices": [{"index": 0, "message": {"role": "assistant",
            "content": "ok"}, "finish_reason": "stop"}]})

    with OpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"],
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        client.chat.completions.create(model=kwargs["model"], messages=[{"role": "user", "content": "hi"}])
    assert sent[0].url.host == "child.example.test"
    assert parent.api_key not in str(sent[0].headers)


def test_unpinned_child_inherits_parent_credentials(parent):
    kwargs = child_kwargs(parent, {})
    assert kwargs["provider"] == parent.provider
    assert kwargs["api_key"] == parent.api_key
    assert kwargs["base_url"] == parent.base_url


def test_named_custom_child_resolves_its_own_credential(parent, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "providers:\n  child:\n    base_url: https://child.example.test/v1\n"
        "    api_key: child-owned-key\n    default_model: child-model\n", encoding="utf-8")
    kwargs = child_kwargs(parent, {"provider": "custom:child"})
    assert kwargs["provider"] == "custom:child"
    assert kwargs["base_url"] == "https://child.example.test/v1"
    assert kwargs["api_key"] == "child-owned-key"


@pytest.mark.parametrize("provider,url", [("custom", "https://child.example.test/v1"),
                                           ("copilot", "https://api.githubcopilot.com")])
def test_direct_runtime_override_without_key_is_keyless(parent, provider, url):
    kwargs = _resolve_child_runtime(
        parent, {}, parent.api_key, model=None, override_provider=provider,
        override_base_url=url, override_api_key=None, override_api_mode=None,
        override_acp_command=None, override_acp_args=None,
    )
    assert kwargs["api_key"] == "no-key"


@pytest.mark.parametrize("url", ["https://child.example.test/v1", "https://api.githubcopilot.com"])
def test_explicit_route_survives_child_pool_lease(parent, monkeypatch, url):
    child, pool = build_leased_child(parent, monkeypatch, {
        "provider": "copilot", "base_url": url, "api_key": "child-key",
    })
    assert (child.provider, child.base_url, child.api_key) == ("copilot", url, "child-key")
    assert not pool._active_leases


def test_unpinned_child_leases_another_parent_credential(parent, monkeypatch):
    child, pool = build_leased_child(parent, monkeypatch, {})
    assert child._credential_pool is pool
    assert child.api_key == "rotated-parent-key"
    assert child._credential_pool_entry_id == "rotation-row"
    assert not pool._active_leases


def test_named_custom_dynamic_key_survives_pool_resolution(parent, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "providers:\n  child:\n    base_url: https://child.example.test/v1\n"
        "    key_cmd: test-token-command\n    default_model: child-model\n", encoding="utf-8")
    child, pool = build_leased_child(parent, monkeypatch, {"provider": "custom:child"})
    assert child.provider == "custom:child"
    assert child.base_url == "https://child.example.test/v1"
    assert callable(child.api_key)
    assert child._client_kwargs["api_key"] is child.api_key
    assert not pool._active_leases


def build_leased_child(parent, monkeypatch, cfg):
    from types import MethodType
    from agent.client_lifecycle import ClientLifecycleMixin
    from agent.credential_pool import CredentialPool, PooledCredential
    from tools import delegate_tool as dt
    from tools.delegate_tool_child_run import _lease_child_credential
    import run_agent

    pool = CredentialPool("copilot", [PooledCredential(
        provider="copilot", id="parent-row", label="parent", auth_type="api_key",
        priority=0, source="manual", access_token="parent-secret", base_url=parent.base_url,
    ), PooledCredential(
        provider="copilot", id="rotation-row", label="rotation", auth_type="api_key",
        priority=1, source="manual", access_token="rotated-parent-key", base_url=parent.base_url,
    )])
    parent._credential_pool = pool
    parent_lease = pool.acquire_lease("parent-row")

    def construct(**kwargs):
        child = SimpleNamespace(**kwargs)
        child._client_kwargs = {"api_key": child.api_key, "base_url": child.base_url}
        child._swap_credential = MethodType(ClientLifecycleMixin._swap_credential, child)
        child._reapply_route_client_config = lambda **kw: None
        child._replace_primary_openai_client = lambda **kw: None
        return child

    monkeypatch.setattr(run_agent, "AIAgent", construct)
    monkeypatch.setattr(dt, "_load_config", lambda: cfg)
    monkeypatch.setattr(dt, "_resolve_child_toolsets", lambda *args: ([], []))
    monkeypatch.setattr(dt, "_build_child_system_prompt", lambda *args, **kwargs: "test")
    monkeypatch.setattr(dt, "_resolve_workspace_hint", lambda *args: None)
    monkeypatch.setattr(dt, "_build_child_progress_callback", lambda *args, **kwargs: None)
    monkeypatch.setattr(dt, "_open_child_session_db", lambda *args: None)
    monkeypatch.setattr(dt, "_attach_child", lambda *args: None)
    bundle = _resolve_delegation_credentials(cfg, parent)
    child = dt._build_child_agent(
        0, "test", None, [], bundle["model"], 1, 1, parent,
        override_provider=bundle["provider"], override_base_url=bundle["base_url"],
        override_api_key=bundle["api_key"], override_api_mode=bundle["api_mode"],
    )
    leased_pool, lease_id = _lease_child_credential(child)
    if lease_id is not None:
        assert leased_pool._active_leases[lease_id] == 1
        leased_pool.release_lease(lease_id)
    assert pool._active_leases == {"parent-row": 1}
    pool.release_lease(parent_lease)
    return child, pool
