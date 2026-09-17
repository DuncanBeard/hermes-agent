"""Primary restoration exercises real builders, replacing only SDK construction."""
from functools import partial
from types import SimpleNamespace

import pytest

from agent import agent_runtime_helpers as helpers


ROUTES = [
    ("copilot", "chat_completions"),
    ("copilot", "codex_responses"),
    ("nous", "chat_completions"),
    ("nous", "anthropic_messages"),
    ("custom", "chat_completions"),
    ("custom", "codex_responses"),
    ("custom", "anthropic_messages"),
]


@pytest.fixture
def sdk(monkeypatch):
    from agent import process_bootstrap, anthropic_adapter
    built = []

    def construct(**kwargs):
        client = SimpleNamespace(**kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(process_bootstrap, "OpenAI", construct)
    monkeypatch.setattr(anthropic_adapter, "_require_sdk", lambda *a: SimpleNamespace(Anthropic=construct, Omit=object))
    return built


def fallback_agent(provider, mode):
    url = "https://primary.example.test" + ("" if mode == "anthropic_messages" else "/v1")
    rt = dict(model="primary-model", provider=provider, requested_provider=provider,
              base_url=url, api_mode=mode, api_key="primary-key",
              client_kwargs={"base_url": url, "api_key": "primary-key"},
              anthropic_api_key="primary-key", anthropic_base_url=url,
              is_anthropic_oauth=False, use_prompt_caching=True,
              runtime_capabilities={"primary": True}, request_overrides={"primary": True},
              compressor_model="primary-model", compressor_context_length=100000,
              compressor_base_url=url, compressor_api_key="primary-key", compressor_provider=provider)
    updates = []
    agent = SimpleNamespace(
        _primary_runtime=rt, model="fallback-model", provider="custom",
        requested_provider="custom", base_url="https://fallback.example.test/v1",
        api_mode="chat_completions", api_key="fallback-key", client=object(),
        _anthropic_client=None, _anthropic_api_key="fallback-key", _anthropic_base_url=None,
        _is_anthropic_oauth=False, _reasoning_echo_flag=False,
        _client_kwargs={"api_key": "fallback-key"}, request_overrides={"fallback": True},
        runtime_capabilities={"fallback": True}, _transport_cache={"fallback": object()},
        _use_prompt_caching=False, _use_native_cache_layout=False,
        _fallback_activated=True, _fallback_index=1, _credential_pool=None,
        context_compressor=SimpleNamespace(update_model=lambda **kw: updates.append(kw)),
        _build_keepalive_http_client=lambda *a, **kw: None,
        _client_log_context=lambda: "test", _cached_system_prompt="Model: fallback-model\nProvider: custom",
    )
    agent._create_openai_client = partial(helpers.create_openai_client, agent)
    return agent, updates


@pytest.mark.parametrize("provider,mode", ROUTES)
def test_next_turn_restores_primary_client_and_route(provider, mode, sdk):
    agent, updates = fallback_agent(provider, mode)
    old_client = agent.client
    assert helpers.restore_primary_runtime(agent) is True
    assert (agent.provider, agent.model, agent.api_mode) == (provider, "primary-model", mode)
    assert agent.api_key == "primary-key"
    client = agent._anthropic_client if mode == "anthropic_messages" else agent.client
    assert client is not None and client is not old_client and client is sdk[-1]
    assert client.base_url == agent.base_url
    assert getattr(client, "api_key", getattr(client, "auth_token", None)) == "primary-key"
    assert agent._fallback_activated is False
    assert agent._fallback_index == 0
    assert agent.request_overrides == {"primary": True}
    assert agent.runtime_capabilities == {"primary": True}
    assert updates[-1]["model"] == agent.model
    assert agent._cached_system_prompt == f"Model: primary-model\nProvider: {provider}"


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses", "anthropic_messages"])
def test_failed_primary_build_keeps_fallback_route_atomic(mode, sdk, monkeypatch):
    from agent import process_bootstrap, anthropic_adapter
    agent, updates = fallback_agent("custom", mode)
    before = dict(vars(agent))
    before["_transport_cache"] = dict(agent._transport_cache)

    def fail(**kwargs):
        raise RuntimeError("SDK construction failed")

    monkeypatch.setattr(process_bootstrap, "OpenAI", fail)
    monkeypatch.setattr(anthropic_adapter, "_require_sdk", lambda *a: SimpleNamespace(Anthropic=fail, Omit=object))
    assert helpers.restore_primary_runtime(agent) is False
    for key in ("model", "provider", "requested_provider", "base_url", "api_mode", "api_key",
                "client", "_anthropic_client", "_anthropic_api_key", "_anthropic_base_url",
                "request_overrides", "runtime_capabilities", "_client_kwargs", "_transport_cache",
                "_use_prompt_caching", "_use_native_cache_layout", "_fallback_activated", "_fallback_index"):
        assert getattr(agent, key) == before[key], key
    assert updates == []


@pytest.mark.parametrize("provider,mode", [route for route in ROUTES if route != ("nous", "chat_completions")])
def test_transient_primary_recovery_uses_same_builder_signature(provider, mode, sdk, monkeypatch):
    import httpx
    agent, _ = fallback_agent(provider, mode)
    agent._fallback_activated = False
    agent.provider = provider
    agent.api_mode = mode
    agent._is_openrouter_url = lambda: False
    retired = []
    agent._retire_shared_openai_client = lambda client, **kw: retired.append(client)
    agent._vprint = lambda *a, **kw: None
    agent.log_prefix = ""
    old_client = agent.client
    monkeypatch.setattr(helpers.time, "sleep", lambda _: None)
    assert helpers.try_recover_primary_transport(agent, httpx.ReadTimeout("test"),
                                                 retry_count=2, max_retries=2) is True
    client = agent._anthropic_client if mode == "anthropic_messages" else agent.client
    assert client is sdk[-1]
    assert retired == [old_client]


def test_retired_primary_rejected_before_route_mutation(sdk):
    agent, updates = fallback_agent("openrouter", "chat_completions")
    old_client = agent.client
    assert helpers.restore_primary_runtime(agent) is False
    assert agent.model == "fallback-model"
    assert agent.provider == "custom"
    assert agent.client is old_client
    assert agent._fallback_activated
    assert sdk == [] and updates == []
