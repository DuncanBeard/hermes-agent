"""Inference boundaries reject retired routes even with explicit credentials."""
from types import SimpleNamespace
import pytest

def test_retained_entrypoints_import_without_external_runtime_bridges():
    import importlib
    for module in ('gateway.run', 'tui_gateway.server', 'acp_adapter.entry',
                   'agent.turn_context_compaction', 'agent.interrupt_control'):
        importlib.import_module(module)


RETIRED = ('openrouter', 'anthropic', 'openai-codex', 'copilot-acp', 'bedrock', 'vertex', 'gemini', 'moa')

@pytest.mark.parametrize('mode', ['bedrock_converse', 'codex_app_server'])
def test_retired_wire_mode_is_not_registered(mode):
    from agent.transports import get_transport
    assert get_transport(mode) is None

@pytest.mark.parametrize('module', ['agent.copilot_acp_client', 'agent.acp_openai_bridge'])
def test_retired_external_client_is_not_importable(module):
    from importlib.util import find_spec
    assert find_spec(module) is None

@pytest.mark.parametrize('provider', RETIRED)
def test_agent_rejects_retired_provider_before_build(provider):
    from run_agent import AIAgent
    with pytest.raises(ValueError, match='(?i)(unsupported|retired)'):
        AIAgent(provider=provider, model='test',
                   api_key='test-only-key', base_url='http://localhost:9999/v1')

@pytest.mark.parametrize('provider', RETIRED)
def test_task_override_cannot_launder_retired_provider_through_endpoint(provider):
    from agent.auxiliary_client import _resolve_task_provider_model
    with pytest.raises(ValueError, match='(?i)(unsupported|retired)'):
        _resolve_task_provider_model(provider=provider, model='test',
                                     base_url='http://localhost:9999/v1', api_key='test-key')

@pytest.mark.parametrize('provider', RETIRED)
def test_retired_fallback_is_not_silently_skipped(provider):
    from agent.agent_init import _fallback_entries
    with pytest.raises(ValueError, match='Unsupported model provider'):
        _fallback_entries([{'provider': provider, 'model': 'test'},
                           {'provider': 'custom', 'model': 'test'}])

@pytest.mark.parametrize('provider', RETIRED)
def test_auxiliary_rejects_retired_provider_before_build(provider):
    from agent.auxiliary_client import resolve_provider_client
    with pytest.raises(ValueError, match='(?i)(unsupported|retired)'):
        resolve_provider_client(provider, model='test', explicit_api_key='test-only-key',
                                explicit_base_url='http://localhost:9999/v1')
