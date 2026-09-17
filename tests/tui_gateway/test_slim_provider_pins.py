"""RPC session-resume and config synchronization retain explicit provider refusals."""
from types import SimpleNamespace
import pytest
import tui_gateway.server as server

@pytest.mark.parametrize('provider', ['openrouter', 'copilot-acp', 'moa'])
def test_resumed_retired_provider_is_not_replaced_by_global_default(provider):
    with pytest.raises(ValueError, match='Unsupported model provider'):
        server._resolve_agent_model_runtime({'model': 'test', 'provider': provider,
            'base_url': 'http://localhost:9999/v1', 'api_key': 'test-key'}, None)


def test_retired_config_sync_stops_before_continuing_old_agent(monkeypatch):
    monkeypatch.setattr(server, '_config_model_target', lambda: ('test', 'openrouter'))
    session = {'agent': SimpleNamespace(provider='copilot', model='gpt-5.4')}
    with pytest.raises(ValueError, match='Unsupported model provider'):
        server._sync_agent_model_with_config('test', session)
    assert 'config_model_seen' not in session
