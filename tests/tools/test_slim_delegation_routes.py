"""Delegation cannot revive a retired provider through an explicit endpoint or command."""
from types import SimpleNamespace
import pytest
from tools.delegate_tool_config import _resolve_delegation_credentials

@pytest.mark.parametrize('provider', ['openrouter', 'anthropic', 'copilot-acp', 'moa'])
def test_direct_endpoint_does_not_bypass_provider_policy(provider):
    with pytest.raises(ValueError, match='(?i)(unsupported|retired)'):
        _resolve_delegation_credentials({'provider': provider, 'base_url': 'http://localhost:9999/v1',
                                         'api_key': 'test-key'}, SimpleNamespace())

def test_external_command_does_not_override_direct_provider():
    with pytest.raises(ValueError, match='(?i)(unsupported|retired)'):
        _resolve_delegation_credentials({'provider': 'custom', 'base_url': 'http://localhost:9999/v1',
                                         'command': 'python'}, SimpleNamespace())
