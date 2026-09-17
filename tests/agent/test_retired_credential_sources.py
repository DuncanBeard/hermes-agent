"""Retired inference credentials cannot be seeded or refreshed."""
from unittest.mock import patch
import pytest

@pytest.mark.parametrize('provider', ['anthropic', 'openrouter', 'qwen-oauth', 'gemini', 'bedrock', 'vertex', 'moa'])
def test_retired_pool_is_empty_without_reading_credentials(provider):
    from agent.credential_pool import load_pool
    with patch('agent.credential_pool.read_credential_pool', side_effect=AssertionError('credential read')):
        assert load_pool(provider).entries() == []
