"""Retired implementations must not remain importable."""
from importlib.util import find_spec
import pytest

@pytest.mark.parametrize("module", ["bedrock_adapter", "vertex_adapter", "gemini_native_adapter", "azure_identity_adapter", "moa_loop", "moa_trace"])
def test_retired_adapter_not_importable(module):
    assert find_spec(f"agent.{module}") is None
