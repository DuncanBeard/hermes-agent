"""The supported account boundary is shared by every model entry point."""
import importlib
import pytest


def test_supported_provider_contract():
    policy = importlib.import_module("hermes_cli.provider_policy")
    assert policy.SUPPORTED_BUILTIN_PROVIDERS == frozenset({"copilot", "nous", "custom"})
    for raw, expected in [(" Copilot ", "copilot"), ("NOUS", "nous"), ("custom", "custom"), ("CUSTOM:Lab", "custom:lab")]:
        assert policy.is_supported_provider_id(raw)
        assert policy.require_supported_provider(raw) == expected
    for value in (None, "", "auto", "custom:", "openrouter", "openai", "anthropic", "copilot-acp", "moa", "lab"):
        assert not policy.is_supported_provider_id(value)
        with pytest.raises(policy.UnsupportedProviderError, match="Copilot, Nous, or a custom endpoint"):
            policy.require_supported_provider(value)
