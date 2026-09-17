"""Rejected model identities must not probe networks or read vendor credentials."""
from unittest.mock import patch

import pytest

from hermes_cli.provider_policy import UnsupportedProviderError


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "openai-codex", "lmstudio", "ollama", "bedrock", "moa", "minimax", "qwen-oauth"])
def test_validation_rejects_before_any_catalog_probe(provider):
    from hermes_cli.models_validate import validate_requested_model

    with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")), patch("hermes_cli.models.probe_api_models", side_effect=AssertionError("unexpected probe")):
        with pytest.raises(UnsupportedProviderError, match="Unsupported model provider"):
            validate_requested_model("example-model", provider, base_url="https://example.invalid/v1", api_key="test-key")


def test_custom_catalog_never_adopts_vendor_api_keys(monkeypatch):
    from hermes_cli import models

    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-service-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unrelated-service-key")
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    with patch.object(models, "_get_custom_base_url", return_value="https://example.invalid/v1"), patch.object(models, "_get_model_config_dict", return_value={}), patch.object(models, "fetch_api_models", return_value=["example"]) as fetch:
        assert models._custom_catalog("custom", False) == ["example"]
        assert fetch.call_args.args[0] == ""
