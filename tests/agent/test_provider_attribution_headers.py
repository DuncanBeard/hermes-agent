"""Retained route headers and destination-scoped user WAF overrides.

Attribution tests for retired provider accounts are intentionally removed;
shared SDK-header preservation and custom endpoint overrides remain covered.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


@patch("agent.process_bootstrap.OpenAI")
def test_routed_client_preserves_openai_sdk_custom_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    routed_client = SimpleNamespace(
        api_key="test-key",
        base_url="https://inference-api.nousresearch.com/v1",
        _custom_headers={"X-Endpoint-Tag": "retained-route"},
    )
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(
        routed_client, "test-model",
    )):
        agent = AIAgent(
            provider="nous",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    headers = agent._client_kwargs["default_headers"]
    assert headers["X-Endpoint-Tag"] == "retained-route"


@patch("agent.process_bootstrap.OpenAI")
def test_user_default_headers_override_sdk_user_agent(mock_openai):
    """The configured endpoint can replace SDK headers rejected by its WAF."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        model="my-custom-model",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    config = {"model": {
        "base_url": "http://localhost:8080/v1",
        "default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"},
    }}
    with patch("hermes_cli.config.load_config", return_value=config), patch(
        "hermes_cli.config.load_config_readonly", return_value=config,
    ):
        agent._apply_client_headers_for_base_url("http://localhost:8080/v1")
    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"] == "curl/8.7.1"
    assert headers["X-Extra"] == "1"
