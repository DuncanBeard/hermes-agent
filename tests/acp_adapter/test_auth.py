"""Tests for acp_adapter.auth — provider detection."""

from acp_adapter.auth import (
    TERMINAL_SETUP_AUTH_METHOD_ID,
    build_auth_methods,
    detect_provider,
)


class TestDetectProviderPresence:
    def test_has_provider_with_resolved_runtime(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "copilot", "api_key": "fixture-copilot-key"},
        )
        assert detect_provider() is not None

    def test_has_provider_false_without_credentials(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "copilot", "api_key": ""},
        )
        assert detect_provider() is None


class TestDetectProvider:
    def test_detect_copilot(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "copilot", "api_key": "fixture-copilot-key"},
        )
        assert detect_provider() == "copilot"


class TestBuildAuthMethods:
    def test_build_auth_methods_returns_provider_and_terminal_when_configured(self, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "copilot")

        methods = build_auth_methods()
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in methods]

        assert payloads[0]["id"] == "copilot"
        assert payloads[0]["name"] == "copilot runtime credentials"
        assert any(payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID for payload in payloads)
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]

