"""Generic bearer transport regressions; no hosted identity SDK required."""
from __future__ import annotations
import pytest

class TestMaterializeBearerForHttp:
    """The only helper that mints a real bearer JWT — must call the
    callable exactly once and never fall through to display masking."""

    def test_callable_is_invoked_and_returns_token(self):
        from agent.bearer_auth import materialize_bearer_for_http

        invoked = {"count": 0}

        def provider():
            invoked["count"] += 1
            return "fresh-jwt"

        assert materialize_bearer_for_http(provider) == "fresh-jwt"
        assert invoked["count"] == 1



    def test_empty_string_raises(self):
        from agent.bearer_auth import materialize_bearer_for_http
        with pytest.raises(ValueError):
            materialize_bearer_for_http("")
        with pytest.raises(ValueError):
            materialize_bearer_for_http(None)


# ---------------------------------------------------------------------------
# build_bearer_http_client — the Anthropic-on-Foundry bridge
# ---------------------------------------------------------------------------


class TestBuildBearerHttpClient:
    """``build_bearer_http_client`` returns an ``httpx.Client`` whose
    request event hook mints a fresh JWT per outbound request. This is
    how Entra ID auth reaches the Anthropic SDK (which does not accept
    callable ``auth_token``)."""


    def test_hook_overrides_authorization_header(self):
        import httpx
        from agent.bearer_auth import build_bearer_http_client

        minted_tokens = []

        def provider():
            minted_tokens.append(f"jwt-{len(minted_tokens) + 1}")
            return minted_tokens[-1]

        client = build_bearer_http_client(provider)
        try:
            hook = client.event_hooks["request"][0]
            # Build a request with conflicting pre-set headers and verify
            # the hook strips them and installs the fresh bearer.
            req = httpx.Request(
                "POST", "https://example.com/v1/messages",
                headers={
                    "Authorization": "Bearer stale-token",
                    "api-key": "static-key",
                    "x-api-key": "static-key",
                },
                json={"hello": "world"},
            )
            hook(req)
            assert req.headers["Authorization"] == "Bearer jwt-1"
            # The static-key headers must be stripped — sending both
            # auth values would be ambiguous on Azure.
            assert "api-key" not in req.headers
            assert "x-api-key" not in req.headers

            # Second invocation mints a fresh token.
            req2 = httpx.Request("GET", "https://example.com/v1/models")
            hook(req2)
            assert req2.headers["Authorization"] == "Bearer jwt-2"
            assert len(minted_tokens) == 2
        finally:
            client.close()

    def test_hook_strips_auth_headers_and_warns_when_token_provider_fails(self, caplog):
        """When the token provider fails (chain exhausted, IMDS down, az
        login expired), the hook must:
          1. Log at WARNING level so the misconfiguration is visible at
             default log level (not buried at DEBUG).
          2. Strip any pre-set Authorization headers — including the
             placeholder ``entra-id-bearer-via-http-hook`` sentinel that
             :func:`_build_anthropic_client_with_bearer_hook` sets on the
             Anthropic SDK constructor. This produces a clean
             "missing auth" 401 from Azure rather than a sentinel-bearing
             401 that's harder to diagnose AND avoids leaking the
             sentinel string into upstream access logs.
        """
        import logging
        import httpx
        from agent.bearer_auth import build_bearer_http_client

        def bad_provider():
            return ""  # empty token → materialize_bearer_for_http raises

        client = build_bearer_http_client(bad_provider)
        try:
            hook = client.event_hooks["request"][0]
            req = httpx.Request(
                "POST", "https://example.com/v1/messages",
                headers={
                    "Authorization": "Bearer entra-id-bearer-via-http-hook",
                    "api-key": "leaked-placeholder",
                },
            )
            with caplog.at_level(logging.WARNING, logger="agent.bearer_auth"):
                hook(req)  # Must not raise.
            # Pre-set auth headers stripped — no sentinel makes it to Azure.
            assert "Authorization" not in req.headers
            assert "api-key" not in req.headers
            # WARNING was logged so the user sees the misconfiguration.
            assert any(
                rec.levelno == logging.WARNING and "token provider" in rec.message
                for rec in caplog.records
            )
        finally:
            client.close()




class TestIsTokenProvider:
    def test_callable_is_token_provider(self):
        from agent.bearer_auth import is_token_provider
        assert is_token_provider(lambda: "x") is True

    def test_string_is_not_token_provider(self):
        from agent.bearer_auth import is_token_provider
        assert is_token_provider("static-key") is False
        # ``str`` instances are technically callable in some edge cases
        # — confirm they're never classified as token providers.
        assert is_token_provider("") is False
