"""Provider-neutral callable bearer credentials for custom HTTP transports."""
from __future__ import annotations
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)
_AUTH_HEADERS = ("Authorization", "authorization", "Api-Key", "api-key", "X-Api-Key", "x-api-key")

def is_token_provider(value: Any) -> bool:
    """True when ``value`` is a callable token provider (vs. a string API key)."""
    return callable(value) and not isinstance(value, str)


def materialize_bearer_for_http(value: Any) -> str:
    """Mint a fresh Bearer JWT for a manual HTTP request (calls the provider once). Only for sites building
    ``Authorization`` outside the OpenAI SDK; the Anthropic SDK can't take a callable, so
    :func:`build_bearer_http_client` calls this from an httpx hook. ``ValueError`` on an unusable value/empty token."""
    if is_token_provider(value):
        token = value()
        if not isinstance(token, str) or not token:
            raise ValueError("token provider returned empty value")
        return token
    if isinstance(value, str) and value:
        return value
    raise ValueError("no usable api_key / token provider")


def _strip_auth_headers(request: Any) -> None:
    for header_name in _AUTH_HEADERS:
        request.headers.pop(header_name, None)


def build_bearer_http_client(token_provider: Callable[[], str], **httpx_kwargs: Any) -> Any:
    """``httpx.Client`` minting a fresh bearer token per outbound request. The Anthropic SDK computes
    ``Authorization`` once at construction, so per-request refresh needs a ``request`` hook: mint (cheap —
    the provider caches), strip pre-set auth headers, set ``Authorization: Bearer``. ``httpx_kwargs`` are
    forwarded verbatim (``timeout``, ``transport``...)."""
    if not is_token_provider(token_provider):
        raise ValueError("build_bearer_http_client requires a zero-arg callable token provider")
    import httpx

    def _inject_bearer(request: "httpx.Request") -> None:
        try:
            token = materialize_bearer_for_http(token_provider)
        except ValueError as exc:
            # Provider exhausted or expired: strip ALL auth headers (incl. the anthropic_adapter placeholder
            # sentinel) so the endpoint returns a clean "missing auth" 401 and the sentinel never reaches upstream logs.
            # WARNING so the misconfiguration is visible at default levels.
            logger.warning("Bearer hook: token provider returned empty (%s) "
                           "— stripping Authorization headers. The endpoint will reject unauthenticated requests. "
                           "Check the configured token provider to recover.", exc)
            _strip_auth_headers(request)
            return
        _strip_auth_headers(request)
        request.headers["Authorization"] = f"Bearer {token}"

    return httpx.Client(event_hooks={"request": [_inject_bearer]}, **httpx_kwargs)
