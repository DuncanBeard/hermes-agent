"""Shared runtime provider resolution for CLI, gateway, cron, and helpers: the resolution ORDER
(:func:`resolve_runtime_provider`), api_mode / base_url helpers and the pool / OAuth / explicit paths.
Custom-provider lookup lives in :mod:`hermes_cli.runtime_provider_custom`; Azure Foundry,
OpenRouter/bare-custom, Bedrock and external-process builders in
:mod:`hermes_cli.runtime_provider_backends` — both re-exported here so
``hermes_cli.runtime_provider.<name>`` imports and test patches keep working."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from hermes_cli import auth as auth_mod
from agent.credential_pool import (  # custom_provider_pool_key_candidates is read via origin by runtime_provider_custom
    CredentialPool, PooledCredential, credential_pool_matches_provider, custom_provider_pool_key_candidates,  # noqa: F401
    load_pool,
)
from agent.secret_scope import get_secret_str
from hermes_cli.auth import (AuthError, PROVIDER_REGISTRY, _agent_key_is_usable, _nous_inference_env_override, format_auth_error, resolve_provider, resolve_nous_runtime_credentials, resolve_api_key_provider_credentials, has_usable_secret)
from hermes_cli import config as _config_mod
from hermes_cli import models as _models  # attribute access keeps ``hermes_cli.models.<name>`` patches effective

from hermes_cli.providers import determine_api_mode, is_actual_route, is_official_openai_host, nous_api_mode
from utils import base_url_host_matches, base_url_hostname, env_int


# Late-bound delegates, deliberately NOT module-level from-imports: this module is often imported
# lazily, so its first import can happen while a test has ``hermes_cli.config.load_config`` patched
# — a from-import would bind the MagicMock permanently and poison every later caller.
def load_config():
    return _config_mod.load_config()


def get_compatible_custom_providers(config=None):
    return _config_mod.get_compatible_custom_providers(config)


def normalize_extra_headers(value):
    return _config_mod.normalize_extra_headers(value)


def _loopback_hostname(host: str) -> bool:
    return (host or "").lower().rstrip(".") in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _resolves_to_custom(name: str) -> bool:
    """True when a provider alias (ollama, vllm, llamacpp, …) resolves to ``custom``."""
    try:
        return auth_mod.resolve_provider(name) == "custom"
    except Exception:
        return False


def _config_base_url_trustworthy_for_bare_custom(cfg_base_url: str, cfg_provider: str) -> bool:
    """Whether ``model.base_url`` may back bare ``custom`` runtime resolution. The picker can select
    Custom while ``model.provider`` still names a previous provider, so non-loopback URLs are rejected
    unless the YAML provider is already ``custom`` or a local-server alias (ollama/vllm/llamacpp —
    else a legit LAN ollama endpoint falls through to OpenRouter): a stale OpenRouter/Z.ai base_url
    cannot hijack local sessions.

    See #14676.
    """
    cfg_provider_norm = (cfg_provider or "").strip().lower()
    bu = (cfg_base_url or "").strip()
    # A bare or ``auto`` provider is the caller currently resolving auto. Asking
    # ``resolve_provider`` whether it aliases custom re-enters that same path.
    return bool(bu) and (cfg_provider_norm == "custom" or (
        cfg_provider_norm not in {"", "auto"} and _resolves_to_custom(cfg_provider_norm)
    )
                         or (not base_url_host_matches(bu, "openrouter.ai") and _loopback_hostname(base_url_hostname(bu))))


# ── api_mode detection ─────────────────────────────────────────────────────────────────────

# Hosts that only speak one wire protocol. Mirrors host_mandated_api_mode in hermes_cli/providers.py
# so the runtime resolver stays in lockstep: api.meta.ai — prompt caching only on Responses;
# api.router.com — /v1/chat/completions is a minimal shim; api.anthropic.com — native Messages.
_HOST_MANDATED_API_MODES = {
    "api.x.ai": "codex_responses", "api.meta.ai": "codex_responses", "api.actual.inc": "chat_completions",
    "api.router.com": "codex_responses", "api.anthropic.com": "anthropic_messages",
}

# codex_app_server is opt-in: hand the whole turn to a `codex app-server` subprocess (Codex's own
# tool runtime), gated on `model.openai_runtime == "codex_app_server"` AND provider in {openai, openai-codex}.
_VALID_API_MODES = {"chat_completions", "codex_responses", "anthropic_messages"}


def _detect_api_mode_for_url(base_url: str) -> Optional[str]:
    """Auto-detect api_mode from the resolved base URL, or None. Exact-hostname matches reject
    lookalike subdomains (api.anthropic.com.attacker.test) and path-segment spoofing
    (proxy.test/api.anthropic.com/v1). Official OpenAI hosts (incl. us./eu. data-residency hosts)
    need Responses for GPT-5.x tool calls with reasoning.

    - Direct api.anthropic.com endpoints must use the native Messages API (``/v1/messages``). Anthropic also
    exposes an OpenAI-compat ``/chat/completions`` shim on the same host, but Pro/Max OAuth subscriptions
    are only billed against the native Messages route; hitting the shim accounts against a separate "extra
    usage" pool that is empty by default and surfaces as HTTP 400 "You're out of extra usage."  See issue
    #32243. - Third-party Anthropic-compatible gateways (MiniMax, Zhipu GLM, LiteLLM proxies, etc.)
    conventionally expose the native Anthropic protocol under a ``/anthropic`` suffix — treat those as
    ``anthropic_messages`` transport instead of the default ``chat_completions``. - Kimi Code's
    ``api.kimi.com/coding`` endpoint also speaks the Anthropic Messages protocol (the /coding route accepts
    Claude Code's native request shape).
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    hostname = base_url_hostname(base_url)
    mandated = _HOST_MANDATED_API_MODES.get(hostname) or ("codex_responses" if is_official_openai_host(base_url) else None)
    if mandated:
        return mandated
    path = urlparse(normalized).path.rstrip("/")
    if path.endswith(("/anthropic", "/anthropic/v1")) or (hostname == "api.kimi.com" and "/coding" in normalized):
        # Direct native Anthropic host: realign with providers.determine_api_mode, which already maps this
        # host to anthropic_messages. The exact-hostname match rejects lookalike subdomains
        # (api.anthropic.com.attacker.test) and path-segment spoofing (proxy.test/api.anthropic.com/v1).
        # (#32243)
        return "anthropic_messages"
    return None


def _parse_api_mode(raw: Any) -> Optional[str]:
    """Validate an api_mode from config (None if invalid). Legacy/alias spellings (``openai``,
    ``anthropic``, ``responses``, …) are canonicalized first so old configs keep their transport
    instead of silently falling through to hostname-based detection."""
    normalized = _config_mod._canonical_api_mode(raw).lower() if isinstance(raw, str) else ""
    return normalized if normalized in _VALID_API_MODES else None


def _fallback_api_mode(provider: str, base_url: str, model: str = "") -> str:
    """api_mode when no explicit/persisted mode applies: URL detection (host-mandated wire shapes)
    first, then the transport the provider overlay declares via ``providers.determine_api_mode``
    (``openai-api`` pointed at us.api.openai.com 400'd on every tool call without it), then
    ``chat_completions``."""
    if is_actual_route(provider, base_url):
        return "chat_completions"
    return _detect_api_mode_for_url(base_url) or determine_api_mode(provider, base_url, model) or "chat_completions"


def _resolve_plain_custom_api_mode(model_cfg: Dict[str, Any], base_url: str) -> str:
    return _parse_api_mode(model_cfg.get("api_mode")) or _detect_api_mode_for_url(base_url) or "chat_completions"


def _provider_supports_explicit_api_mode(provider: Optional[str], configured_provider: Optional[str] = None) -> bool:
    """Whether a persisted api_mode may be honored for ``provider`` — only when the config's
    provider matches (or none is recorded), so a stale mode never leaks across a switch."""
    p, c = (provider or "").strip().lower(), (configured_provider or "").strip().lower()
    return not c or (c == "custom" or c.startswith("custom:") if p == "custom" else c == p)


def _configured_api_mode(provider: str, model_cfg: Dict[str, Any]) -> Optional[str]:
    """Persisted ``model.api_mode`` when valid and recorded for this provider, else None."""
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    return configured_mode if configured_mode and _provider_supports_explicit_api_mode(provider, _cfg_provider(model_cfg)) else None


def _effective_model(model_cfg: Dict[str, Any], target_model: Optional[str]) -> str:
    """The caller's target model (e.g. /model switch) beats the persisted default, else api_mode
    is computed from a stale default."""
    return target_model or model_cfg.get("default") or ""


def _copilot_runtime_api_mode(model_cfg: Dict[str, Any], api_key: str, *, target_model: Optional[str] = None) -> str:
    configured_mode = _configured_api_mode("copilot", model_cfg)
    if configured_mode:
        return configured_mode
    # Use the model being resolved, not the persisted default: a Claude MoA slot inheriting
    # codex_responses from a GPT-5 default fails with "model ... does not support Responses API".
    model_name = str(_effective_model(model_cfg, target_model)).strip()
    try:
        return _models.copilot_model_api_mode(model_name, api_key=api_key) if model_name else "chat_completions"
    except Exception:
        return "chat_completions"












# ── base_url / credential helpers ──────────────────────────────────────────────────────────





def _runtime(provider: str, api_mode: str, base_url: Any, api_key: Any, **extra: Any) -> Dict[str, Any]:
    from hermes_cli.provider_policy import require_supported_provider
    provider = require_supported_provider(provider)
    if provider == "custom" or provider.startswith("custom:"):
        from hermes_cli.runtime_provider_backends import validate_custom_endpoint
        base_url = validate_custom_endpoint(base_url)
    return {"provider": provider, "api_mode": api_mode, "base_url": base_url, "api_key": api_key, **extra}


def _cfg_provider(model_cfg: Dict[str, Any]) -> str:
    return str(model_cfg.get("provider") or "").strip().lower()


def _config_base_url_for_provider(model_cfg: Dict[str, Any], provider: str) -> str:
    return str(model_cfg.get("base_url") or "").strip().rstrip("/") if _cfg_provider(model_cfg) == provider else ""











def _host_derived_api_key(base_url: str) -> str:
    """``<VENDOR>_API_KEY`` from the env, vendor = registrable hostname label (``api.deepseek.com``
    → ``deepseek``). Lookalike hosts pick the ATTACKER's label (api.deepseek.com.attacker.test →
    "attacker") so DEEPSEEK_API_KEY stays put. "" for IPs/loopback/single-label hosts and for
    OPENAI/OPENROUTER/OLLAMA, which have their own host-gated paths."""
    hostname = base_url_hostname(base_url)
    if not hostname or any(ch.isdigit() for ch in hostname.split(".")[-1]) or hostname == "localhost" or ":" in hostname:
        return ""
    labels = [lbl for lbl in hostname.split(".") if lbl]
    while labels and labels[0] in ("api", "www"):
        labels.pop(0)
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in labels[-2]).upper() if len(labels) >= 2 else ""
    if not sanitized or not sanitized[0].isalpha() or sanitized in ("OPENAI", "OPENROUTER", "OLLAMA"):
        return ""
    return (get_secret_str(f"{sanitized}_API_KEY", "") or "").strip()


def _host_gated_env_key_candidates(base_url: str, *, ollama: bool) -> list:
    """Env API keys gated on their authoritative hosts, then the host-derived ``<VENDOR>_API_KEY``.
    Sending OPENAI/OPENROUTER/OLLAMA keys to an unrelated endpoint leaks credentials
    (GHSA-76xc-57q6-vm5m); match on HOST, not substring. ``_host_derived_api_key`` skips OLLAMA, so
    callers that want it opt in via ``ollama``."""
    is_openai = base_url_host_matches(base_url, "openai.com") or base_url_host_matches(base_url, "openai.azure.com")
    candidates = [get_secret_str("OLLAMA_API_KEY", "").strip() if base_url_host_matches(base_url, "ollama.com") else ""] if ollama else []
    return candidates + [get_secret_str("OPENAI_API_KEY", "").strip() if is_openai else "",
                         get_secret_str("OPENROUTER_API_KEY", "").strip() if base_url_host_matches(base_url, "openrouter.ai") else "",
                         _host_derived_api_key(base_url)]


def _pool_entry_api_key(entry: Any) -> str:
    return getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")


def _pool_entry_base_url(entry: Any) -> str:
    return getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or ""


def _nous_entry_key_usable(entry: Any, min_ttl: int) -> bool:
    return _agent_key_is_usable({k: getattr(entry, k, None) for k in ("agent_key", "agent_key_expires_at", "scope")}, min_ttl)


def _nous_min_key_ttl() -> int:
    return max(60, env_int("HERMES_NOUS_MIN_KEY_TTL_SECONDS", 1800))


def _resolve_nous_creds() -> Dict[str, Any]:
    return resolve_nous_runtime_credentials(timeout_seconds=float(get_secret_str("HERMES_NOUS_TIMEOUT_SECONDS", "15")))


def _finalize_base_url(provider: str, api_mode: str, base_url: str) -> str:
    return base_url.rstrip("/")


# ── model config ───────────────────────────────────────────────────────────────────────────


def _auto_detect_local_model(base_url: str) -> str:
    """Query a local server for its model name when only one model is loaded."""
    if not base_url:
        return ""
    try:
        import requests
        url = base_url.rstrip("/")
        resp = requests.get((url if url.endswith("/v1") else url + "/v1") + "/models", timeout=(2, 3))
        if resp.ok:
            models = resp.json().get("data", [])
            if len(models) == 1 and models[0].get("id", ""):
                return models[0]["id"]
    except Exception as exc:
        logger.debug("Auto-detect model from %s failed: %s", base_url, exc)
    return ""


def _get_model_config(*, discover: bool = True) -> Dict[str, Any]:
    """``model`` config section with ``model`` accepted as an alias for ``default``, a dict
    ``default`` split into model/provider, and a local single-model server auto-detected."""
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    if not isinstance(model_cfg, dict):
        return {}
    cfg = dict(model_cfg)
    if not cfg.get("default") and cfg.get("model"):
        cfg["default"] = cfg["model"]
    _default = cfg.get("default")
    if isinstance(_default, dict):
        cfg_model, cfg_provider = _config_mod.split_model_config_default(_default)
        cfg_provider = cfg_provider or str(model_cfg.get("provider") or "")
        cfg["default"] = cfg_model
        if cfg_provider and not cfg.get("provider"):
            cfg["provider"] = cfg_provider
        _default = cfg_model
    base_url = (cfg.get("base_url") or "").strip()
    if discover and not str(_default or "").strip() and base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1"):
        validate_provider_pins(model_config=cfg)
        detected = _auto_detect_local_model(base_url)
        if detected:
            cfg["default"] = detected
    return cfg


def validate_provider_pins(requested: Optional[str] = None, *, model_config: Optional[Dict[str, Any]] = None) -> None:
    """Reject explicit unsupported selections before discovery, credentials, or guest bootstrap.

    Resolve only explicit pins: the auto ladder would inspect credentials and an empty setup
    must remain eligible for Nous onboarding. Configured bare custom names remain valid.
    """
    # A concrete caller selection supersedes ambient pins. Bootstrap/discovery (no
    # concrete selection) still validates every saved pin before any account I/O.
    if requested and str(requested).strip().lower() != "auto":
        resolve_provider(str(requested).strip().lower())
        return
    cfg = _get_model_config(discover=False) if model_config is None else model_config
    for pin in (requested, cfg.get("provider"), get_secret_str("HERMES_INFERENCE_PROVIDER", "")):
        normalized = str(pin or "").strip().lower()
        if normalized and normalized != "auto":
            resolve_provider(normalized)


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Provider request from explicit arg, then config, then ``HERMES_INFERENCE_PROVIDER``, else
    "auto". Config beats the env so chat uses the endpoint the user last saved, not a stale
    shell/.env override."""
    if requested and requested.strip():
        return requested.strip().lower()
    cfg_provider = _get_model_config(discover=False).get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()
    return get_secret_str("HERMES_INFERENCE_PROVIDER", "").strip().lower() or "auto"


# ── extracted collaborators (re-exported; see module docstring) ────────────────────────────

from hermes_cli.runtime_provider_custom import (  # noqa: E402,F401
    _apply_custom_provider_extras, _custom_provider_request_overrides, _filter_capabilities, _find_custom_identity,
    _get_named_custom_provider, _lift_common_custom_fields, _lift_extra_headers,
    _lift_model_capabilities, _normalize_base_url_for_match, _normalize_custom_provider_name, _resolve_named_custom_runtime,
    _try_resolve_from_custom_pool, canonical_custom_identity, find_custom_provider_identity,
    find_custom_provider_identity_by_model, has_named_custom_provider, is_routable_provider,
)
from hermes_cli.runtime_provider_backends import _resolve_custom_runtime  # noqa: E402


# ── credential-pool entries ────────────────────────────────────────────────────────────────

# Pool-entry providers whose api_mode is fixed: provider -> (api_mode, default base_url when the
# pool entry carries none). Callables are evaluated lazily (registry lookups). MiniMax OAuth tokens
# are valid only against the Anthropic Messages endpoint, so a stale model.api_mode from a prior
# OpenAI-compatible provider is never honoured for it (it would 404 on /chat/completions).



def _pool_entry_mode_and_url(provider, entry, model_cfg, effective_model, base_url) -> tuple:
    if provider == "nous":
        return nous_api_mode(effective_model), (_nous_inference_env_override() or "") or base_url
    if provider == "copilot":
        mode = _copilot_runtime_api_mode(model_cfg, getattr(entry, "runtime_api_key", ""), target_model=effective_model)
        return mode, base_url or PROVIDER_REGISTRY["copilot"].inference_base_url
    return _configured_api_mode(provider, model_cfg) or _fallback_api_mode(provider, base_url, effective_model), base_url


def _resolve_runtime_from_pool_entry(*, provider: str, entry: PooledCredential, requested_provider: str,
                                     model_cfg: Optional[Dict[str, Any]] = None, pool: Optional[CredentialPool] = None,
                                     target_model: Optional[str] = None) -> Dict[str, Any]:
    model_cfg = model_cfg or _get_model_config()
    api_mode, base_url = _pool_entry_mode_and_url(provider, entry, model_cfg, _effective_model(model_cfg, target_model),
                                                  _pool_entry_base_url(entry).rstrip("/"))
    base_url = _finalize_base_url(provider, api_mode, base_url)
    return _runtime(provider, api_mode, base_url, _pool_entry_api_key(entry), source=getattr(entry, "source", "pool"),
                    credential_pool=pool, requested_provider=requested_provider)





def _refresh_nous_pool_entry(pool: CredentialPool, entry: Any, pool_api_key: str):
    """Nous pool entries carry the agent_key (an invoke JWT) which the pool does not refresh on
    selection (avoids network calls in `hermes auth list`); refresh here before falling back to
    singleton auth resolution. Returns (entry, pool_api_key) — key "" when still unusable."""
    min_ttl = _nous_min_key_ttl()
    if _nous_entry_key_usable(entry, min_ttl):
        return entry, pool_api_key
    logger.debug("Nous pool entry agent_key expired/missing, refreshing selected pool entry")
    try:
        refreshed = pool.try_refresh_current()
    except Exception as exc:
        logger.debug("Nous pool entry refresh failed: %s", exc)
        refreshed = None
    if refreshed is not None:
        entry, pool_api_key = refreshed, _pool_entry_api_key(refreshed)
    if not pool_api_key or not _nous_entry_key_usable(entry, min_ttl):
        logger.debug("Nous pool entry agent_key still unavailable, falling through to runtime resolution")
        pool_api_key = ""
    return entry, pool_api_key


def _resolve_from_pool(provider: str, requested_provider: str, model_cfg: Dict[str, Any], explicit_api_key, explicit_base_url,
                       target_model) -> Optional[Dict[str, Any]]:
    """Runtime from the provider's credential pool, or None to continue down the ladder."""
    try:
        pool = load_pool(provider)
    except Exception:
        pool = None
    if not (pool and pool.has_credentials()):
        return None
    entry = pool.select()
    if entry is None:
        return None
    pool_api_key = _pool_entry_api_key(entry)
    if provider == "nous":
        entry, pool_api_key = _refresh_nous_pool_entry(pool, entry, pool_api_key)
    if pool_api_key and credential_pool_matches_provider(pool, provider, base_url=_pool_entry_base_url(entry)):
        return _resolve_runtime_from_pool_entry(provider=provider, entry=entry, requested_provider=requested_provider,
                                                model_cfg=model_cfg, pool=pool, target_model=target_model)
    return None


# ── explicit (--api-key / --base-url) path ─────────────────────────────────────────────────





def _creds_fallback(api_key, explicit_base_url, base_url, expiry, expiry_key, resolve):
    """When no explicit key was given, take api_key / expiry / base_url from stored credentials
    (an explicit --base-url still wins over the stored one)."""
    if api_key:
        return api_key, base_url, expiry
    creds = resolve()
    return creds.get("api_key", ""), explicit_base_url or creds.get("base_url", "").rstrip("/") or base_url, creds.get(expiry_key)





def _explicit_nous(requested_provider, model_cfg, api_key, explicit_base_url, target_model):
    state = auth_mod.get_provider_auth_state("nous") or {}
    base_url = (explicit_base_url or _nous_inference_env_override()
                or str(state.get("inference_base_url") or auth_mod.DEFAULT_NOUS_INFERENCE_URL).strip().rstrip("/"))
    # The agent_key compatibility field is used for inference only when it holds a NAS invoke JWT;
    # raw OAuth access_token fallback is handled by resolve_nous_runtime_credentials().
    api_key = api_key or (str(state.get("agent_key") or "").strip() if _agent_key_is_usable(state, _nous_min_key_ttl()) else "")
    api_key, base_url, expires_at = _creds_fallback(api_key, explicit_base_url, base_url,
                                                    state.get("agent_key_expires_at") or state.get("expires_at"), "expires_at",
                                                    _resolve_nous_creds)
    return _runtime("nous", nous_api_mode(_effective_model(model_cfg, target_model)), base_url, api_key, source="explicit",
                    expires_at=expires_at, requested_provider=requested_provider)








def _explicit_api_key_provider(provider, pconfig, requested_provider, model_cfg, api_key, base_url, target_model):
    if not api_key:
        creds = resolve_api_key_provider_credentials(provider)
        api_key = creds.get("api_key", "")
        base_url = base_url or creds.get("base_url", "")
    base_url = base_url or get_secret_str(pconfig.base_url_env_var, "") or pconfig.inference_base_url
    mode = _copilot_runtime_api_mode(model_cfg, api_key, target_model=target_model)
    return _runtime(provider, mode, base_url.rstrip("/"), api_key, source="explicit", requested_provider=requested_provider)


# Providers with a dedicated explicit-credential builder; everything else goes through the
# registry ``api_key`` path (or None when the provider takes no explicit creds).
_EXPLICIT_RESOLVERS: Dict[str, Callable[..., Dict[str, Any]]] = {"nous": _explicit_nous}


def _resolve_explicit_runtime(*, provider: str, requested_provider: str, model_cfg: Dict[str, Any],
                              explicit_api_key: Optional[str] = None, explicit_base_url: Optional[str] = None,
                              target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url = str(explicit_base_url or "").strip().rstrip("/")
    if not explicit_api_key and not explicit_base_url:
        return None
    resolver = _EXPLICIT_RESOLVERS.get(provider)
    if resolver is not None:
        return resolver(requested_provider, model_cfg, explicit_api_key, explicit_base_url, target_model)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if not (pconfig and pconfig.auth_type == "api_key"):
        return None
    return _explicit_api_key_provider(provider, pconfig, requested_provider, model_cfg, explicit_api_key, explicit_base_url, target_model)


# ── OAuth / auth-store providers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _OAuthRuntimeSpec:
    """Env/auth-store OAuth providers resolved by a single credential call."""

    resolve: Callable[[], Dict[str, Any]]
    api_mode: Any  # str, or callable(model) -> str
    default_source: str
    expiry_key: str
    failure_msg: str
    default_base_url: str = ""


# ``resolve`` entries are late-bound lambdas so tests can monkeypatch the module-level
# ``resolve_*_runtime_credentials`` names.
_OAUTH_RUNTIME_PROVIDERS: Dict[str, _OAuthRuntimeSpec] = {
    "nous": _OAuthRuntimeSpec(_resolve_nous_creds, nous_api_mode, "portal", "expires_at",
                              "Nous credentials failed"),
}


def _resolve_oauth_runtime(provider, requested_provider, model_cfg, target_model) -> Optional[Dict[str, Any]]:
    """Runtime from an ``_OAUTH_RUNTIME_PROVIDERS`` spec. On AuthError: re-raise for an explicit
    request; for "auto" (auto-detected but credentials stale/revoked) log and return None so the
    ladder falls through to env-var providers (e.g. OpenRouter)."""
    spec = _OAUTH_RUNTIME_PROVIDERS[provider]
    try:
        creds = spec.resolve()
    except AuthError:
        if requested_provider != "auto":
            raise
        logger.info("%s; falling through to next provider.", spec.failure_msg)
        return None
    api_mode = spec.api_mode(_effective_model(model_cfg, target_model)) if callable(spec.api_mode) else spec.api_mode
    return _runtime(provider, api_mode, (creds.get("base_url") or "").rstrip("/") or spec.default_base_url,
                    creds.get("api_key", ""), source=creds.get("source", spec.default_source),
                    **{spec.expiry_key: creds.get(spec.expiry_key)}, requested_provider=requested_provider)





# ── env/config paths for anthropic and registry api_key providers ──────────────────────────








def _api_key_provider_runtime(provider, pconfig, requested_provider, model_cfg, target_model) -> Dict[str, Any]:
    creds = resolve_api_key_provider_credentials(provider)
    if not has_usable_secret(creds.get("api_key")):
        raise AuthError("No usable Copilot credentials. Run 'hermes model' to sign in.", provider=provider, code="missing_api_key")
    base_url = _config_base_url_for_provider(model_cfg, provider) or creds.get("base_url", "").rstrip("/")
    mode = _copilot_runtime_api_mode(model_cfg, creds["api_key"], target_model=target_model)
    return _runtime(provider, mode, base_url, creds["api_key"], source=creds.get("source", "env"), requested_provider=requested_provider)


# ── the resolution ladder ──────────────────────────────────────────────────────────────────





def _raise_if_provider_disabled(requested_provider: str) -> None:
    """Honour ``providers.<name>.enabled: false`` for built-ins too (the custom lookup gate only
    covers custom blocks); a typed error lets the fallback chain advance."""
    full_cfg = _config_mod.load_config()
    provs_cfg = full_cfg.get("providers") if isinstance(full_cfg, dict) else None
    block = provs_cfg.get(requested_provider) if isinstance(provs_cfg, dict) else None
    if isinstance(block, dict) and not _config_mod.is_provider_enabled(block):
        raise ValueError(f"provider {requested_provider!r} is disabled in config "
                         f"(providers.{requested_provider}.enabled: false)")











def _tag(runtime: Optional[Dict[str, Any]], requested_provider: str) -> Optional[Dict[str, Any]]:
    """Stamp ``requested_provider`` on a runtime built by a collaborator that does not set it."""
    if runtime:
        runtime["requested_provider"] = requested_provider
        if runtime.get("provider") == "custom":
            from hermes_cli.runtime_provider_custom import _finalize_custom_runtime
            runtime = _finalize_custom_runtime(runtime)
    return runtime








def resolve_runtime_provider(*, requested: Optional[str] = None, explicit_api_key: Optional[str] = None,
                             explicit_base_url: Optional[str] = None, target_model: Optional[str] = None) -> Dict[str, Any]:
    """Resolve a supported account or explicitly configured custom endpoint before auth/network I/O."""
    from hermes_cli.provider_policy import require_supported_provider
    validate_provider_pins(requested)
    requested_provider = resolve_requested_provider(requested)
    _raise_if_provider_disabled(requested_provider)
    provider = require_supported_provider(resolve_provider(
        requested_provider, explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url))
    if provider == "custom" or provider.startswith("custom:"):
        runtime = _resolve_named_custom_runtime(requested_provider=provider, explicit_api_key=explicit_api_key,
                                               explicit_base_url=explicit_base_url, target_model=target_model)
        if runtime:
            return _tag(runtime, requested_provider)
        if provider.startswith("custom:"):
            raise ValueError(f"Custom endpoint {provider!r} is not configured. Run 'hermes model' to add it.")
        return _tag(_resolve_custom_runtime(requested_provider=provider, explicit_api_key=explicit_api_key,
                                            explicit_base_url=explicit_base_url), requested_provider)
    model_cfg = _get_model_config()
    runtime = _resolve_explicit_runtime(provider=provider, requested_provider=requested_provider, model_cfg=model_cfg,
                                       explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url,
                                       target_model=target_model)
    if runtime:
        return runtime
    runtime = _resolve_from_pool(provider, requested_provider, model_cfg, explicit_api_key, explicit_base_url, target_model)
    if runtime:
        return runtime
    if provider == "nous":
        return _resolve_oauth_runtime(provider, provider, model_cfg, target_model)
    return _api_key_provider_runtime(provider, PROVIDER_REGISTRY[provider], requested_provider, model_cfg, target_model)








def format_runtime_provider_error(error: Exception) -> str:
    return format_auth_error(error) if isinstance(error, AuthError) else str(error)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'custom_provider_aliases': ('hermes_cli.providers', 'custom_provider_aliases'),
    'custom_provider_slug': ('hermes_cli.providers', 'custom_provider_slug'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
