"""Runtime construction for explicitly configured generic HTTP endpoints."""
from __future__ import annotations

from urllib.parse import urlsplit
from agent.secret_scope import get_secret_str


class MissingCustomEndpointError(ValueError):
    """No custom destination was configured; distinct from a malformed one."""


def validate_custom_endpoint(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    if not url:
        raise MissingCustomEndpointError("No custom endpoint configured. Run 'hermes model' to configure it.")
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username and not parsed.password
        parsed.port  # reject malformed ports before a client/network call
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("A custom endpoint requires an explicit valid http:// or https:// base URL. Run 'hermes model' to configure it.")
    return url


def _resolve_custom_runtime(*, requested_provider: str, explicit_api_key=None, explicit_base_url=None):
    import hermes_cli.runtime_provider as rp
    cfg = rp._get_model_config(discover=False)
    configured_provider = str(cfg.get("provider") or "").strip().lower()
    use_config = configured_provider in {"", "auto", "custom"} or rp._resolves_to_custom(configured_provider)
    base_url = validate_custom_endpoint(explicit_base_url or get_secret_str("CUSTOM_BASE_URL", "")
                                        or (cfg.get("base_url") if use_config else ""))
    from hermes_cli.runtime_provider_custom import _destination_bound_custom_config
    # A named destination may supply its own credentials, never the old model's.
    identity = rp.find_custom_provider_identity(base_url)
    if identity:
        named = rp._resolve_named_custom_runtime(requested_provider=identity, explicit_api_key=explicit_api_key,
                                                 explicit_base_url=base_url)
        if named:
            return named
    cfg = _destination_bound_custom_config(cfg, base_url) if use_config else {}
    mode = rp._parse_api_mode(cfg.get("api_mode")) if use_config else None
    key_env = str(cfg.get("key_env") or cfg.get("api_key_env") or "") if use_config else ""
    candidates = [explicit_api_key, cfg.get("api_key") if use_config else "", get_secret_str(key_env, "") if key_env else "",
                  *rp._host_gated_env_key_candidates(base_url, ollama=True)]
    api_key = next((k if callable(k) else str(k).strip() for k in candidates if callable(k) or rp.has_usable_secret(k)), "")
    if not explicit_api_key:
        pooled = rp._try_resolve_from_custom_pool(base_url, "custom", mode)
        if pooled:
            return pooled
    if use_config and cfg.get("key_cmd") and not explicit_api_key:
        from agent.command_token_source import build_command_token_provider
        api_key = build_command_token_provider(cfg["key_cmd"], "custom") or api_key
    runtime = rp._runtime("custom", mode or rp._detect_api_mode_for_url(base_url) or "chat_completions",
                          base_url, api_key or "no-key-required", source="explicit" if explicit_base_url or explicit_api_key else "env/config")
    if use_config:
        rp._apply_custom_provider_extras(cfg, None, runtime)
    return runtime
