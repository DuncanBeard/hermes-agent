"""Per-provider model-selection wizard flows for ``hermes setup`` / ``hermes model``.

main / config / auth / models helpers are imported lazily inside bodies: avoids the main.py import
cycle and lets tests patch ``hermes_cli.config.load_config`` etc. at call time. The shared skeleton
lives in :mod:`hermes_cli.model_setup_flows_common`; the custom / Azure / Bedrock flows live in their
own ``model_setup_flows_*`` modules.
"""

from __future__ import annotations

import contextlib
import argparse
import os

from hermes_cli.config import clear_model_endpoint_credentials
from hermes_cli.model_setup_flows_common import (
    _HTTP, _activate_provider_model, _ask, _commit_model_config, _curses_choice,
    _ensure_dict_section, _ensure_flow_api_key, _finish_model,
    _load_config_model_section, _models_dev_merged, _oauth_gate, _persist_model, _pick_model_or_prompt,
    _print_numbered, _prompt_auth_credentials_choice,
    _run_login, _say, _show_curated)
from hermes_cli.model_setup_flows_custom import _model_flow_custom, _model_flow_named_custom




def _env_base_url(base_url_env: str) -> str:
    """Base-URL override from ``.env`` then the process environment ('' when unset)."""
    from hermes_cli.config import get_env_value
    if not base_url_env:
        return ""
    return get_env_value(base_url_env) or os.getenv(base_url_env, "")


def _prompt_base_url_override(effective_base: str, base_url_env: str, *, persist_env: bool = True) -> str:
    """Optional ``Base URL [...]`` prompt; a valid override is saved to *base_url_env*."""
    from hermes_cli.config import save_env_value
    override = _ask(f"Base URL [{effective_base}]: ", cancel_msg="", on_cancel="")
    if override and base_url_env:
        if not override.startswith(_HTTP):
            print("  Invalid URL — must start with http:// or https://. Keeping current value.")
        else:
            if persist_env:
                save_env_value(base_url_env, override)
            return override
    return effective_base


def _report_live_models(model_list, source: str) -> None:
    if model_list:
        print(f"  Found {len(model_list)} model(s) from {source}")








def _nous_login_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        portal_url=getattr(args, "portal_url", None), inference_url=getattr(args, "inference_url", None),
        client_id=getattr(args, "client_id", None), scope=getattr(args, "scope", None),
        no_browser=bool(getattr(args, "no_browser", False)), timeout=getattr(args, "timeout", None) or 15.0,
        ca_bundle=getattr(args, "ca_bundle", None), insecure=bool(getattr(args, "insecure", False)))


def _nous_model_catalog(free_tier: bool, portal_url: str, model_ids: list, pricing: dict):
    """Free/paid-tier catalog for the Nous picker: ``(model_ids, pricing, unavailable_models,
    unavailable_message, policy_narrowed)`` or None (message already printed) when nothing is selectable."""
    from hermes_cli.models_pricing import nous_policy_allowed_ids, restrict_to_nous_policy
    from hermes_cli.models import (
        partition_nous_models_by_tier,
        union_with_portal_free_recommendations,
        union_with_portal_paid_recommendations,
    )

    # Free users: union with the Portal's freeRecommendedModels (newly launched free models appear
    # before the curated list catches up), then partition selectable/unavailable by Portal pricing.
    # Paid users: paidRecommendedModels, no partition. Org policy narrows BEFORE the tier split so a
    # rescued id still has to pass the free/paid predicate.
    unavailable_models: list[str] = []
    unavailable_message = ""
    _policy_allowed = nous_policy_allowed_ids()
    if free_tier:
        try:
            from hermes_cli.nous_account import format_nous_portal_entitlement_message, get_nous_portal_account_info
            _account_info = get_nous_portal_account_info(force_fresh=True)
            unavailable_message = format_nous_portal_entitlement_message(_account_info, capability="paid Nous models") or ""
        except Exception:
            unavailable_message = ""
        model_ids, pricing = union_with_portal_free_recommendations(model_ids, pricing, portal_url)
    else:
        model_ids, pricing = union_with_portal_paid_recommendations(model_ids, pricing, portal_url)
    _before_policy = model_ids
    model_ids = restrict_to_nous_policy(model_ids, _policy_allowed, rescue_empty=True)
    _policy_narrowed = model_ids != _before_policy
    if free_tier:
        model_ids, unavailable_models = partition_nous_models_by_tier(model_ids, pricing, free_tier=True)

    if not model_ids and not unavailable_models:
        print("No models available for Nous Portal after filtering.")
        return None
    if free_tier and not model_ids:
        print("No free models currently available.")
        if unavailable_models:
            from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL
            _url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print(unavailable_message or f"Upgrade at {_url} to access paid models.")
        return None
    return model_ids, pricing, unavailable_models, unavailable_message, _policy_narrowed


def _nous_verified_credentials(creds_or_none=None):
    """Resolve Nous runtime credentials; on failure print the diagnosis (re-login when the
    session expired) and return None."""
    from hermes_cli.auth import (
        AuthError, PROVIDER_REGISTRY, _login_nous, format_auth_error, resolve_nous_runtime_credentials)

    try:
        return resolve_nous_runtime_credentials()
    except Exception as exc:
        relogin = isinstance(exc, AuthError) and exc.relogin_required
        msg = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
        if relogin:
            _say(f"Session expired: {msg}", "Re-authenticating with Nous Portal...\n")
            try:
                _login_nous(_nous_login_args(None), PROVIDER_REGISTRY["nous"])
            except Exception as login_exc:
                print(f"Re-login failed: {login_exc}")
            return None
        print(f"Could not verify credentials: {msg}")
        return None


def _nous_persist_selection(selected: str, creds: dict) -> dict:
    """Nous persist step: model choice + provider state, then rewrite ``model`` on a fresh
    config (the caller's may carry stale custom-provider fields) and clear a conflicting
    OPENAI_BASE_URL / OPENAI_API_KEY. Returns the saved config."""
    from hermes_cli.auth import _save_model_choice, _update_config_for_provider
    from hermes_cli.config import get_env_value, load_config, save_config, save_env_value
    _save_model_choice(selected)
    inference_url = creds.get("base_url", "")
    _update_config_for_provider("nous", inference_url)
    config = load_config()
    current_model_cfg = config.get("model")
    if isinstance(current_model_cfg, dict):
        model_cfg = dict(current_model_cfg)
    elif isinstance(current_model_cfg, str) and current_model_cfg.strip():
        model_cfg = {"default": current_model_cfg.strip()}
    else:
        model_cfg = {}
    model_cfg["provider"] = "nous"
    model_cfg["default"] = selected
    if inference_url and inference_url.strip():
        model_cfg["base_url"] = inference_url.rstrip("/")
    else:
        model_cfg.pop("base_url", None)
    clear_model_endpoint_credentials(model_cfg)
    config["model"] = model_cfg
    if get_env_value("OPENAI_BASE_URL"):
        save_env_value("OPENAI_BASE_URL", "")
        save_env_value("OPENAI_API_KEY", "")
    save_config(config)
    return config


def _model_flow_nous(config, current_model="", args=None):
    """Nous Portal provider: ensure logged in, then pick model."""
    from hermes_cli.auth import get_provider_auth_state, _prompt_model_selection, _login_nous, PROVIDER_REGISTRY
    from hermes_cli.config import load_config
    from hermes_cli.nous_subscription import prompt_enable_tool_gateway
    state = get_provider_auth_state("nous")
    if not state or not state.get("access_token"):
        _say("Not logged into Nous Portal. Starting login...", "")

        def _login_then_offer_gateway(login_args, pconfig):
            _login_nous(login_args, pconfig)
            # Offer Tool Gateway enablement for paid subscribers
            with contextlib.suppress(Exception):
                prompt_enable_tool_gateway(load_config() or {})

        # login_nous already handles model selection + config update
        _run_login(_login_then_offer_gateway, _nous_login_args(args), PROVIDER_REGISTRY["nous"])
        return

    # Already logged in — the curated list (agentic models users know from OpenRouter)
    # instead of the hundreds returned by the live /models endpoint.
    from hermes_cli.models import check_nous_free_tier, get_curated_nous_model_ids
    from hermes_cli.models_pricing import get_pricing_for_provider
    from hermes_cli.model_switch_providers import _free_tier_nous_row
    tier_row = _free_tier_nous_row({"name": "Nous Portal", "models": []})
    if tier_row is None:
        print("The Nous free tier is off for this install; sign in with `hermes auth upgrade` to use Nous models.")
        return
    if tier_row["models"]:
        # Free-tier identity: the welcome host serves the single pinned model; no Portal catalog,
        # pricing, or account lookups apply.
        creds = _nous_verified_credentials()
        if creds is None:
            return
        selected = tier_row["models"][0]
        _nous_persist_selection(selected, creds)
        print(f"Default model set to: {selected} (via {tier_row['name']})")
        return
    model_ids = get_curated_nous_model_ids()
    if not model_ids:
        print("No curated models available for Nous Portal.")
        return

    # Verify credentials are still valid (catches expired sessions early)
    creds = _nous_verified_credentials()
    if creds is None:
        return

    pricing = get_pricing_for_provider("nous")
    # Force fresh account data so recent credit purchases are reflected immediately.
    free_tier = check_nous_free_tier(force_fresh=True)
    if not free_tier:
        from hermes_cli.auth import resolve_nous_runtime_credentials
        try:
            creds = resolve_nous_runtime_credentials(force_refresh=True) or creds
        except Exception:
            # Runtime inference has its own paid-entitlement recovery; don't block.
            pass

    # Portal URL is needed for upgrade links and the recommendations endpoints.
    _nous_portal_url = ""
    with contextlib.suppress(Exception):
        _nous_portal_url = (get_provider_auth_state("nous") or {}).get("portal_base_url", "")

    catalog = _nous_model_catalog(free_tier, _nous_portal_url, model_ids, pricing)
    if catalog is None:
        return
    model_ids, pricing, unavailable_models, unavailable_message, _policy_narrowed = catalog

    from hermes_cli.nous_account import nous_policy_notice
    _policy_notice = nous_policy_notice(removed=_policy_narrowed)
    if _policy_notice:
        print(_policy_notice)
    print(f'Showing {len(model_ids)} curated models — use "Enter custom model name" for others.')

    selected = _prompt_model_selection(
        model_ids, current_model=current_model, pricing=pricing, unavailable_models=unavailable_models,
        portal_url=_nous_portal_url, unavailable_message=unavailable_message, confirm_provider="nous",
        confirm_base_url=creds.get("base_url", ""), confirm_api_key=creds.get("api_key", ""))
    if not selected:
        print("No change.")
        return
    config = _nous_persist_selection(selected, creds)
    print(f"Default model set to: {selected} (via Nous Portal)")
    # Offer Tool Gateway enablement for paid subscribers
    prompt_enable_tool_gateway(config)










def _copilot_model_list(live_ids) -> list:
    """Live GitHub Copilot ids, or the curated fallback with a warning."""
    from hermes_cli.models import _PROVIDER_MODELS
    if live_ids:
        model_list = [model_id for model_id in live_ids if model_id]
        print(f"  Found {len(model_list)} model(s) from GitHub Copilot")
        return model_list
    model_list = _PROVIDER_MODELS.get("copilot", [])
    if model_list:
        _say("  ⚠ Could not auto-detect models from GitHub Copilot — showing defaults.",
             '    Use "Enter custom model name" if you do not see your model.')
    return model_list


def _copilot_catalog(api_key: str):
    """``(catalog, catalog_ids, normalize)`` for a GitHub token; *normalize* canonicalizes a
    model id against the catalog (identity when unknown)."""
    from hermes_cli.models import fetch_github_model_catalog, normalize_copilot_model_id
    catalog = fetch_github_model_catalog(api_key)
    ids = [item.get("id", "") for item in catalog if item.get("id")] if catalog else []

    def _normalize(mid):
        return normalize_copilot_model_id(mid, catalog=catalog, api_key=api_key) or mid

    return catalog, ids, _normalize


def _copilot_obtain_token() -> bool:
    """No Copilot token yet: offer device-code login or manual entry. False = stop."""
    from hermes_cli.config import save_env_value
    _say("No GitHub token configured for GitHub Copilot.", "", "  Supported token types:",
         "    → OAuth token (gho_*)          via `copilot login` or device code flow",
         "    → Fine-grained PAT (github_pat_*)  with Copilot Requests permission",
         "    → GitHub App token (ghu_*)     via environment variable",
         "    ✗ Classic PAT (ghp_*)          NOT supported by Copilot API", "", "  Options:",
         "    1. Login with GitHub (OAuth device code flow)", "    2. Enter a token manually", "    3. Cancel", "")
    choice = _ask("  Choice [1-3]: ", raw=True, cancel_msg="")
    if choice is None:
        return False
    if choice == "1":
        try:
            from hermes_cli.copilot_auth import copilot_device_code_login
            token = copilot_device_code_login()
            if not token:
                print("  Login cancelled or failed.")
                return False
            save_env_value("COPILOT_GITHUB_TOKEN", token)
            _say("  Copilot token saved.", "")
        except Exception as exc:
            print(f"  Login failed: {exc}")
            return False
        return True
    if choice == "2":
        new_key = _ask("  Token (COPILOT_GITHUB_TOKEN): ", secret=True, cancel_msg="")
        if new_key is None:
            return False
        if not new_key:
            print("  Cancelled.")
            return False
        # Validate token type
        with contextlib.suppress(ImportError):
            from hermes_cli.copilot_auth import validate_copilot_token
            valid, msg = validate_copilot_token(new_key)
            if not valid:
                print(f"  ✗ {msg}")
                return False
        save_env_value("COPILOT_GITHUB_TOKEN", new_key)
        _say("  Token saved.", "")
        return True
    print("  Cancelled.")
    return False


def _model_flow_copilot(config, current_model=""):
    """GitHub Copilot flow using env vars, gh CLI, or OAuth device code. The reasoning-effort step
    is the shared post-pick one in ``select_provider_and_model`` (Copilot's per-model level set
    comes from ``github_model_reasoning_efforts`` there)."""
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    from hermes_cli.models import fetch_api_models, copilot_model_api_mode
    provider_id = "copilot"
    pconfig = PROVIDER_REGISTRY[provider_id]
    creds = resolve_api_key_provider_credentials(provider_id)
    api_key = creds.get("api_key", "")
    source = creds.get("source", "")
    if not api_key:
        if not _copilot_obtain_token():
            return
        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = creds.get("api_key", "")
    else:
        if source in {"GITHUB_TOKEN", "GH_TOKEN"}:
            from hermes_cli.env_loader import format_secret_source_suffix
            _say(f"  GitHub token: {api_key[:8]}... ✓ ({source}{format_secret_source_suffix(source)})", "")
        else:
            _say("  GitHub token: ✓ (from `gh auth token`)" if source == "gh auth token" else "  GitHub token: ✓", "")

    effective_base = pconfig.inference_base_url
    catalog, live_models, _normalize = _copilot_catalog(api_key)
    if not catalog:
        live_models = fetch_api_models(api_key, effective_base)

    selected = _pick_model_or_prompt(
        _copilot_model_list(live_models), "Model name: ", current_model=_normalize(current_model),
        confirm_provider=provider_id, confirm_base_url=effective_base, confirm_api_key=api_key)
    if not selected:
        print("No change.")
        return
    selected = _normalize(selected)
    _persist_model(selected, provider_id, base_url=effective_base,
                   api_mode=copilot_model_api_mode(selected, catalog=catalog, api_key=api_key))
    print(f"Default model set to: {selected} (via {pconfig.name})")

























# provider id -> (pconfig, curated, api_key_for_probe, effective_base) -> model list











# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import subprocess  # noqa: F401,E402
import urllib.parse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'BEDROCK_GEO_PREFIXES': ('hermes_cli.model_setup_flows_bedrock', 'BEDROCK_GEO_PREFIXES'),
    'bedrock_model_routable_from_region': ('hermes_cli.model_setup_flows_bedrock', 'bedrock_model_routable_from_region'),
    'bedrock_region_geo_prefix': ('hermes_cli.model_setup_flows_bedrock', 'bedrock_region_geo_prefix'),
    'custom_provider_slug': ('hermes_cli.providers', 'custom_provider_slug'),
    'line_input': ('hermes_cli.cli_output', 'line_input'),
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
