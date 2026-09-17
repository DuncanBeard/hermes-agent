"""OAuth provider dashboard routes: catalog/status, disconnect, and in-browser device-code login flows.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import asyncio
import contextlib
import logging
import os
import secrets
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_oauth import (
    _oauth_profile_name, _oauth_sessions, _oauth_sessions_lock, _truncate_token,
)
from hermes_cli.web_routers._common import scoped_to_thread

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_require_token = late("_require_token")
_resolve_profile_dir = late("_resolve_profile_dir", "hermes_cli.web_server_profiles")
_OAUTH_PROVIDER_CATALOG = LateState("_OAUTH_PROVIDER_CATALOG", "hermes_cli.web_server_oauth")
# Pollers are late-bound: they run on a background thread AFTER the route returns, so a
# test's monkeypatch on web_server_oauth must win at spawn time, not router-import time.
# A direct import here made those mocks no-ops — the real poller then hit the network
# from the leaked thread and segfaulted a later test's collection (CI flake, 2026-09-09).
_nous_plain_poller = late("_nous_plain_poller", "hermes_cli.web_server_oauth")
_nous_promotion_poller = late("_nous_promotion_poller", "hermes_cli.web_server_oauth")
_OAUTH_SESSION_TTL_SECONDS = 15 * 60


def _new_oauth_session(provider_id: str, flow: str, profile: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """Create + register a new OAuth session, return (session_id, session_dict)."""
    sid = secrets.token_urlsafe(16)
    sess = {
        "session_id": sid, "provider": provider_id, "flow": flow, "profile": _oauth_profile_name(profile),
        "created_at": time.time(),
        "status": "pending",  # pending | approved | denied | expired | error
        "error_message": None,
    }
    with _oauth_sessions_lock:
        _oauth_sessions[sid] = sess
    return sid, sess


def _drop_oauth_session(sid: str) -> None:
    """Forget a session that never got started, so the dashboard does not poll a corpse forever."""
    with _oauth_sessions_lock:
        _oauth_sessions.pop(sid, None)


def _start_poller(target, sid: str, prefix: str = "oauth-poll") -> None:
    threading.Thread(target=target, args=(sid,), daemon=True, name=f"{prefix}-{sid[:6]}").start()


def _device_session_started(
    provider_id: str, profile: Optional[str], poller, fields: Dict[str, Any],
    user_code, verification_url, expires_in: int, poll_interval: int,
) -> Dict[str, Any]:
    """Register a device-code session carrying ``fields``, start its poller, return the /start body."""
    sid, sess = _new_oauth_session(provider_id, "device_code", profile=profile)
    sess.update(fields)
    _start_poller(poller, sid)
    return {
        "session_id": sid, "flow": "device_code", "user_code": user_code,
        "verification_url": verification_url, "expires_in": expires_in, "poll_interval": poll_interval,
    }


async def _httpx_call(fn: Callable[[Any], Any], timeout: float = 15.0, **client_kwargs) -> Any:
    """Run ``fn(client)`` off-loop with a short-lived JSON-accepting ``httpx.Client``."""
    import httpx

    def _call():
        with httpx.Client(
            timeout=httpx.Timeout(timeout), headers={"Accept": "application/json"}, **client_kwargs
        ) as client:
            return fn(client)

    return await asyncio.get_running_loop().run_in_executor(None, _call)


_OMIT: Any = object()


def _status_card(
    raw: dict, source, source_label, token_preview, expires_at, has_refresh_token, last_refresh=_OMIT
) -> Dict[str, Any]:
    card = {
        "logged_in": bool(raw.get("logged_in")), "source": source, "source_label": source_label,
        "token_preview": token_preview, "expires_at": expires_at, "has_refresh_token": has_refresh_token,
    }
    if last_refresh is not _OMIT:
        card["last_refresh"] = last_refresh
    return card


# Hand-written status cards per provider id: (hauth getter name, raw -> card).
# Providers absent here fall through to the slug-driven ``get_auth_status``.
# nous: refresh-free local snapshot so listing providers never performs an OAuth
# refresh. xai: source_label is a human-readable origin (auth-store path /
# credential source), not the internal auth_mode string ("oauth_pkce").
_PROVIDER_STATUS: Dict[str, tuple[str, Callable[[dict], dict]]] = {
    "nous": ("get_nous_auth_status_local", lambda r: {**_status_card(
        r, "nous_portal", r.get("portal_base_url") or "Nous Portal",
        _truncate_token(r.get("access_token")), r.get("access_expires_at"), bool(r.get("has_refresh_token")),
    ), "free_tier": bool(r.get("free_tier")), "account_tier": r.get("account_tier")}),
}


def _resolve_provider_status(provider_id: str, status_fn) -> Dict[str, Any]:
    """Dispatch to the right status helper for an OAuth provider entry."""
    try:
        if status_fn is not None:
            return status_fn()
        from hermes_cli import auth as hauth
        entry = _PROVIDER_STATUS.get(provider_id)
        if entry is not None:
            getter, shape = entry
            return shape(getattr(hauth, getter)())
        # Catalog-derived providers (status_fn=None, no hand-written card) still
        # reflect real login state via the canonical slug-driven dispatcher, so
        # a new OAuth/account provider plugin never renders permanently logged-out.
        raw = hauth.get_auth_status(provider_id)
        if isinstance(raw, dict) and "logged_in" in raw:
            return _status_card(
                raw,
                raw.get("source") or raw.get("provider") or provider_id,
                raw.get("source_label") or raw.get("auth_store") or raw.get("auth_store_path")
                or raw.get("base_url") or raw.get("name") or "",
                _truncate_token(raw.get("access_token") or raw.get("api_key")),
                raw.get("expires_at") or raw.get("access_expires_at"),
                bool(raw.get("has_refresh_token")),
            )
    except Exception as e:
        return {"logged_in": False, "error": str(e)}
    return {"logged_in": False}


async def _start_nous_device_code(profile: Optional[str]) -> Dict[str, Any]:
    """Start a Nous sign-in. Over a free-tier identity (``nous.guest`` on) the whole sign-in is the
    shared ``anon_auth.run_sign_in`` flow: this route creates the generator, pulls its first state
    (the transfer's consent link and code) and hands that to the UI, then the poller drains the rest.
    Without a free-tier identity it is the plain device-code flow."""
    from hermes_cli import anon_auth
    from hermes_cli.auth import PROVIDER_REGISTRY, _request_device_code
    from hermes_cli.web_server_profiles import _config_profile_scope, _profile_scope
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL") or pconfig.portal_base_url
    ).rstrip("/")
    with _profile_scope(_oauth_profile_name(profile)):
        guest = anon_auth.current_nous_state() if anon_auth.guest_enabled() else None

    if not anon_auth.is_guest_state(guest):
        device_data = await _httpx_call(lambda client: _request_device_code(
            client=client, portal_base_url=portal_base_url, client_id=pconfig.client_id,
            scope=pconfig.scope))
        expires_in, interval = int(device_data["expires_in"]), int(device_data["interval"])
        fields = dict(
            device_code=str(device_data["device_code"]), portal_base_url=portal_base_url,
            client_id=pconfig.client_id, scope=pconfig.scope,
            interval=interval, expires_at=time.time() + expires_in)
        return _device_session_started(
            "nous", profile, _nous_plain_poller, fields, str(device_data["user_code"]),
            str(device_data["verification_uri_complete"]), expires_in, interval)

    # The session is registered BEFORE the generator exists, so a cancel landing in the start
    # window is already visible to the flow's own cancel check and persist guard.
    sid, sess = _new_oauth_session("nous", "device_code", profile=profile)

    def _cancelled() -> bool:
        with _oauth_sessions_lock:
            return bool(sess.get("cancelled"))

    @contextlib.contextmanager
    def _persist_guard():
        # The desktop's guarantee: the final cancellation check and the save share one lock.
        with _oauth_sessions_lock:
            if sess.get("cancelled"):
                sess["status"] = "cancelled"
                yield False
            else:
                yield True

    gen = anon_auth.run_sign_in(
        timeout_seconds=15.0,
        cancelled=_cancelled,
        # A DELETE from this machine means "not here": nothing is persisted and the install
        # re-mints a free tier on next use.
        cancel_wins_after_promotion=True,
        persist_guard=_persist_guard,
        # Config + auth store only, so the light contextvar scope -- never the skills-module one,
        # whose process-global lock would be held across the whole wait.
        scope=lambda: _config_profile_scope(_oauth_profile_name(profile)),
    )
    try:
        first = await _httpx_call(lambda _client: next(gen))
    except Exception:
        with contextlib.suppress(Exception):
            gen.close()
        _drop_oauth_session(sid)
        raise
    if getattr(first, "kind", "") != "code":     # already signed in, or the free tier is unavailable
        with contextlib.suppress(Exception):
            gen.close()
        _drop_oauth_session(sid)
        raise HTTPException(400, detail=first.copy_terminal)
    with _oauth_sessions_lock:
        # ``device_code`` stays present because other routes read it; the generator owns the real one.
        sess.update(dict(
            portal_base_url=portal_base_url, client_id=pconfig.client_id, scope=pconfig.scope,
            device_code="", claim_code=first.code, interval=first.interval,
            expires_at=time.time() + first.expires_in, _sign_in=gen))
    _start_poller(_nous_promotion_poller, sid)   # last: nothing observes `sess` before it is complete
    return {
        "session_id": sid, "flow": "device_code", "user_code": first.code,
        "verification_url": first.link, "expires_in": first.expires_in,
        "poll_interval": first.interval,
    }


async def _start_copilot_device_code(profile: Optional[str]) -> Dict[str, Any]:
    from hermes_cli import copilot_auth
    data = await asyncio.to_thread(
        copilot_auth._post_form, "https://github.com/login/device/code",
        {"client_id": copilot_auth.COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"}, 15)
    if not data.get("device_code") or not data.get("user_code"):
        raise ValueError("GitHub did not return a device code")
    expires_in = max(1, int(data.get("expires_in") or 900))
    interval = max(1, int(data.get("interval") or 5))
    return _device_session_started(
        "copilot", profile, late("_copilot_poller", "hermes_cli.web_server_oauth"),
        {"device_code": data["device_code"], "interval": interval,
         "expires_at": time.time() + expires_in},
        data["user_code"], data.get("verification_uri") or "https://github.com/login/device",
        expires_in, interval)


_DEVICE_CODE_STARTERS = {"nous": _start_nous_device_code, "copilot": _start_copilot_device_code}


async def _start_device_code_flow(provider_id: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """Hit the provider's device-auth endpoint, spawn its poller, return the display fields."""
    starter = _DEVICE_CODE_STARTERS.get(provider_id)
    if starter is None:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support device-code flow")
    return await starter(profile)


def _require_oauth_provider(provider_id: str) -> None:
    from hermes_cli.provider_policy import require_supported_provider
    try:
        require_supported_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    if provider_id not in {"copilot", "nous"}:
        raise HTTPException(400, detail="Custom endpoints use API keys, not device login")


def _oauth_provider_disconnect_hint(provider: Dict[str, Any], status: Dict[str, Any]) -> Optional[str]:
    """Return the manual disconnect path when the API cannot clear this provider."""
    if status.get("source") == "gh_cli" and status.get("logged_in"):
        return "Managed by GitHub CLI; remove the credential there."
    if status.get("source") == "env_var":
        return "Remove the API key from Settings → Keys instead."
    return None


def _build_oauth_catalog() -> list[Dict[str, Any]]:
    return [dict(p) for p in _OAUTH_PROVIDER_CATALOG if p["id"] in {"copilot", "nous"}]


@router.get("/api/providers/oauth")
async def list_oauth_providers(profile: Optional[str] = None):
    """Every OAuth-capable provider with current status (token_preview is the last
    N chars, never the full token; disconnect_command only for external providers)."""
    def _run():
        providers = []
        for p in _build_oauth_catalog():
            status = _resolve_provider_status(p["id"], p.get("status_fn"))
            disconnect_hint = _oauth_provider_disconnect_hint(p, status)
            providers.append({
                "id": p["id"], "name": p["name"], "flow": p["flow"],
                "cli_command": p["cli_command"],
                "docs_url": p["docs_url"], "disconnect_hint": disconnect_hint,
                "disconnect_command": None,
                "disconnectable": disconnect_hint is None, "status": status,
            })
        return {"providers": providers}

    return await scoped_to_thread(profile, _run)


def _reject_if_not_disconnectable(provider: Dict[str, Any], status: Dict[str, Any]) -> None:
    disconnect_hint = _oauth_provider_disconnect_hint(provider, status)
    if disconnect_hint:
        raise HTTPException(400, f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}")


@router.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(provider_id: str, request: Request, profile: Optional[str] = None):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    _require_token(request)

    _require_oauth_provider(provider_id)

    def _run():
        catalog_by_id = {p["id"]: p for p in _build_oauth_catalog()}
        provider = catalog_by_id.get(provider_id)
        if provider is None:
            raise HTTPException(400, f"Unknown provider: {provider_id}. Available: {', '.join(sorted(catalog_by_id))}")
        # Flow-only rejection first so external providers never reach status resolution.
        _reject_if_not_disconnectable(provider, {})
        _reject_if_not_disconnectable(provider, _resolve_provider_status(provider_id, provider.get("status_fn")))

        try:
            from hermes_cli.auth import clear_provider_auth, invalidate_nous_auth_status_cache
            cleared = clear_provider_auth(provider_id)
            if provider_id == "nous":
                invalidate_nous_auth_status_cache()
            _log.info("oauth/disconnect: %s (cleared=%s)", provider_id, cleared)
            return {"ok": bool(cleared), "provider": provider_id}
        except Exception as e:
            _log.exception("disconnect %s failed", provider_id)
            raise HTTPException(status_code=500, detail=str(e))

    return await scoped_to_thread(profile, _run)


# Device-code sessions are in-memory, cancelled on expiry, and collected on /start.


def _gc_oauth_sessions() -> None:
    cutoff = time.time() - _OAUTH_SESSION_TTL_SECONDS
    with _oauth_sessions_lock:
        for sid in [sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff]:
            sess = _oauth_sessions.pop(sid, None)
            if sess:
                sess["cancelled"] = True


def _validate_oauth_profile(profile: Optional[str]) -> str:
    """Validate the requested profile (404 via ``_resolve_profile_dir``) and return its name."""
    profile_name = _oauth_profile_name(profile)
    if profile_name:
        _resolve_profile_dir(profile_name)
    return profile_name


@router.post("/api/providers/oauth/{provider_id}/start")
async def start_oauth_login(provider_id: str, request: Request, profile: Optional[str] = None):
    """Initiate an OAuth login flow. Token-protected."""
    _require_token(request)
    _require_oauth_provider(provider_id)
    _gc_oauth_sessions()
    _validate_oauth_profile(profile)
    catalog_entry = next((p for p in _OAUTH_PROVIDER_CATALOG if p["id"] == provider_id), None)
    if catalog_entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider_id}")
    if catalog_entry["flow"] == "external":
        raise HTTPException(400, f"{provider_id} uses an external CLI; run `{catalog_entry['cli_command']}` manually")
    try:
        if catalog_entry["flow"] == "device_code":
            return await _start_device_code_flow(provider_id, profile=profile)
    except HTTPException:
        raise
    except Exception as e:
        _log.warning("oauth/start %s failed (%s)", provider_id, type(e).__name__)
        raise HTTPException(status_code=500, detail="Device authorization could not start. Please try again.") from None
    raise HTTPException(status_code=400, detail="Unsupported flow")


@router.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(provider_id: str, session_id: str, profile: Optional[str] = None):
    """Poll a session's status (no auth — read-only state). One endpoint serves
    every device-code flow: all report progress via the worker-updated ``status``."""
    _require_oauth_provider(provider_id)
    requested_profile = _validate_oauth_profile(profile)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if sess["provider"] != provider_id:
        raise HTTPException(status_code=400, detail="Provider mismatch for session")
    if sess.get("profile") != requested_profile:
        raise HTTPException(status_code=400, detail="OAuth session profile mismatch")
    return {
        "session_id": session_id, "status": sess["status"],
        "error_message": sess.get("error_message"), "expires_at": sess.get("expires_at"),
        # Nous over a free-tier identity: why a transfer ended, who signed in, and the default model
        # the completion settled on (None when the config was on the user's own model).
        "reason": sess.get("reason"), "account_email": sess.get("account_email"), "model": sess.get("model"),
        # Failed sign-ins over a free-tier identity: can a later attempt succeed, and after how long.
        "retryable": sess.get("retryable"), "retry_after": sess.get("retry_after"),
    }


@router.delete("/api/providers/oauth/sessions/{session_id}")
async def cancel_oauth_session(session_id: str, request: Request, profile: Optional[str] = None):
    """Cancel a pending OAuth session. Token-protected.

    Marks the session dict ``cancelled`` before popping it so a background
    worker still holding that dict stops
    polling/exchanging/saving instead of completing the login after the user
    believed it was aborted.
    """
    _require_token(request)
    requested_profile = _validate_oauth_profile(profile)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        if sess is not None:
            if sess.get("profile") != requested_profile:
                raise HTTPException(status_code=400, detail="OAuth session profile mismatch")
            sess["cancelled"] = True
            _oauth_sessions.pop(session_id, None)
    if sess is None:
        return {"ok": False, "message": "session not found"}
    return {"ok": True, "session_id": session_id}
