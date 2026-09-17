"""Dashboard Nous and direct GitHub Copilot device login/status helpers."""

import contextlib
import logging
import functools
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


_LOGGED_OUT: Dict[str, Any] = {"logged_in": False, "source": None}


def _truncate_token(value: Optional[str], visible: int = 6) -> str:
    """``…XXXXXX`` (last N chars) for UI display. JWTs show only the tail of the signature
    segment. A callable (Azure Foundry Entra-ID bearer provider) is NEVER invoked — it yields
    the ``<entra-id-bearer>`` placeholder."""
    if not value:
        return ""
    if callable(value) and not isinstance(value, str):
        return "<entra-id-bearer>"
    s = str(value)
    if s.count(".") >= 2:
        s = s.rsplit(".", 1)[-1]
    return s if len(s) <= visible else f"…{s[-visible:]}"


_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
    {"id": "nous", "name": "Nous Portal", "flow": "device_code", "cli_command": "hermes auth add nous",
     "docs_url": "https://portal.nousresearch.com", "status_fn": None},
    {"id": "copilot", "name": "GitHub Copilot", "flow": "device_code", "cli_command": "hermes auth add copilot",
     "docs_url": "https://docs.github.com/en/copilot", "status_fn": lambda: _copilot_status()},
)
_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_oauth_sessions_lock = threading.Lock()


def _copilot_status() -> Dict[str, Any]:
    """Local credential evidence only; listing accounts must not exchange tokens."""
    from hermes_cli.auth import read_credential_pool
    from hermes_cli.copilot_auth import resolve_copilot_token
    for entry in read_credential_pool("copilot"):
        if entry.get("access_token"):
            return {"logged_in": True, "source": "credential_pool", "source_label": "Hermes",
                    "token_preview": _truncate_token(entry["access_token"])}
    token, source = resolve_copilot_token()
    return {"logged_in": bool(token), "source": "env_var" if source and source != "gh auth token" else "gh_cli",
            "source_label": source, "token_preview": _truncate_token(token)}


def _oauth_profile_name(profile: Optional[str]) -> Optional[str]:
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return None
    return requested


def _oauth_session_profile(session_id: str, fallback: Optional[str] = None) -> Optional[str]:
    """Return the profile that owns an OAuth session, if one was provided."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        profile = sess.get("profile") if sess else None
    return profile or _oauth_profile_name(fallback)


def _oauth_poller(label: str):
    """Wrap a device-code poller body ``fn(session_id, sess)``: vanished session is a no-op,
    success marks ``approved``, any exception records ``error`` + ``error_message`` on the
    session instead of raising (the thread has no caller; the dashboard reads the status)."""
    def deco(fn):
        @functools.wraps(fn)
        def poller(session_id: str) -> None:
            with _oauth_sessions_lock:
                sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            try:
                fn(session_id, sess)
                with _oauth_sessions_lock:
                    # A body that already settled the session (a sign-in the user declined in the
                    # browser is ``denied`` with a ``reason``) keeps its verdict.
                    settled = sess["status"] != "pending"
                    if not settled:
                        sess["status"] = "approved"
                if settled:
                    _log.info("oauth/device: %s login ended %s (session=%s)", label, sess["status"], session_id)
                else:
                    _log.info("oauth/device: %s login completed (session=%s)", label, session_id)
            except Exception as e:
                # Provider exceptions may contain tokens or raw response bodies.
                _log.warning("%s device-code poll failed (session=%s, type=%s)", label, session_id, type(e).__name__)
                with _oauth_sessions_lock:
                    sess["status"] = "cancelled" if sess.get("cancelled") else "error"
                    sess["error_message"] = "Device authorization failed. Please try again."
        return poller
    return deco


@_oauth_poller("copilot")
def _copilot_poller(session_id: str, sess: Dict[str, Any]) -> None:
    from hermes_cli import copilot_auth, auth
    from hermes_cli.web_server_profiles import _config_profile_scope
    interval = sess["interval"]
    while time.time() < sess["expires_at"]:
        if sess.get("cancelled"):
            sess["status"] = "cancelled"
            return
        time.sleep(interval + copilot_auth._DEVICE_CODE_POLL_SAFETY_MARGIN)
        if sess.get("cancelled"):
            sess["status"] = "cancelled"
            return
        result = copilot_auth._post_form("https://github.com/login/oauth/access_token", {
            "client_id": copilot_auth.COPILOT_OAUTH_CLIENT_ID,
            "device_code": sess["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}, 10)
        token = result.get("access_token")
        if token:
            valid, _ = copilot_auth.validate_copilot_token(token)
            if not valid:
                raise ValueError("GitHub returned an unsupported token")
            # Scope is captured on the session, never re-derived after cancellation.
            with _config_profile_scope(sess.get("profile")):
                with _oauth_sessions_lock:
                    if sess.get("cancelled"):
                        sess["status"] = "cancelled"
                        return
                    entries = auth.read_credential_pool("copilot")
                    entries = [e for e in entries if e.get("source") != "manual:dashboard_device"]
                    entries.append({"id": session_id, "source": "manual:dashboard_device", "label": "GitHub device login",
                                    "auth_type": "api_key", "access_token": token, "priority": 0})
                    auth.write_credential_pool("copilot", entries)
                    auth.mark_provider_active_if_unset("copilot")
            return
        error = result.get("error")
        if error == "slow_down":
            interval = max(interval + 5, int(result.get("interval") or 0))
        elif error == "access_denied":
            sess.update(status="denied", error_message="Authorization was denied. Please try again.")
            return
        elif error == "expired_token":
            break
        elif error != "authorization_pending":
            raise ValueError("GitHub device authorization failed. Please try again.")
    sess.update(status="expired", error_message="Device code expired. Please try again.")


def _record_sign_in_state(sess: Dict[str, Any], state: Any) -> None:
    """Write one ``anon_auth.SignInState`` onto the dashboard session, under the sessions lock.

    The whole desktop mapping lives here: the state carries its own copy, so nothing below turns a
    reason into a string. ``completed`` deliberately leaves ``status`` on ``"pending"`` so the
    :func:`_oauth_poller` wrapper stamps ``"approved"`` when the poller returns.
    """
    kind = getattr(state, "kind", "")
    if kind in ("code", "waiting"):
        return          # the start route already published the code
    with _oauth_sessions_lock:
        if kind == "completed":
            sess["account_email"] = state.email or None
            # None when the config was left on the user's own model, as the poll route documents.
            sess["model"] = state.model if state.model_changed else None
            return
        if kind == "declined":
            sess["status"], sess["reason"] = "denied", "user_declined"
            sess["error_message"] = state.copy
            return
        if kind == "timed_out":
            sess["status"], sess["reason"] = "error", "timeout"
            # The enriched device-auth guidance, which the dashboard has room for.
            sess["error_message"] = state.detail or state.copy
            return
        if kind == "retired":
            sess["status"], sess["reason"] = "error", "account_retired"
            sess["error_message"] = state.copy
            return
        if kind == "superseded":
            # Two producers, two screens: the user's own DELETE is a cancellation, while a sign-in
            # started somewhere else (a chat, the terminal) voided this code and is an error the
            # renderer has a dedicated screen for.
            if sess.get("cancelled"):
                sess["status"] = "cancelled"
            else:
                sess["status"], sess["reason"] = "error", "superseded"
                sess["error_message"] = state.copy
            return
        if kind == "failed":
            sess["status"] = "error"
            sess["reason"] = state.reason or "error"
            sess["error_message"] = state.copy    # the chat form: no raw exception reaches the UI
            # Whether a later attempt can succeed, and the wait the service named (seconds).
            sess["retryable"] = bool(getattr(state, "retryable", False))
            sess["retry_after"] = int(getattr(state, "retry_after", 0) or 0)
            return
        # already_signed_in / unavailable: the start route refuses these, so this is unreachable
        # through the dashboard; record rather than crash.
        sess["status"] = "error"
        sess["reason"] = kind or "error"
        sess["error_message"] = state.copy


@_oauth_poller("nous")
def _nous_promotion_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Drain the sign-in the start route began: one shared flow, rendered onto the session.

    The generator was created and advanced to its ``Code`` state by ``_start_nous_device_code``, so
    it is already holding the transfer's codes and its HTTP client. Nothing here is wrapped in
    ``_profile_scope``: that context manager holds a process-global lock and swaps module
    attributes across its ``yield``, and this loop can last the sign-in code's whole expiry. The
    generator scopes its own short config/auth-store sections instead.
    """
    gen = sess.get("_sign_in")
    if gen is None:
        return
    try:
        for state in gen:
            _record_sign_in_state(sess, state)
    finally:
        with contextlib.suppress(Exception):
            gen.close()     # unwinds the suspended HTTP client if we leave early


@_oauth_poller("nous")
def _nous_plain_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller for a plain Nous device-code login (no free-tier identity to transfer).

    A sign-in that carries the free tier's connectors runs through ``anon_auth.run_sign_in`` and
    ``_nous_promotion_poller`` instead; this is the "connect another Nous account" path.
    """
    from hermes_cli.web_server_profiles import _profile_scope
    from hermes_cli.auth import _poll_for_token, persist_nous_credentials, refresh_nous_oauth_from_state
    from hermes_cli import anon_auth
    import httpx
    portal_base_url, client_id = sess["portal_base_url"], sess["client_id"]

    def _cancelled() -> bool:
        # The user abandoned this sign-in (DELETE /sessions/{id}) while this thread was blocked
        # on the portal: nothing it learns afterwards may reach the auth store.
        with _oauth_sessions_lock:
            if sess.get("cancelled"):
                sess["status"] = "cancelled"
                return True
            return False

    with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
        token_data = _poll_for_token(
            client=client, portal_base_url=portal_base_url, client_id=client_id,
            device_code=sess["device_code"], expires_in=max(60, int(sess["expires_at"] - time.time())),
            poll_interval=sess["interval"],
        )
    if _cancelled():
        return
    # Same post-processing as _nous_device_code_login (validate/refresh JWT)
    now = datetime.now(timezone.utc)
    token_ttl = int(token_data.get("expires_in") or 0)
    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": token_data.get("inference_base_url"),
        "client_id": client_id,
        "scope": token_data.get("scope") or sess.get("scope"),
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": (
            datetime.fromtimestamp(now.timestamp() + token_ttl, tz=timezone.utc).isoformat()
            if token_ttl else None
        ),
        "expires_in": token_ttl,
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        full_state = refresh_nous_oauth_from_state(auth_state, timeout_seconds=15.0, force_refresh=False)
        # The final cancellation check and the save share the session lock, so a cancel cannot
        # land between them.
        with _oauth_sessions_lock:
            if sess.get("cancelled"):
                sess["status"] = "cancelled"
                return
            persist_nous_credentials(full_state)
        # A config left on the free tier's route by a retired identity still has to move.
        settled = anon_auth.settle_after_upgrade(full_state)
    with _oauth_sessions_lock:
        sess["account_email"] = None
        sess["model"] = settled.get("model") or None
