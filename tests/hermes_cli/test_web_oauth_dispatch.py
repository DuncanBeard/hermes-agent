"""Nous OAuth and profile-isolation regressions for the dashboard."""
import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app
import hermes_cli.web_routers.oauth as _rt_oauth
import hermes_cli.web_server_oauth as _web_server_oauth

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def _make_profile_home(tmp_path, monkeypatch, profile="coder"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile_home = tmp_path / "profiles" / profile
    profile_home.mkdir(parents=True)
    return profile_home


def test_oauth_provider_status_uses_profile_query(tmp_path, monkeypatch):
    from hermes_cli import web_server as ws
    from hermes_constants import get_hermes_home

    profile_home = _make_profile_home(tmp_path, monkeypatch)
    observed_homes = []

    def fake_status():
        observed_homes.append(get_hermes_home())
        return {"logged_in": False, "source": None}

    fake_catalog = ({
        "id": "copilot",
        "name": "Fake OAuth",
        "flow": "pkce",
        "cli_command": "hermes auth add copilot",
        "docs_url": "https://example.com",
        "status_fn": fake_status,
    },)
    monkeypatch.setattr(_web_server_oauth, "_OAUTH_PROVIDER_CATALOG", fake_catalog)

    resp = client.get("/api/providers/oauth?profile=coder", headers=HEADERS)

    assert resp.status_code == 200, resp.text
    assert observed_homes == [profile_home]


def test_oauth_session_cannot_be_polled_or_cancelled_from_another_profile(
    tmp_path, monkeypatch
):
    """A named-profile OAuth session must reject default-profile retargeting."""
    from hermes_cli import web_server as ws

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "profiles" / "worker").mkdir(parents=True)
    session_id, _session = _rt_oauth._new_oauth_session(
        "copilot", "device_code", profile="worker"
    )
    try:
        poll_resp = client.get(
            f"/api/providers/oauth/copilot/poll/{session_id}",
            headers=HEADERS,
        )
        assert poll_resp.status_code == 400, poll_resp.text
        assert "profile" in poll_resp.text.lower()

        cancel_resp = client.delete(
            f"/api/providers/oauth/sessions/{session_id}",
            headers=HEADERS,
        )
        assert cancel_resp.status_code == 400, cancel_resp.text
        assert "profile" in cancel_resp.text.lower()
        assert session_id in _web_server_oauth._oauth_sessions

        correct_poll = client.get(
            f"/api/providers/oauth/copilot/poll/{session_id}?profile=worker",
            headers=HEADERS,
        )
        assert correct_poll.status_code == 200, correct_poll.text

        correct_cancel = client.delete(
            f"/api/providers/oauth/sessions/{session_id}?profile=worker",
            headers=HEADERS,
        )
        assert correct_cancel.status_code == 200, correct_cancel.text
    finally:
        _web_server_oauth._oauth_sessions.pop(session_id, None)


def test_cancel_oauth_session_marks_dict_cancelled_before_popping(tmp_path, monkeypatch):
    """The DELETE endpoint must flag the session dict before removing it.

    A background worker holds its own reference to the same dict object;
    it can only observe cancellation if the flag is set on that shared
    object prior to (or instead of) removal from the global session map.
    """
    from hermes_cli import web_server as ws

    _make_profile_home(tmp_path, monkeypatch, profile="coder")
    session_id = "cancel-flag-test"
    _web_server_oauth._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "copilot",
        "flow": "device_code",
        "profile": "coder",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
    }
    worker_ref = _web_server_oauth._oauth_sessions[session_id]

    resp = client.delete(
        f"/api/providers/oauth/sessions/{session_id}?profile=coder",
        headers=HEADERS,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "session_id": session_id}
    assert session_id not in _web_server_oauth._oauth_sessions
    assert worker_ref["cancelled"] is True


def test_nous_dashboard_poller_preserves_effective_scope_when_token_omits_scope(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    session_id = "nous-effective-scope-test"
    _web_server_oauth._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "nous",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "hermes-cli",
        "device_code": "device-code",
        "interval": 5,
        "expires_at": time.time() + 600,
        "scope": auth_mod.DEFAULT_NOUS_SCOPE,
    }
    captured_state = {}

    def fake_refresh_nous_oauth_from_state(state, **kwargs):
        captured_state.update(state)
        return {**state, "agent_key": "jwt-agent-key"}

    monkeypatch.setattr(
        auth_mod,
        "_poll_for_token",
        lambda **kwargs: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "refresh_nous_oauth_from_state",
        fake_refresh_nous_oauth_from_state,
    )
    monkeypatch.setattr(auth_mod, "persist_nous_credentials", lambda state: None)

    try:
        _web_server_oauth._nous_plain_poller(session_id)
        assert captured_state["scope"] == auth_mod.DEFAULT_NOUS_SCOPE
        assert _web_server_oauth._oauth_sessions[session_id]["status"] == "approved"
    finally:
        _web_server_oauth._oauth_sessions.pop(session_id, None)


def test_accounts_offers_every_oauth_provider_from_catalog():
    """PARITY CONTRACT: every accounts-tab provider in the unified catalog (the
    `hermes model` universe) must be offered by /api/providers/oauth. This keeps
    the desktop Accounts tab in lockstep with the CLI picker — no provider the
    CLI can sign into may be missing from the GUI.
    """
    from hermes_cli.provider_catalog import provider_catalog

    resp = client.get("/api/providers/oauth", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    offered = {p["id"] for p in resp.json()["providers"]}
    for d in provider_catalog():
        if d.tab == "accounts":
            assert d.slug in offered, (
                f"{d.slug} is an accounts-tab provider in `hermes model` but is "
                f"missing from the desktop Accounts tab (/api/providers/oauth)"
            )
