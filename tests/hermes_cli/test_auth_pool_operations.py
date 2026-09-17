"""OAuth control commands against a loopback token endpoint."""
import time
from types import SimpleNamespace

import pytest

from hermes_cli import auth_commands
from hermes_cli.auth import read_credential_pool, write_credential_pool


@pytest.fixture(autouse=True)
def isolated_external_auth_stores(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))


def _rows():
    return [dict(id=f"row{i}", label=f"account{i}", source="manual:device_code",
                 auth_type="oauth", access_token=f"fixture-access-{i}",
                 refresh_token=f"fixture-refresh-{i}", priority=i,
                 last_status="exhausted", last_status_at=time.time(),
                 last_error_code=429, last_error_reset_at=time.time()+3600)
            for i in range(2)]


def test_add_priority_places_reauthenticated_row_in_multi_entry_pool(monkeypatch):
    rows = _rows()
    rows[1]["source"] = "device_code"
    write_credential_pool("nous", rows)
    monkeypatch.setattr(auth_commands.auth_mod, "_read_shared_nous_state", lambda: None)
    monkeypatch.setattr(auth_commands.auth_mod, "_nous_device_code_login", lambda **_kwargs: {
        "access_token": "fixture-renewed", "refresh_token": "fixture-renewed-refresh",
        "agent_key": "fixture-agent-key", "expires_at": time.time() + 3600,
    })
    auth_commands.auth_add_command(SimpleNamespace(
        provider="nous", auth_type="oauth", priority=0, label="reauthenticated"))
    entries = read_credential_pool("nous")
    assert [e["id"] for e in entries] == ["row1", "row0"]
    assert entries[0]["priority"] == 0


def test_refresh_rejects_ambiguous_and_non_oauth_targets():
    rows = _rows()
    write_credential_pool("nous", rows)
    with pytest.raises(SystemExit, match="pass an index"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="nous", target=None))
    with pytest.raises(SystemExit, match="No credential matching"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="nous", target="missing"))
    rows[0].update(auth_type="api_key", source="manual")
    write_credential_pool("custom:manual", rows[:1])
    with pytest.raises(SystemExit, match="not a refreshable"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="custom:manual", target=None))
    # Nous's resolver refreshes only its singleton, never an independent pool grant.
    write_credential_pool("nous", _rows())
    before = read_credential_pool("nous")
    with pytest.raises(SystemExit, match="not a refreshable"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="nous", target="row0"))
    assert read_credential_pool("nous") == before
