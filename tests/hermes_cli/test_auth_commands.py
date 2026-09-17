"""Tests for auth subcommands backed by the credential pool."""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest
import yaml


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _write_groq_provider_config(
    tmp_path, *, provider_key="groq", name="Groq", base_url=None
) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    provider_key: {
                        "name": name,
                        "base_url": base_url or "https://api.groq.com/openai/v1",
                        "key_env": "GROQ_API_KEY",
                        "discover_models": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _jwt_with_email(email: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"email": email}).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_auth_add_api_key_persists_manual_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    _write_groq_provider_config(tmp_path)

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "custom:groq"
        auth_type = "api-key"
        api_key = "sk-or-manual"
        label = "personal"

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["custom:groq"]
    entry = next(item for item in entries if item["source"] == "manual")
    assert entry["label"] == "personal"
    assert entry["auth_type"] == "api_key"
    assert entry["source"] == "manual"
    assert entry["access_token"] == "sk-or-manual"


def test_auth_add_configured_provider_uses_canonical_pool_key(tmp_path, monkeypatch):
    """A keyed providers row must keep its runtime slug in the auth pool."""
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    _write_groq_provider_config(tmp_path)

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "groq"
        auth_type = "api-key"
        api_key = "gsk-test"
        label = "primary"

    auth_add_command(_Args())

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    assert "custom:groq" in payload["credential_pool"]
    assert "groq" not in payload["credential_pool"]


def test_auth_add_preserves_prefixed_key_for_configured_provider(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "custom:groq": [
                    {
                        "id": "legacy-key",
                        "label": "legacy",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "gsk-legacy",
                    }
                ]
            },
        },
    )
    _write_groq_provider_config(tmp_path)

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "groq"
        auth_type = "api-key"
        api_key = "gsk-new"
        label = "new"

    auth_add_command(_Args())

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    assert "groq" not in payload["credential_pool"]
    assert {
        entry["access_token"] for entry in payload["credential_pool"]["custom:groq"]
    } == {"gsk-legacy", "gsk-new"}


def test_auth_add_does_not_merge_different_named_endpoint_identities(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "custom:groq": [
                    {
                        "id": "legacy-key",
                        "label": "legacy",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "gsk-legacy",
                    }
                ]
            },
        },
    )
    _write_groq_provider_config(tmp_path, provider_key="groq-cloud", name="Groq")

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "groq-cloud"
        auth_type = "api-key"
        api_key = "gsk-new"
        label = "new"

    with patch("hermes_cli.models.clear_provider_models_cache") as clear_cache:
        auth_add_command(_Args())

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    assert "groq" not in payload["credential_pool"]
    assert {
        entry["access_token"]
        for entry in payload["credential_pool"]["custom:groq-cloud"]
    } == {"gsk-new"}
    assert payload["credential_pool"]["custom:groq"][0]["access_token"] == "gsk-legacy"
    clear_cache.assert_not_called()


def test_auth_add_non_registry_configured_provider_preserves_endpoint(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    _write_groq_provider_config(
        tmp_path,
        provider_key="private-groq",
        base_url="https://private.example/v1",
    )

    from hermes_cli.auth_commands import auth_add_command

    auth_add_command(
        type(
            "Args",
            (),
            {
                "provider": "private-groq",
                "auth_type": "api-key",
                "api_key": "private-key",
                "label": "private",
            },
        )()
    )

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    entry = payload["credential_pool"]["custom:private-groq"][0]
    from hermes_cli.runtime_provider import resolve_runtime_provider
    resolved = resolve_runtime_provider(requested="custom:private-groq")
    assert resolved["base_url"] == "https://private.example/v1"
    assert resolved["api_key"] == entry["access_token"]


def test_auth_list_includes_non_registry_configured_provider(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_groq_provider_config(tmp_path, provider_key="private-groq")
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "custom:private-groq": [
                    {
                        "id": "private-key",
                        "label": "private",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "secret",
                    }
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_list_command

    auth_list_command(type("Args", (), {"provider": None})())

    assert "custom:private-groq (1 credentials):" in capsys.readouterr().out


def test_auth_list_shows_entry_id_and_priority(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "custom:groq": [
                    {
                        "id": "ab12cd",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "secret-1",
                    },
                    {
                        "id": "ef56gh",
                        "label": "backup",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "secret-2",
                    },
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_list_command

    auth_list_command(type("Args", (), {"provider": "custom:groq"})())

    out = capsys.readouterr().out
    assert "id=ab12cd priority=0" in out
    assert "id=ef56gh priority=1" in out


def test_interactive_auth_add_accepts_non_registry_configured_provider(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    _write_groq_provider_config(tmp_path, provider_key="private-groq")

    from hermes_cli import auth_commands

    monkeypatch.setattr(auth_commands, "_pick_provider", lambda _prompt: "private-groq")
    monkeypatch.setattr(auth_commands, "line_input", lambda _prompt: "private")
    monkeypatch.setattr(auth_commands, "masked_secret_prompt", lambda _prompt: "private-key")

    auth_commands._interactive_add()

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    assert payload["credential_pool"]["custom:private-groq"][0]["access_token"] == "private-key"


def test_interactive_auth_add_normalizes_display_name_to_provider_key(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    _write_groq_provider_config(
        tmp_path, provider_key="groq-cloud", name="Groq Enterprise"
    )

    from hermes_cli import auth_commands

    answers = iter(["Groq Enterprise", "primary"])
    monkeypatch.setattr(auth_commands, "line_input", lambda _prompt: next(answers))
    monkeypatch.setattr(auth_commands, "masked_secret_prompt", lambda _prompt: "gsk-test")

    auth_commands._interactive_add()

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    assert "custom:groq-enterprise" not in payload["credential_pool"]
    assert payload["credential_pool"]["custom:groq-cloud"][0]["access_token"] == "gsk-test"


def test_auth_add_explicit_custom_provider_keeps_prefixed_pool_key(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    _write_groq_provider_config(
        tmp_path,
        name="Groq Proxy",
        base_url="https://proxy.example/v1",
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "custom:groq"
        auth_type = "api-key"
        api_key = "proxy-key"
        label = "proxy"

    auth_add_command(_Args())

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    assert "custom:groq" in payload["credential_pool"]
    assert "groq" not in payload["credential_pool"]


def test_auth_add_nous_oauth_persists_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    token = _jwt_with_email("nous@example.com")
    monkeypatch.setattr(
        "hermes_cli.auth._nous_device_code_login",
        lambda **kwargs: {
            "portal_base_url": "https://portal.example.com",
            "inference_base_url": "https://inference.example.com/v1",
            "client_id": "hermes-cli",
            "scope": "inference:invoke",
            "token_type": "Bearer",
            "access_token": token,
            "refresh_token": "refresh-token",
            "obtained_at": "2026-03-23T10:00:00+00:00",
            "expires_at": "2026-03-23T11:00:00+00:00",
            "expires_in": 3600,
            "agent_key": token,
            "agent_key_id": None,
            "agent_key_expires_at": "2026-03-23T10:30:00+00:00",
            "agent_key_expires_in": 1800,
            "agent_key_reused": False,
            "agent_key_obtained_at": "2026-03-23T10:00:10+00:00",
            "tls": {"insecure": False, "ca_bundle": None},
        },
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "nous"
        auth_type = "oauth"
        api_key = None
        label = None
        portal_url = None
        inference_url = None
        client_id = None
        scope = None
        no_browser = False
        timeout = None
        insecure = False
        ca_bundle = None

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())

    # Pool has exactly one canonical `device_code` entry — not a duplicate
    # pair of `manual:device_code` + `device_code` (the latter would be
    # materialised by _seed_from_singletons on every load_pool).
    entries = payload["credential_pool"]["nous"]
    device_code_entries = [
        item for item in entries if item["source"] == "device_code"
    ]
    assert len(device_code_entries) == 1, entries
    assert not any(item["source"] == "manual:device_code" for item in entries)
    entry = device_code_entries[0]
    assert entry["source"] == "device_code"
    assert entry["agent_key"] == token
    assert entry["portal_base_url"] == "https://portal.example.com"

    # `hermes auth add nous` must also populate providers.nous so the
    # 401-recovery path (resolve_nous_runtime_credentials) can refresh an
    # invoke JWT when the token expires. If this mirror is missing, recovery
    # raises "Hermes is not logged into Nous Portal" and the agent dies.
    singleton = payload["providers"]["nous"]
    assert singleton["access_token"] == token
    assert singleton["refresh_token"] == "refresh-token"
    assert singleton["agent_key"] == token
    assert singleton["portal_base_url"] == "https://portal.example.com"
    assert singleton["inference_base_url"] == "https://inference.example.com/v1"


def test_auth_add_nous_oauth_honors_custom_label(tmp_path, monkeypatch):
    """`hermes auth add nous --type oauth --label <name>` must preserve the
    custom label end-to-end — it was silently dropped in the first cut of the
    persist_nous_credentials helper because `--label` wasn't threaded through.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    token = _jwt_with_email("nous@example.com")
    monkeypatch.setattr(
        "hermes_cli.auth._nous_device_code_login",
        lambda **kwargs: {
            "portal_base_url": "https://portal.example.com",
            "inference_base_url": "https://inference.example.com/v1",
            "client_id": "hermes-cli",
            "scope": "inference:invoke",
            "token_type": "Bearer",
            "access_token": token,
            "refresh_token": "refresh-token",
            "obtained_at": "2026-03-23T10:00:00+00:00",
            "expires_at": "2026-03-23T11:00:00+00:00",
            "expires_in": 3600,
            "agent_key": token,
            "agent_key_id": None,
            "agent_key_expires_at": "2026-03-23T10:30:00+00:00",
            "agent_key_expires_in": 1800,
            "agent_key_reused": False,
            "agent_key_obtained_at": "2026-03-23T10:00:10+00:00",
            "tls": {"insecure": False, "ca_bundle": None},
        },
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "nous"
        auth_type = "oauth"
        api_key = None
        label = "my-nous"
        portal_url = None
        inference_url = None
        client_id = None
        scope = None
        no_browser = False
        timeout = None
        insecure = False
        ca_bundle = None

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())

    # Custom label reaches the pool entry …
    pool_entry = payload["credential_pool"]["nous"][0]
    assert pool_entry["source"] == "device_code"
    assert pool_entry["label"] == "my-nous"

    # … and survives in providers.nous so a subsequent load_pool() re-seeds
    # it without reverting to the auto-derived fingerprint.
    assert payload["providers"]["nous"]["label"] == "my-nous"


def test_auth_remove_reindexes_priorities(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Prevent pool auto-seeding from host env vars and file-backed sources
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "custom:other": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-ant-api-primary",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "sk-ant-api-secondary",
                    },
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_remove_command

    class _Args:
        provider = "custom:other"
        target = "1"

    auth_remove_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["custom:other"]
    assert len(entries) == 1
    assert entries[0]["label"] == "secondary"
    assert entries[0]["priority"] == 0


def test_clear_provider_auth_removes_provider_pool_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "active_provider": "custom:other",
            "providers": {
                "custom:other": {"access_token": "legacy-token"},
            },
            "credential_pool": {
                "custom:other": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:hermes_pkce",
                        "access_token": "pool-token",
                    }
                ],
                "custom:groq": [
                    {
                        "id": "cred-2",
                        "label": "other-provider",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-or-test",
                    }
                ],
            },
        },
    )

    from hermes_cli.auth import clear_provider_auth

    assert clear_provider_auth("custom:other") is True

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert payload["active_provider"] is None
    assert "custom:other" not in payload.get("providers", {})
    assert "custom:other" not in payload.get("credential_pool", {})
    assert "custom:groq" in payload.get("credential_pool", {})


def test_unsuppress_credential_source_clears_marker(tmp_path, monkeypatch):
    """unsuppress_credential_source() removes a previously-set marker."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1})

    from hermes_cli.auth import suppress_credential_source, unsuppress_credential_source, is_source_suppressed

    suppress_credential_source("nous", "device_code")
    assert is_source_suppressed("nous", "device_code") is True

    cleared = unsuppress_credential_source("nous", "device_code")
    assert cleared is True
    assert is_source_suppressed("nous", "device_code") is False

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    # Empty suppressed_sources dict should be cleaned up entirely
    assert "suppressed_sources" not in payload


def test_unsuppress_credential_source_preserves_other_markers(tmp_path, monkeypatch):
    """Clearing one marker must not affect unrelated markers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1})

    from hermes_cli.auth import (
        suppress_credential_source,
        unsuppress_credential_source,
        is_source_suppressed,
    )

    suppress_credential_source("nous", "device_code")
    suppress_credential_source("custom:other", "config:custom:other")

    assert unsuppress_credential_source("nous", "device_code") is True
    assert is_source_suppressed("custom:other", "config:custom:other") is True


# =============================================================================
# Unified credential-source stickiness — every source Hermes reads from has a
# registered RemovalStep in agent.credential_sources, and every seeding path
# gates on is_source_suppressed.  Below: one test per source proving remove
# sticks across a fresh load_pool() call.
# =============================================================================


def test_credential_sources_registry_has_expected_steps():
    """Sanity check — the registry contains the expected RemovalSteps.

    Adding a new credential source is routine, so this is a structural
    invariant check (every step has a description, every step is unique,
    core steps are present) rather than a frozen snapshot. Frozen
    snapshots of catalog-like data violate the AGENTS.md "don't write
    change-detector tests" rule — they break every time someone adds a
    provider.
    """
    from agent.credential_sources import _REGISTRY

    descriptions = [step.description for step in _REGISTRY]
    # No empty descriptions, no duplicates.
    assert all(d for d in descriptions), "Every removal step must have a description"
    assert len(descriptions) == len(set(descriptions)), (
        f"Registry has duplicate step descriptions: {descriptions}"
    )
    # Core steps must be present — these are the ones the rest of the code
    # assumes exist. When deliberately dropping one, update this list.
    required = {
        "gh auth token / COPILOT_GITHUB_TOKEN / GH_TOKEN",
        "Any env-seeded credential (XAI_API_KEY, DEEPSEEK_API_KEY, etc.)",
        "auth.json providers.nous",
        "Custom provider config.yaml api_key field",
    }
    missing = required - set(descriptions)
    assert not missing, f"Registry missing required steps: {missing}"


def test_credential_sources_find_step_copilot_before_generic_env(tmp_path, monkeypatch):
    """copilot env:GH_TOKEN must dispatch to the copilot step, not the
    generic env-var step.  The copilot step handles the duplicate-source
    problem (same token seeded as both gh_cli and env:<VAR>); the generic
    env step would only suppress one of the variants.
    """
    from agent.credential_sources import find_removal_step

    step = find_removal_step("copilot", "env:GH_TOKEN")
    assert step is not None
    assert "copilot" in step.description.lower() or "gh" in step.description.lower()

    # Generic step still matches any other provider's env var
    step = find_removal_step("custom:groq", "env:GROQ_API_KEY")
    assert step is not None
    assert "env-seeded" in step.description.lower()


def test_auth_remove_copilot_suppresses_all_variants(tmp_path, monkeypatch):
    """Removing any copilot source must suppress gh_cli + all env:* variants
    so the duplicate-seed paths don't resurrect the credential.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # The copilot pool entry is no longer persisted directly in auth.json —
    # `(copilot, gh_cli)` is borrowed and stripped by
    # sanitize_borrowed_credential_payload (PR #31416, May 2026). Tokens are
    # hydrated at runtime via resolve_copilot_token(). Mock that path so the
    # pool has an entry to remove.
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {"copilot": []},
        },
    )

    from types import SimpleNamespace
    from hermes_cli.auth import is_source_suppressed
    from hermes_cli.auth_commands import auth_remove_command

    with patch(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        return_value=("ghp_fake", "gh"),
    ), patch(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        return_value=("ghu_fake_api", None),
    ):
        auth_remove_command(SimpleNamespace(provider="copilot", target="1"))

    assert is_source_suppressed("copilot", "gh_cli")
    assert is_source_suppressed("copilot", "env:COPILOT_GITHUB_TOKEN")
    assert is_source_suppressed("copilot", "env:GH_TOKEN")
    assert is_source_suppressed("copilot", "env:GITHUB_TOKEN")


def test_auth_remove_env_seeded_dotenv_with_bom_no_shell_hint(tmp_path, monkeypatch, capsys):
    """A Notepad-edited .env carries a UTF-8 BOM. The dotenv-vs-shell
    detector must still see the first variable as living in .env (and not
    warn about a phantom shell export). Regression for the reader that
    dropped encoding='utf-8-sig' and misread the BOM'd first line.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write_groq_provider_config(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # BOM prefix (utf-8-sig) + the target var as the FIRST line.
    (hermes_home / ".env").write_bytes(
        b"\xef\xbb\xbfGROQ_API_KEY=sk-ds-only\n"
    )
    monkeypatch.setenv("GROQ_API_KEY", "sk-ds-only")

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "custom:groq": [{
                    "id": "env-1",
                    "label": "GROQ_API_KEY",
                    "auth_type": "api_key",
                    "priority": 0,
                    "source": "env:GROQ_API_KEY",
                    "access_token": "sk-ds-only",
                }]
            },
        },
    )

    from types import SimpleNamespace
    from agent.credential_sources import find_removal_step
    step = find_removal_step("custom:groq", "env:GROQ_API_KEY")
    result = step.remove_fn("custom:groq", SimpleNamespace(source="env:GROQ_API_KEY"))
    out = "\n".join(result.cleaned + result.hints)
    assert "Cleared GROQ_API_KEY from .env" in out
    assert "still set in your shell environment" not in out
