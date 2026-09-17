"""Retired SDK accounts must not participate in status or credential discovery."""
import builtins


def test_interactive_auth_does_not_import_retired_sdk_adapters(monkeypatch, capsys):
    from hermes_cli import auth_commands
    imports = []
    original = builtins.__import__
    def track(name, *args, **kwargs):
        imports.append(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", track)
    monkeypatch.setattr(auth_commands, "auth_list_command", lambda args: None)
    monkeypatch.setattr(auth_commands, "_ask", lambda *args: "5")
    auth_commands._interactive_auth()
    assert not {"agent.bedrock_adapter", "agent.azure_identity_adapter", "boto3"}.intersection(imports)
    assert "Credential Pool Status" in capsys.readouterr().out


def test_retired_routing_config_is_not_an_explicit_credential(monkeypatch):
    from hermes_cli import auth, config
    monkeypatch.setattr(config, "load_config", lambda: {"model": {"provider": "custom"},
                                                      "bedrock": {"region": "us-east-1"}})
    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setattr(auth, "read_credential_pool", lambda provider: [])
    assert not auth.is_provider_explicitly_configured("bedrock")
    assert not auth.is_provider_explicitly_configured("vertex")
    assert auth.is_provider_explicitly_configured("custom")
