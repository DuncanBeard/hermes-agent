"""Displaying config must not mint or expose a callable bearer credential."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock


def test_config_displays_generic_credential_provider(capsys):
    from cli import HermesCLI
    credential = Mock(side_effect=AssertionError("display must not mint tokens"))
    state = SimpleNamespace(api_key="", agent=SimpleNamespace(api_key=credential),
                            model="example", base_url="https://example.invalid", max_turns=1,
                            enabled_toolsets=[], verbose=False, session_start=datetime(2026, 1, 1))
    HermesCLI.show_config(state)
    output = capsys.readouterr().out
    assert "Credential provider" in output
    assert "Entra" not in output
    credential.assert_not_called()
