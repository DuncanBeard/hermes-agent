"""Setup exposes executable flows only for supported model accounts."""
from hermes_cli.provider_policy import SUPPORTED_BUILTIN_PROVIDERS


def test_setup_dispatch_uses_supported_accounts_only():
    from hermes_cli.main import _PROVIDER_MODEL_FLOWS
    assert set(_PROVIDER_MODEL_FLOWS) == SUPPORTED_BUILTIN_PROVIDERS


def test_retired_auth_add_rejected_before_pool_mutation(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from hermes_cli import auth_commands
    from hermes_cli.provider_policy import UnsupportedProviderError
    monkeypatch.setattr(auth_commands, "load_pool", lambda *a, **kw: pytest.fail("retired credential pool was opened"))
    with pytest.raises(UnsupportedProviderError):
        auth_commands.auth_add_command(SimpleNamespace(provider="openrouter", api_key="test-key"))
