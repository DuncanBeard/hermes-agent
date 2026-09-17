"""Only supported model accounts can register, including external plugins."""
import pytest

from hermes_cli.provider_policy import SUPPORTED_BUILTIN_PROVIDERS, UnsupportedProviderError


def test_catalog_and_bundled_profiles_share_supported_boundary():
    from providers import list_providers
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.provider_catalog import provider_catalog
    assert {p.name for p in list_providers()} == SUPPORTED_BUILTIN_PROVIDERS
    assert set(PROVIDER_REGISTRY) <= SUPPORTED_BUILTIN_PROVIDERS
    assert {p.slug for p in provider_catalog()} == SUPPORTED_BUILTIN_PROVIDERS


def test_external_registration_cannot_restore_retired_account():
    from providers import register_provider, get_provider_profile
    from providers.base import ProviderProfile
    with pytest.raises(UnsupportedProviderError):
        register_provider(ProviderProfile(name="openrouter", base_url="https://example.test/v1"))
    with pytest.raises(UnsupportedProviderError):
        register_provider(ProviderProfile(name="custom:proxy", aliases=("openrouter",)))
    assert get_provider_profile("openrouter") is None


def test_retired_catalog_lookup_does_not_query_models_dev(monkeypatch):
    from hermes_cli import providers, models
    from hermes_cli.provider_policy import UnsupportedProviderError
    monkeypatch.setattr(providers, "_models_dev_info", lambda *a, **k: pytest.fail("models.dev queried for retired account"))
    assert providers.get_provider("openrouter") is None
    assert providers.resolve_provider_full("gemini") is None
    with pytest.raises(UnsupportedProviderError):
        models.provider_model_ids("openrouter")


def test_picker_does_not_probe_retired_catalogs(monkeypatch):
    from hermes_cli import model_switch_providers, models
    monkeypatch.setattr(models, "fetch_ollama_cloud_models", lambda *a, **k: pytest.fail("retired cloud catalog probed"))
    monkeypatch.setattr(models, "get_curated_nous_model_ids", lambda: ["test-model"])
    assert set(model_switch_providers._build_curated_lists("", "", "")) <= SUPPORTED_BUILTIN_PROVIDERS
