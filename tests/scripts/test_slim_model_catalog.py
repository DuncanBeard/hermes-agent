"""Published model catalogs follow the same provider boundary as runtime."""
from scripts.build_model_catalog import build_catalog
from hermes_cli.provider_policy import SUPPORTED_BUILTIN_PROVIDERS


def test_published_model_catalog_only_advertises_supported_accounts():
    catalog = build_catalog()
    assert set(catalog["providers"]) <= SUPPORTED_BUILTIN_PROVIDERS
    assert "custom" not in catalog["providers"]  # endpoints are private per-profile configuration
    assert "nous" in catalog["providers"]
    assert "copilot" in catalog["providers"]
    for entry in catalog["providers"].values():
        assert entry["metadata"]["display_name"]
        assert entry["models"]
        assert all(model["id"] for model in entry["models"])
