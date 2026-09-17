"""Doctor must never discover credentials for retired LLM accounts."""
from hermes_cli import doctor_connectivity as dc
from hermes_cli.provider_catalog import provider_catalog


def test_doctor_only_probes_catalog_accounts_without_reading_retired_credentials(monkeypatch):
    import os
    import httpx
    from hermes_cli import auth, runtime_provider, models

    retired = {"OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "GLM_API_KEY"}
    for key in retired:
        monkeypatch.setenv(key, "retired-test-credential")
    reads = []
    getenv = os.getenv
    monkeypatch.setattr(os, "getenv", lambda key, *args: (reads.append(key), getenv(key, *args))[1])
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected direct HTTP")))
    monkeypatch.setattr(auth, "is_provider_explicitly_configured", lambda slug: slug != "custom")
    resolved = []
    def resolve(**kwargs):
        resolved.append(kwargs["requested"])
        return {"api_key": "test-key", "base_url": "https://test.invalid/v1", "api_mode": "chat_completions"}
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(models, "fetch_api_models", lambda *args, **kwargs: ["test-model"])
    probes = dc.build_probes()
    results = dc.run_probes(probes)
    assert not retired.intersection(reads)
    assert {label for label, _ in probes} == {d.label for d in provider_catalog()}
    assert set(resolved) == {d.slug for d in provider_catalog() if d.slug != "custom"}
    assert all(not result.issues for result in results)


def test_doctor_probes_selected_named_custom_with_its_wire_headers(monkeypatch, tmp_path):
    import yaml
    from hermes_cli import models
    from hermes_cli.config import get_config_path

    config = {"model": {"provider": "custom:private", "default": "test-model"},
              "custom_providers": [{"name": "private", "base_url": "https://private.invalid/v1",
                                    "api_key": "custom-test-key", "api_mode": "anthropic_messages",
                                    "extra_headers": {"X-Private-Tenant": "test-tenant"}}]}
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []
    def fetch(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["test-model"]
    monkeypatch.setattr(models, "fetch_api_models", fetch)
    custom = next(fn for label, fn in dc.build_probes() if label == next(
        d.label for d in provider_catalog() if d.slug == "custom"))
    result = custom()
    assert not result.issues
    assert calls == [("custom-test-key", "https://private.invalid/v1", {
        "timeout": 10, "api_mode": "anthropic_messages", "headers": {"X-Private-Tenant": "test-tenant"}})]
