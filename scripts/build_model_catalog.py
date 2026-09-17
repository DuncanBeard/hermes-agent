#!/usr/bin/env python3
"""Publish curated models for this fork's supported account providers.

Custom endpoints are private configuration and never belong in this manifest.
Live account catalogs remain authoritative; this publishes offline fallbacks.
Run: python scripts/build_model_catalog.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from hermes_cli.models import PREFERRED_SILENT_DEFAULT_MODEL, _PROVIDER_MODELS  # noqa: E402
from hermes_cli.provider_policy import SUPPORTED_BUILTIN_PROVIDERS  # noqa: E402

OUTPUT_PATH = os.path.join(REPO_ROOT, "website", "static", "api", "model-catalog.json")
CATALOG_VERSION = 1
_ACCOUNT_LABELS = {"nous": "Nous Portal", "copilot": "GitHub Copilot"}


def build_catalog() -> dict:
    providers = {}
    for provider, label in _ACCOUNT_LABELS.items():
        if provider not in SUPPORTED_BUILTIN_PROVIDERS:
            continue
        models = []
        for model_id in _PROVIDER_MODELS.get(provider, []):
            entry = {"id": model_id}
            if provider == "nous" and model_id == PREFERRED_SILENT_DEFAULT_MODEL:
                entry["default"] = True
            models.append(entry)
        providers[provider] = {
            "metadata": {
                "display_name": label,
                "note": "Curated fallback; available models depend on the account's live catalog.",
            },
            "models": models,
        }
    return {
        "version": CATALOG_VERSION,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": {
            "source": "DuncanBeard/hermes-agent",
            "docs": "https://github.com/DuncanBeard/hermes-agent/blob/main/website/docs/reference/model-catalog.md",
        },
        "providers": providers,
    }


def main() -> int:
    catalog = build_catalog()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as stream:
        json.dump(catalog, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {OUTPUT_PATH}")
    for provider, block in catalog["providers"].items():
        print(f"  {provider}: {len(block['models'])} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
