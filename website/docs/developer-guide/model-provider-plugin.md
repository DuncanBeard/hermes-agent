---
title: Model Provider Plugins
sidebar_label: Model Provider Plugins
---

# Model Provider Plugins in This Fork

This fork limits model account identities to `copilot`, `nous`, and custom endpoints. External model-provider plugins cannot register additional vendor inference providers or replace the retained providers to bypass that policy. Upstream instructions for installing arbitrary `ProviderProfile` plugins do not apply here.

## Extend with configuration

Use named entries in `config.yaml` instead of a model-provider plugin:

```yaml
providers:
  my-server:
    api: https://inference.example.com/v1
    key_env: MY_SERVER_KEY
    transport: chat_completions
    default_model: YOUR_SERVER_MODEL_ID
model:
  provider: custom:my-server
  default: YOUR_SERVER_MODEL_ID
```

The domain and model ID are placeholders. Set the secret in your private environment. Select `codex_responses` for a Responses-compatible server or `anthropic_messages` for a Messages-compatible server. These are wire protocols, not account identities. A configured custom endpoint owns its credentials, headers, and request options; it must not borrow a previously selected provider's token.

See [AI Providers](/integrations/providers) for keyed and legacy custom configuration, live model discovery, and `key_cmd` credential helpers.

## What remains extensible

This restriction concerns **model inference provider registration only**. It does not remove tool plugins, browser/search/image/video providers, speech integrations, memory backends, MCP servers, skills, or other plugin kinds. Their respective developer guides still apply.

The Hermes ACP server is also retained. The removed Copilot-ACP model backend is not the editor-facing ACP server.

## Contributor checks

- Enforce the supported-provider policy before creating clients or doing authentication/network discovery.
- Reject explicit unsupported IDs with an actionable error; never reinterpret them as custom endpoints automatically.
- Require a real configured endpoint for a named custom route.
- Keep protocol adapters independent of model account identities.
- Preserve stored credentials and history when rejecting or migrating configuration.
- Test main, auxiliary, fallback, delegation, cron, and API-server routes, not just the picker.

See [Provider Runtime](./provider-runtime.md) and [Adding Providers](./adding-providers.md).
