---
title: Adding Custom Endpoints
sidebar_label: Adding Providers
---

# Adding Custom Endpoints

This fork does not accept new built-in vendor inference integrations. The supported identities are direct `copilot`, `nous`, and generic/named custom endpoints. Configure a compatible endpoint rather than adding a vendor to an auth registry or installing an external model-provider plugin.

## Choose the protocol, not a vendor identity

| Config transport | API contract |
|---|---|
| `chat_completions` | OpenAI-compatible Chat Completions |
| `codex_responses` | OpenAI-compatible Responses; the historical name does not imply Codex OAuth |
| `anthropic_messages` | Anthropic-compatible Messages; does not enable native vendor login |

Claude models served by Copilot still use Copilot's chat-completions route. GPT-family Copilot routing can use Responses. Model family, account provider, and transport are distinct concepts.

## Configuration checklist

1. Obtain the server's documented base URL and exact model ID.
2. Add a named `providers:` entry with `api`, `transport`, and `default_model`.
3. Use `key_env` for static credentials, or a trusted `key_cmd` helper for short-lived credentials. Keep secrets out of source control.
4. Select `model.provider: custom:<name>` and the exact model ID, or use `hermes model`.
5. Verify a text response, a tool call, and any required streaming/vision behavior against the real server.
6. Exercise auxiliary and fallback routes if you configure them. Unsupported explicit IDs must fail instead of trying another account.

See [AI Providers](/integrations/providers) for complete examples and legacy `custom_providers:` migration.

## Runtime changes

Changes to retained providers or shared protocol adapters should use the shared runtime resolver and supported-provider policy. Do not add a hidden provider via model aliases, custom-name collisions, auxiliary auto-discovery, a fallback entry, or plugin discovery. A name alone does not constitute a configured custom endpoint.

Preserve dynamic credentials on client rebuilds and isolate endpoint credentials, headers, and request bodies on model switches. Do not delete stale auth records or history as a side effect of rejecting a provider.

Unrelated tool/media/memory plugins remain extensible; see their specific plugin guides. Nous Portal onboarding and dashboard OAuth must remain independent of this inference restriction.

## Related docs

- [Provider Runtime](./provider-runtime.md)
- [Model Provider Plugins](./model-provider-plugin.md)
- [Fallback Providers](/user-guide/features/fallback-providers)
