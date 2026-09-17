---
title: Provider Runtime Resolution
sidebar_label: Provider Runtime
---

# Provider Runtime Resolution

Model runtime resolution separates **account identity**, **model ID**, **endpoint**, **credentials**, and **wire protocol**. This fork permits only direct Copilot, Nous, and generic/named custom endpoints.

## Policy boundary

`hermes_cli/provider_policy.py` defines the supported account identities. `auto` is a selector, not an account provider. Resolve a named entry against actual configuration before applying the boundary: `custom:<name>` is not proof that an endpoint exists.

An explicit retired or unknown provider must produce an actionable error before provider-specific auth, plugin discovery, or network work. Do not silently switch it to Copilot, Nous, or a custom server. Rejection must preserve credentials and history. Apply the same policy to chat, gateway, auxiliary tasks, fallback, delegation, cron, aliases, and API-server routes.

External model-provider plugins cannot re-enable retired vendors or override retained providers. This does not disable other plugin kinds, the Nous Tool Gateway, or Nous dashboard authentication.

## Shared runtime data

`hermes_cli/runtime_provider.py` resolves client configuration including `provider`, `api_mode`, `base_url`, credentials, and source metadata. Model selection/config remains the source of truth; unrelated tool keys must not auto-select an inference account.

Named entries live under `providers:`; legacy `custom_providers:` entries remain readable. Preserve credential references, transport, headers, model metadata, and endpoint options during format migration. A stale vendor selection is not a custom-format migration.

## Transport dispatch

| Route | Transport |
|---|---|
| Copilot GPT-5+ except `gpt-5-mini` | Responses |
| Copilot Claude, Gemini, and other chat models | Chat Completions |
| Nous | Portal runtime resolution |
| Custom | Explicit `chat_completions`, `codex_responses`, or `anthropic_messages` |

`codex_responses` is the retained internal Responses protocol name, not a supported Codex account provider. Likewise, an Anthropic-compatible adapter is needed by custom servers even though the native Anthropic account integration is removed.

## Credential and request isolation

- Resolve credentials for the selected account/endpoint only.
- Preserve callable `key_cmd` sources across request-client rebuilds and fallback activation.
- Do not send a previous route's key, headers, or `extra_body` to another host.
- Do not probe retired providers merely because their environment variables or saved auth records exist.
- Keep provider errors and diagnostics free of tokens and helper output.

## Auxiliary and fallback routing

Auxiliary tasks use the shared resolver. `main` is an auxiliary selector for the active route; `auto` normally follows the main route and declared fallback policy. Neither may escape the supported set. A custom endpoint or fallback must be explicitly configured.

Fallback changes the active client/model in place while preserving conversation history. Re-evaluate protocol, credentials, capabilities, and request options for the selected route rather than inheriting vendor assumptions. Delegation and cron routes have the same restriction; see [Fallback Providers](/user-guide/features/fallback-providers) for configuration and inheritance.

## ACP distinction

The editor-facing Hermes ACP server (`acp_adapter`) remains. The Copilot-ACP client/provider is removed; do not remove the server as part of inference cleanup.

## Related docs

- [Adding Custom Endpoints](./adding-providers.md)
- [Model Provider Plugins](./model-provider-plugin.md)
- [ACP Internals](./acp-internals.md)
