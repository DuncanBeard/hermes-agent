---
title: Fallback Providers
sidebar_label: Fallback Providers
---

# Fallback Providers

Fallback lets Hermes continue a turn on another explicitly configured route after a provider failure. In this fork, every primary and fallback route must be **Copilot**, **Nous**, or a **generic/named custom endpoint**. Retired provider IDs are configuration errors, not invitations to auto-migrate or try another account.

## Configure a primary fallback chain

Use a top-level `fallback_providers:` list in `config.yaml`. The model IDs below are placeholders except the Copilot example; replace them with exact IDs offered by your account/server. Authenticate Nous before adding it as a fallback.

```yaml
model:
  provider: copilot
  default: gpt-5.4

providers:
  local:
    api: http://localhost:8000/v1
    transport: chat_completions
    default_model: YOUR_LOCAL_MODEL_ID

fallback_providers:
  - provider: nous
    model: YOUR_NOUS_MODEL_ID
  - provider: custom:local
    model: YOUR_LOCAL_MODEL_ID
```

A generic endpoint can instead be specified directly:

```yaml
fallback_providers:
  - provider: custom
    model: YOUR_SERVER_MODEL_ID
    base_url: https://inference.example.com/v1
    key_env: MY_SERVER_KEY
    api_mode: chat_completions
```

Set the secret in your private environment. Custom endpoints can also use `codex_responses` (Responses) or `anthropic_messages` (Messages) when the server implements that protocol. Neither transport enables the retired Codex or Anthropic account provider. Named custom entries can use a trusted `key_cmd` helper for dynamic credentials.

The legacy single `fallback_model:` mapping remains a compatibility format; prefer `fallback_providers:` for new configuration. Existing credentials are not erased when a provider is rejected. Update provider references deliberately rather than assuming old vendor IDs have become custom endpoints.

## When fallback runs

Fallback can activate after exhausted retries on rate limits/server errors or repeated invalid responses, and on non-retryable authentication/not-found errors. A model refusal is not an empty response to retry indefinitely.

Hermes resolves the candidate's credentials, endpoint, protocol, and model, then switches clients while preserving conversation history. A different model may interpret the history differently and can have different tool, vision, and context limits.

Fallback is turn-scoped: a new user turn normally tries the primary again. A known rate-limit reset or cooldown can delay eligibility. Becoming eligible is not a guarantee of recovery. Switching model/account may lose the warm prompt cache and incur a full input read; do not assume fallback is free.

## Auxiliary tasks

Side tasks such as vision, compression, session search, and title generation use `auxiliary.<task>`. `auto` normally follows the main route and your declared fallback policy. `main` explicitly selects the active main route for an auxiliary task; it is not a top-level account provider.

```yaml
auxiliary:
  compression:
    provider: auto
    model: ""
    fallback_chain:
      - provider: custom:local
        model: YOUR_LOCAL_MODEL_ID
  vision:
    provider: copilot
    model: YOUR_COPILOT_VISION_MODEL_ID
```

Task fallback entries and the top-level chain are restricted to the same supported routes. A `base_url` is not a way to bypass rejection of a retired explicit provider: declare `custom` and the intended protocol. No branch may discover a retired inference provider just because a vendor key remains in `.env` for a tool.

Choose a vision-capable model for image tasks. If no configured route can serve a task, expect a warning/error or that task's documented degraded behavior rather than silent billing on an unrelated account. Compression can degrade to dropping middle turns without a summary.

## Delegation and cron

Delegated children may declare `delegation.fallback_providers`. Pinned children use their declared chain; unpinned children can inherit the parent chain. An explicit empty list disables the child chain.

```yaml
delegation:
  provider: copilot
  model: YOUR_COPILOT_MODEL_ID
  fallback_providers:
    - provider: custom:local
      model: YOUR_LOCAL_MODEL_ID
```

Cron agents inherit the configured primary fallback policy and can have per-job primary overrides. These routes must also use Copilot, Nous, or configured custom endpoints. See [Delegation](/user-guide/features/delegation) and [Cron](/user-guide/features/cron).

## Tool providers are separate

Speech, image/video generation, browser/search, and memory integrations keep their own provider settings and credentials. Removing a vendor inference integration does not remove these tools. Nous Portal onboarding and the [Tool Gateway](/user-guide/features/tool-gateway) remain available.

See [AI Providers](/integrations/providers) for setup and migration.
