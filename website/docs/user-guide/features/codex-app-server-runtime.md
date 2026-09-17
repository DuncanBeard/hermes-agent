---
title: Codex App-Server Runtime (optional)
sidebar_label: Codex App-Server Runtime
---

# Codex App-Server Runtime

This inference integration is **not supported in this fork**. This page remains as a migration notice so existing documentation links do not break.

The Codex account/app-server inference route is removed. Custom servers may still use the Responses protocol (`codex_responses`); the protocol name does not restore Codex login or the app-server runtime. The [Hermes ACP server](/user-guide/features/acp) remains available.

## Migrate deliberately

1. Use `hermes model` to select direct Copilot, Nous, or an explicit custom endpoint.
2. For a custom server, configure its URL, exact model ID, credentials, and supported protocol: Chat Completions, Responses, or Anthropic Messages. A native vendor API is not automatically compatible.
3. Update any explicit retired IDs in auxiliary settings, aliases, fallback chains, delegation, cron, and API routes.
4. Preserve your auth store, history, and tool credentials. Unsupported selections produce errors, not automatic replacements or secret deletion.

Nous Portal login, free-tier/guest onboarding, dashboard authentication, and unrelated media/search/memory integrations remain supported. Tool-provider configuration is separate from inference selection.

See [AI Providers](/integrations/providers) and [Fallback Providers](/user-guide/features/fallback-providers).
