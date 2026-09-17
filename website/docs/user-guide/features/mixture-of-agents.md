---
sidebar_position: 7
title: "Mixture of Agents"
description: "Create named MoA presets that appear as selectable models under the Mixture of Agents provider"
---

# Mixture of Agents

This inference integration is **not supported in this fork**. This page remains as a migration notice so existing documentation links do not break.

The `moa` virtual provider and its provider/preset selection surface are removed. Ordinary [subagent delegation](/user-guide/features/delegation) remains available; it is not a replacement that is enabled automatically.

## Migrate deliberately

1. Use `hermes model` to select direct Copilot, Nous, or an explicit custom endpoint.
2. For a custom server, configure its URL, exact model ID, credentials, and supported protocol: Chat Completions, Responses, or Anthropic Messages. A native vendor API is not automatically compatible.
3. Update any explicit retired IDs in auxiliary settings, aliases, fallback chains, delegation, cron, and API routes.
4. Preserve your auth store, history, and tool credentials. Unsupported selections produce errors, not automatic replacements or secret deletion.

Nous Portal login, free-tier/guest onboarding, dashboard authentication, and unrelated media/search/memory integrations remain supported. Tool-provider configuration is separate from inference selection.

See [AI Providers](/integrations/providers) and [Fallback Providers](/user-guide/features/fallback-providers).
