---
title: Provider Routing
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

This fork selects among **direct Copilot**, **Nous Portal**, and **custom endpoints**. An upstream service may route a request internally, but that is different from selecting a Hermes provider.

## Select a Hermes route

Use `hermes model` to authenticate and configure a supported route. Use `/model` within a session to select an already-configured model. Named custom servers use `model.provider: custom:<name>` with an entry under `providers:`.

See [AI Providers](/integrations/providers) for setup and the three supported custom protocols: Chat Completions, Responses, and Anthropic Messages.

<a id="per-model-overrides-models"></a>

## Upstream routing

Nous Portal chooses routing centrally and does not accept Hermes's old OpenRouter `provider_routing` preferences. Copilot likewise controls its own model catalog and service routing.

The upstream OpenRouter-specific provider selector is not a supported built-in route in this fork. Do not copy old `provider_routing` examples as a way to enable it. If you deliberately configure a custom proxy that accepts routing options, follow that server's API contract and scope any `extra_body` to its named custom entry; these options are not portable to every endpoint.

## Fallback is a separate policy

[Fallback Providers](/user-guide/features/fallback-providers) configures which supported account/endpoint Hermes may try if a model request fails. Do not assume an upstream routing preference configures Hermes fallback, or that saved credentials authorize automatic use of another account.

Explicit retired provider selections fail with a configuration error. Choose a supported route deliberately and preserve stored secrets/history while migrating.
