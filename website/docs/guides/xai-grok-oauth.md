---
sidebar_position: 16
title: "xAI Tool Credentials"
description: "xAI credentials for speech, image/video generation, and X search; native inference is retired in this fork"
---

# xAI Tool Credentials

The built-in `xai` / `xai-oauth` **inference providers are removed in this fork**. Do not select them with `hermes model`, `model.provider`, or a fallback entry. Grok models may still be offered through Copilot, Nous, or a deliberately configured custom endpoint; see [AI Providers](/integrations/providers).

This does **not** remove xAI's tool integrations. Keep existing tool credentials when migrating inference configuration. Model-provider removal must not erase the auth store or unrelated tool secrets.

## Direct-to-xAI Tools (TTS / Image / Video / Transcription / X Search)

Use `hermes tools` to configure the relevant tool independently of the main model:

```bash
hermes tools
# Text-to-Speech → xAI TTS
# Image Generation → xAI Grok Imagine (image)
# Video Generation → xAI Grok Imagine
# X (Twitter) Search → the xAI tool option
```

`XAI_API_KEY` remains a tool credential. Tool implementations that support stored xAI OAuth can reuse it; use the tool's setup flow rather than the removed model-provider picker. Tool entitlement and billing remain subject to xAI's account limits.

Video generation is off by default; enable it in `hermes tools` before requesting video. X Search requires configured xAI credentials; it uses xAI's search API independently of which model handles your conversation.

### Models

| Tool | Example model / endpoint |
|---|---|
| Image | `grok-imagine-image`, `grok-imagine-image-2.0`, `grok-imagine-image-quality` |
| Video | `grok-imagine-video`, `grok-imagine-video-1.5-preview` |
| TTS | xAI `/v1/tts` endpoint |

Use the tool's current model picker/catalog for availability rather than treating these examples as an entitlement guarantee.

## Troubleshooting

- **Missing credentials:** configure the tool with `hermes tools` and keep its key in your private `.env`.
- **Expired OAuth:** use the tool's supported authentication flow. Do not switch the main inference provider to `xai-oauth`.
- **HTTP 403:** check the tool's account entitlement/billing. Repeated login does not fix a missing entitlement.
- **Disabled tool:** enable its toolset, then start a new session as required.

## See Also

- [Voice & TTS](../user-guide/features/tts.md)
- [Image Generation](../user-guide/features/image-generation.md)
- [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md)
- [Environment Variables](../reference/environment-variables.md)
