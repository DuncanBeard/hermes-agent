"""Validate a requested ``/model`` value against the active provider's catalog.

Split out of ``hermes_cli.models``. Catalog fetchers defined in ``hermes_cli.models`` are looked up
there at call time (``_m.<name>``) so ``patch("hermes_cli.models.<name>")`` mocks keep intercepting;
local-server probes are looked up on ``hermes_cli.models_local`` (``_ml.<name>``) the same way.

Every provider branch returns a verdict dict (see :func:`_verdict`) or ``None`` for "not decided
here — keep walking the ladder". The ladder ORDER is behavior (see ``_LADDER``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Callable, Optional

from utils import base_url_host_matches
from hermes_constants import openrouter_variant_base


# ── Verdicts ─────────────────────────────────────────────────────────────

def _verdict(accepted: bool, persist: bool, recognized: bool, message: Optional[str]) -> dict[str, Any]:
    return {"accepted": accepted, "persist": persist, "recognized": recognized, "message": message}


def _accept() -> dict[str, Any]:
    return _verdict(True, True, True, None)


def _accept_with_note(message: str) -> dict[str, Any]:
    return _verdict(True, True, True, message)


def _reject(message: str) -> dict[str, Any]:
    return _verdict(False, False, False, message)


def _soft_accept(message: Optional[str]) -> dict[str, Any]:
    """Accept + persist an unrecognized name, with a warning."""
    return _verdict(True, True, False, message)


# ── Catalog matching ─────────────────────────────────────────────────────

@dataclass
class _Match:
    exact: bool = False
    suggestion_text: str = ""

    def verdict(self, req: "_Request") -> Optional[dict[str, Any]]:
        """Accept on exact membership, else None so the branch composes its own message."""
        return _accept() if self.exact else None


def _match_in_catalog(
    query: str,
    candidates,
    *,
    case_insensitive: bool = False,
    suggest_query: Optional[str] = None,
    suggest_cutoff: float = 0.5,
    suggest_label: str = "Similar models",
) -> _Match:
    """Shared ladder: exact membership → suggestion text. Never rewrites the id: a requested model
    that is merely CLOSE to a catalog entry is the user's selection (a newer release the listing
    lacks, a dated snapshot, a qualifier) and goes to the wire verbatim — fuzzy "auto-correction"
    swapped `deepseek-v4.1-flash` for `deepseek-v4-flash`, `gemini-3.8-flash` for `gemini-3.6-flash`
    and `model:nitro` for `model` under the user's own label. The vendor's 400 names the valid ids.
    ``case_insensitive`` matches lower-cased ids and maps results back to the catalog's spelling
    (MiniMax ships mixed-case ids). ``suggest_query`` overrides the string the suggestion search
    uses (some branches search on the raw request, not the lookup form)."""
    pool = list(candidates)
    display = None
    if suggest_query is None:
        suggest_query = query
    if case_insensitive:
        display = {c.lower(): c for c in pool}
        pool = list(display)
        query, suggest_query = query.lower(), suggest_query.lower()

    def _show(cid: str) -> str:
        return display[cid] if display is not None else cid

    if query in set(pool):
        return _Match(exact=True)
    suggestions = get_close_matches(suggest_query, pool, n=3, cutoff=suggest_cutoff)
    if not suggestions:
        return _Match()
    return _Match(suggestion_text=f"\n  {suggest_label}: " + ", ".join(f"`{_show(s)}`" for s in suggestions))


# ── Request context ──────────────────────────────────────────────────────

@dataclass
class _Request:
    requested: str
    lookup: str                 # id used for catalog membership (copilot-normalized / preset base)
    provider: Optional[str]     # raw caller value (Ollama checks look at this, not ``normalized``)
    normalized: str
    api_key: Optional[str]
    base_url: Optional[str]
    api_mode: Optional[str]
    headers: Optional[dict[str, str]]


# ── Provider branches (None = not decided here) ─────────────────────────



def _reject_whitespace(req: _Request) -> Optional[dict[str, Any]]:
    if any(ch.isspace() for ch in req.requested):
        return _reject("Model names cannot contain spaces.")
    return None










def _validate_custom(req: _Request) -> dict[str, Any]:
    from hermes_cli import models as _m

    # Probe with the auth shape the api_mode expects.
    anthropic_style = req.api_mode == "anthropic_messages"
    probe_kwargs = {"api_mode": req.api_mode} if anthropic_style else {}
    probe = _m.probe_api_models(req.api_key, req.base_url, request_headers=req.headers, **probe_kwargs)
    api_models = probe.get("models")
    if api_models is not None:
        match = _match_in_catalog(req.lookup, api_models, suggest_query=req.requested)
        verdict = match.verdict(req)
        if verdict is not None:
            return verdict
        message = (
            f"Note: `{req.requested}` was not found in this custom endpoint's model listing "
            f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
            f"{match.suggestion_text}"
        )
        if probe.get("used_fallback"):
            message += (f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                        "Consider saving that as your base URL.")
        return _soft_accept(message)

    # Many OpenAI-compatible and Anthropic-compatible proxies (DashScope coding plan, Cline,
    # MiniMax) never implement GET /models; /chat/completions works fine. Rejecting the switch
    # here bricked `/model` for them (#12220), so both chat modes persist the name unverified.
    accepted = req.api_mode in ("chat_completions", "anthropic_messages")
    message = f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
    if accepted:
        message += (f"`{req.requested}` was accepted without verification — if this endpoint does not "
                    "serve it, inference will fail; check the provider's model catalog or the model name.")
    else:
        message += f"`{req.requested}` was not saved; the endpoint should expose `/models` for verification."
    if probe.get("suggested_base_url"):
        message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"
    return _verdict(accepted, True, False, message)


def _static_catalog(normalized: str) -> list[str]:
    from hermes_cli import models as _m

    try:
        return _m.provider_model_ids(normalized)
    except Exception:
        return []








def _validate_anthropic_messages(req: _Request) -> dict[str, Any]:
    """Anthropic Messages transport: probe /v1/models and soft-accept either way, but say which
    happened — a proxy that never implemented the listing is a different situation from a reachable
    listing that simply doesn't name the slug (vendors alias ids: ``kimi-k3`` is served as ``k3``)."""
    from hermes_cli import models as _m

    models = _m.fetch_api_models(req.api_key, req.base_url, api_mode=req.api_mode)
    if models is None:
        return _soft_accept(
            f"Note: could not verify `{req.requested}` against this endpoint's model listing.  Many "
            "Anthropic-compatible proxies do not implement GET /v1/models.  The model name has been accepted "
            "without verification."
        )
    # Vendor alias pairs sit below the default 0.5 similarity cutoff (kimi-k3 vs k3 ≈ 0.44).
    match = _match_in_catalog(req.lookup, models, case_insensitive=True, suggest_query=req.requested,
                              suggest_cutoff=0.4)
    return match.verdict(req) or _soft_accept(
        f"Note: `{req.requested}` is not named in this endpoint's model listing (it may still serve it "
        f"under an alias).{match.suggestion_text}"
        "\n  The model name has been accepted without verification."
    )


def _nous_portal_recommended_names() -> set[str]:
    """Lower-cased ids from the Portal's live recommended-models feed (empty on any failure)."""
    from hermes_cli import models as _m

    try:
        payload = _m.fetch_nous_recommended_models(_m._resolve_nous_portal_url())
        return {
            name.lower()
            for tier in ("freeRecommendedModels", "paidRecommendedModels")
            for entry in (payload.get(tier) or [])
            if (name := _m._extract_model_name(entry))
        }
    except Exception:
        return set()


def _validate_live_listing(req: _Request) -> Optional[dict[str, Any]]:
    """Generic live /v1/models probe. Returns None when the API was unreachable (the caller then
    tries Bedrock discovery / the curated catalog)."""
    from hermes_cli import models as _m

    api_models = _m.fetch_api_models(req.api_key, req.base_url)
    if api_models is None:
        return None
    if req.normalized == "gemini":
        # Gemini's OpenAI-compat listing prefixes ids with "models/"; curated list and user input
        # use the bare id, so strip before comparing.
        api_models = [m[len("models/"):] if isinstance(m, str) and m.startswith("models/") else m for m in api_models]
    match = _match_in_catalog(req.lookup, api_models)
    if match.exact:
        return _accept()
    # OpenRouter routing variants (":nitro", ":floor", ...) are request-time modifiers, not
    # catalog entries — validate the BASE but keep the suffixed id.
    variant_base = openrouter_variant_base(req.lookup) if req.normalized == "openrouter" else None
    if variant_base is not None and variant_base in set(api_models):
        return _accept()
    # Listed but not found: the account may reach models absent from the public listing
    # (e.g. Z.AI Pro/Max plans use glm-5 on coding endpoints) — warn but allow where plausible.
    # Curated-catalog soft-accept: providers omit valid models from live listings (stale cache,
    # partial rollout, gated previews). EXCEPTION: official OpenAI hosts (canonical + data-
    # residency regional) — their listing is access-scoped and authoritative, so an absent model
    # is one this key CANNOT serve; a soft-accept would 400 at first use. Custom OpenAI-compatible
    # proxies keep the fallback.
    listing_authoritative = False
    if req.normalized in ("openai", "openai-api"):
        from hermes_cli.providers import is_official_openai_host

        listing_authoritative = is_official_openai_host(req.base_url)
    if not listing_authoritative and _m._model_in_provider_catalog(
        (variant_base or req.lookup).lower(), _m._provider_keys(req.normalized)
    ):
        return _accept_with_note(f"Note: `{req.requested}` was not found in the live /v1/models listing "
                                 "but exists in the curated catalog — accepted.")
    # Nous: the Portal's recommended-models feed can list a model before the curated list or the
    # docs-hosted manifest catches up; `hermes chat` already accepts those at model-list build
    # time, so mirror that source of truth for per-message /model validation.
    if req.normalized == "nous" and req.lookup.lower() in _nous_portal_recommended_names():
        return _accept_with_note(f"Note: `{req.requested}` was not found in the live /v1/models listing "
                                 "but is a current Nous Portal recommendation — accepted.")
    return _reject(f"Model `{req.requested}` was not found in this provider's model listing.{match.suggestion_text}")




def _validate_catalog_fallback(req: _Request) -> dict[str, Any]:
    """/models unreachable: validate against the curated ``provider_model_ids()`` list so gateway
    /model switches keep working while a provider's endpoint is down (otherwise switch_model() would
    fail and the gateway never writes the session override). No catalog → accept with a warning."""
    from hermes_cli import models as _m

    label = _m._PROVIDER_LABELS.get(req.normalized, req.normalized)
    catalog = _static_catalog(req.normalized)
    if not catalog:
        return _soft_accept(f"Note: could not reach the {label} API to validate `{req.requested}`. "
                            "If the service isn't down, this model may not be valid.")
    match = _match_in_catalog(req.lookup, catalog, case_insensitive=True)
    if match.exact:
        return _accept()
    # Same OpenRouter routing-variant rule as the live-listing path.
    if req.normalized == "openrouter":
        variant_base = openrouter_variant_base(req.lookup)
        if variant_base is not None and variant_base.lower() in {m.lower() for m in catalog}:
            return _accept()
    return _soft_accept(
        f"Note: `{req.requested}` was not found in the {label} curated catalog "
        f"and the /models endpoint was unreachable.{match.suggestion_text}"
        f"\n  The model may still work if it exists on the provider."
    )


# ── Orchestrator ─────────────────────────────────────────────────────────

def _is_custom(req: _Request) -> bool:
    return req.normalized == "custom" or req.normalized.startswith("custom:")


def _for(*providers: str) -> Callable[[_Request], bool]:
    return lambda req: req.normalized in providers


# Only supported account identities reach the catalog ladder.
_LADDER = (
    (lambda req: True, _reject_whitespace),
    (_is_custom, _validate_custom),
    (lambda req: req.api_mode == "anthropic_messages", _validate_anthropic_messages),
    (lambda req: True, _validate_live_listing),
    (lambda req: True, _validate_catalog_fallback),
)

def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Validate a ``/model`` value for the active provider → dict with ``accepted`` (switch now),
    ``persist`` (safe to save to config), ``recognized`` (matched a known provider catalog),
    ``message`` (optional warning / guidance). The requested id is never rewritten: what the user
    selected is what the wire sees."""
    from hermes_cli import models as _m

    requested = (model_name or "").strip()
    from hermes_cli.provider_policy import require_supported_provider
    normalized = require_supported_provider(provider or "")
    lookup = requested
    if normalized == "copilot":
        lookup = _m.normalize_copilot_model_id(requested, api_key=api_key) or requested

    if not requested:
        return _reject("Model name cannot be empty.")
    req = _Request(requested, lookup, provider, normalized, api_key, base_url, api_mode, headers)
    for gate, branch in _LADDER:
        if gate(req):
            verdict = branch(req)
            if verdict is not None:
                return verdict
    raise AssertionError("unreachable: _validate_catalog_fallback always decides")
