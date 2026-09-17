"""Provider/model inventory — shared substrate for the dashboard ``/api/model/options``, the TUI
``model.options``/``model.save_key`` RPC handlers, and the interactive picker."""

from __future__ import annotations

from contextvars import copy_context
from dataclasses import dataclass, replace
from threading import Lock, Thread, current_thread
from typing import Any, Optional

_pricing_prewarm_lock = Lock()
_pricing_prewarm_threads: dict[tuple[str, tuple[tuple[str, str], ...]], Thread] = {}


@dataclass(frozen=True)
class ConfigContext:
    """Disk-config snapshot (``load_picker_context()``); the TUI overlays live agent state via
    ``with_overrides()``."""

    current_provider: str
    current_model: str
    current_base_url: str
    user_providers: dict
    custom_providers: list
    excluded_providers: list = None

    def with_overrides(
        self, *, current_provider: Optional[str] = None, current_model: Optional[str] = None,
        current_base_url: Optional[str] = None,
    ) -> "ConfigContext":
        """Copy with TRUTHY overrides applied: the TUI reads agent attributes that may be empty strings
        before an agent is spawned — empties must not clobber the disk-config values."""
        overrides = (("current_provider", current_provider), ("current_model", current_model),
                     ("current_base_url", current_base_url))
        kw = {k: v for k, v in overrides if v}
        return replace(self, **kw) if kw else self


def load_picker_context() -> ConfigContext:
    """Load the disk-config snapshot every consumer needs."""
    from hermes_cli.config import (
        coerce_provider_id, get_compatible_custom_providers, load_config, stringify_provider_map,
    )
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        # PyYAML parses unquoted scalars as int (`provider: 2070`); keep strings so picker/options
        # paths never call `.strip()` on an int.
        current_model = str(model_cfg.get("default", model_cfg.get("name", "")) or "")
        current_provider = coerce_provider_id(model_cfg.get("provider", ""))
        current_base_url = str(model_cfg.get("base_url", "") or "")
    else:  # config.model can be a bare string in older configs
        current_model, current_provider, current_base_url = (str(model_cfg) if model_cfg else ""), "", ""
    excluded = cfg.get("model_catalog", {}).get("excluded_providers") or []
    return ConfigContext(
        current_provider=current_provider, current_model=current_model, current_base_url=current_base_url,
        user_providers=stringify_provider_map(cfg.get("providers")),
        custom_providers=get_compatible_custom_providers(cfg),
        excluded_providers=excluded if isinstance(excluded, list) else [],
    )


def _slug(row: dict) -> str:
    return str(row.get("slug") or "").strip().lower()


def _without_slug(rows: list[dict], slug: str) -> list[dict]:
    return [r for r in rows if _slug(r) != slug]


# ─── Public: payload builder ────────────────────────────────────────────


def build_models_payload(
    ctx: ConfigContext, *, explicit_only: bool = False, include_unconfigured: bool = False,
    picker_hints: bool = False, canonical_order: bool = False, pricing: bool = False,
    pricing_cache_only: bool = False,
    capabilities: bool = False, featured: bool = False, force_fresh_nous_tier: bool = False,
    refresh: bool = False, probe_custom_providers: bool = True, probe_current_custom_provider: bool = False,
    for_picker: bool = False, max_models: int | None = None,
) -> dict:
    """Build the ``{providers, model, provider}`` shape every consumer needs. ``explicit_only`` keeps
    only providers the user explicitly configured — hides ambient/auto-seeded credentials from
    desktop chat pickers. ``pricing_cache_only``: with ``pricing``, use only values already resident
    in process caches (normal picker opens, while a background worker warms cold endpoints)."""
    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers(
        current_provider=ctx.current_provider, current_base_url=ctx.current_base_url,
        current_model=ctx.current_model, user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers, force_fresh_nous_tier=force_fresh_nous_tier,
        max_models=max_models, refresh=refresh, probe_custom_providers=probe_custom_providers,
        probe_current_custom_provider=probe_current_custom_provider, for_picker=for_picker,
        excluded_providers=ctx.excluded_providers or [],
    )

    if explicit_only:
        rows = _filter_explicit_provider_rows(rows, ctx)
        rows = list(rows) + _append_unconfigured_rows(rows, ctx, current_only=True)

    # A local proxy serving a model also in an aggregator's catalog would show under both, and picking
    # the aggregator row silently breaks the call — aggregators only list models no specific provider has.
    _strip_aggregator_overlaps(rows)

    if include_unconfigured:
        rows = list(rows) + _without_slug(_append_unconfigured_rows(rows, ctx), "moa")
    if picker_hints:
        _apply_picker_hints(rows)
    if canonical_order:
        rows = _reorder_canonical(rows)
    if pricing:
        _apply_pricing(rows, force_fresh_nous_tier=force_fresh_nous_tier, cached_only=pricing_cache_only)
    if capabilities:
        _apply_capabilities(rows)
    if featured:
        _apply_featured(rows)
    _apply_custom_aliases(rows)

    return {"providers": rows, "model": ctx.current_model, "provider": ctx.current_provider}


def _strip_aggregator_overlaps(rows: list[dict]) -> None:
    """Drop models from TRUE routing aggregators (OpenRouter, custom:* proxies) that a user-defined
    provider also serves. The is_user_defined guard matters: is_routing_aggregator() is True for every
    custom:* slug, so without it the dedup would empty a user's own custom row. Flat-namespace
    resellers (opencode-go/zen) serve every model first-party and keep shared names."""
    try:
        from hermes_cli.providers import is_routing_aggregator
    except Exception:
        return

    user_models: set[str] = set()
    for row in rows:
        if row.get("is_user_defined"):
            user_models.update(m.lower() for m in (row.get("models") or []))
    if not user_models:
        return
    for row in rows:
        if row.get("is_user_defined") or not is_routing_aggregator(row.get("slug", "")):
            continue
        # Only strip overlaps from TRUE routing aggregators (OpenRouter, custom:* proxies). Flat-namespace
        # resellers (opencode-go / opencode-zen) serve every listed model as a first-party model, so their
        # rows must keep models that a user's proxy happens to share a name with — otherwise a subscription
        # provider's own catalog (minimax-m3, glm-5, deepseek-v4-flash, ...) is silently gutted in the
        # picker. (#47077)
        original = row.get("models") or []
        filtered = [m for m in original if m.lower() not in user_models]
        if len(filtered) < len(original):
            row["models"] = filtered
            row["total_models"] = len(filtered)


def build_model_options_payload(
    ctx: ConfigContext, *, explicit_only: bool = False, include_unconfigured: bool = False,
    refresh: bool = False,
) -> dict:
    """Shared API-server/dashboard/TUI payload. Normal open probes only the current custom provider so
    offline saved endpoints don't block the picker; explicit refresh probes all and busts the cache."""
    refresh = bool(refresh)
    payload = build_models_payload(
        ctx, explicit_only=bool(explicit_only), include_unconfigured=bool(include_unconfigured),
        picker_hints=True, canonical_order=True, pricing=True, pricing_cache_only=not refresh,
        capabilities=True, featured=True,
        refresh=refresh, probe_custom_providers=refresh, probe_current_custom_provider=not refresh,
    )
    if not refresh:
        _prewarm_pricing_async(payload["providers"], current_provider=ctx.current_provider,
                               current_base_url=ctx.current_base_url)
    return payload


# ─── Public: auxiliary-task pickers ─────────────────────────────────────


def build_aux_picker_rows(
    *, current_provider: str = "", current_model: str = "", current_base_url: str = "",
    max_models: int | None = None,
) -> list[dict]:
    """Provider rows for any auxiliary-task picker (vision, compression, …). Honours
    ``excluded_providers``; exhausted-pool providers stay visible (``for_picker``); only the active
    custom endpoint is probed. ``moa`` is excluded: auxiliary_client unwraps it to its aggregator
    anyway, so offering it would be a choice silently rewritten.

    Aux pickers kept re-deriving their own kwargs and each one silently dropped a different slice of the
    user's configuration. Two independent contributor PRs landed against the same two call sites for exactly
    this: 52642 (user ``providers:`` / ``custom_providers:`` entries never appeared) and #66624 (providers
    with an exhausted credential pool were hidden). Both were per-site kwarg patches, so the next aux picker
    would have reintroduced the same gap. Routing through one function makes the correct behaviour the
    default that a new caller cannot forget:
    """
    ctx = load_picker_context().with_overrides(
        current_provider=current_provider, current_model=current_model, current_base_url=current_base_url,
    )
    rows = build_models_payload(
        ctx, for_picker=True, probe_custom_providers=False, probe_current_custom_provider=True,
        max_models=max_models,
    )["providers"]
    return _without_slug(rows, "moa")


def format_aux_picker_entries(
    rows: list[dict], *, current_provider: str = "", current_base_url: str = "",
) -> list[tuple[str, str, list[str]]]:
    """Render aux-picker rows as ``(slug, label, models)``. A raw ``base_url`` custom endpoint is
    "current" only through that URL, never a slug — so with ``current_base_url`` set no row is marked."""
    entries: list[tuple[str, str, list[str]]] = []
    current_slug = str(current_provider or "").strip().lower()
    has_base_url = bool(str(current_base_url or "").strip())
    for row in rows:
        slug = str(row.get("slug") or "")
        name = row.get("name") or slug
        total = row.get("total_models") or len(row.get("models") or [])
        model_hint = f" — {total} models" if total else ""
        marker = "  ← current" if slug.lower() == current_slug and current_slug and not has_base_url else ""
        entries.append((slug, f"{name}{model_hint}{marker}", list(row.get("models") or [])))
    return entries


def _reasoning_catalog_reader(slug: str):
    """Per-model reasoning-capability reader for aggregators that publish one. Cache-only — the picker
    must never block on HTTP; a cold cache warms in the background and reports no restriction until then."""
    try:
        from hermes_cli.models_reasoning_caps import (
            nous_model_reasoning_capabilities,
            openrouter_model_reasoning_capabilities,
            warm_nous_reasoning_caps_async,
            warm_openrouter_reasoning_caps_async,
        )
    except Exception:
        return None

    readers = {
        "nous": (warm_nous_reasoning_caps_async, nous_model_reasoning_capabilities),
        "openrouter": (warm_openrouter_reasoning_caps_async, openrouter_model_reasoning_capabilities),
    }
    if slug not in readers:
        return None
    warm, read = readers[slug]
    warm()
    return read


def _apply_capabilities(rows: list[dict]) -> None:
    """Attach ``{model: {fast, reasoning, ...}}`` per row. ``reasoning`` defaults True when the catalog is
    silent (the dial is a no-op on models that ignore it; hiding it from a capable model is worse). A
    serving aggregator's detail overrides models.dev (adds ``can_disable_reasoning``). ``supported_efforts``
    is deliberately NOT forwarded — it under-reports levels that work."""
    from hermes_cli.models import model_supports_fast_mode

    try:
        from agent.models_dev import get_model_capabilities
    except Exception:
        get_model_capabilities = None  # type: ignore[assignment]

    for row in rows:
        slug = row.get("slug") or ""
        caps: dict[str, dict[str, Any]] = {}
        read_reasoning_catalog = _reasoning_catalog_reader(slug.lower())

        for model in row.get("models") or []:
            reasoning = True
            if get_model_capabilities is not None and slug:
                try:
                    meta = get_model_capabilities(slug, model)
                    if meta is not None:
                        reasoning = bool(meta.supports_reasoning)
                except Exception:
                    reasoning = True

            entry: dict[str, Any] = {"fast": bool(model_supports_fast_mode(model)), "reasoning": reasoning}

            if reasoning and read_reasoning_catalog is not None:
                try:
                    detail = read_reasoning_catalog(model)
                except Exception:
                    detail = None
                if detail and not detail.get("supports_reasoning"):
                    # Aggregator catalog beats models.dev for a route it serves: no reasoning param
                    # means no reasoning controls, so no disable to describe either.
                    entry["reasoning"] = False
                elif detail:
                    entry["can_disable_reasoning"] = not detail.get("mandatory")

            caps[model] = entry

        row["capabilities"] = caps


# Newest N models per lab an aggregator row features by default (older tail behind search/show-all);
# 5 keeps a lab's headliners without letting a prolific vendor flood the view.
_FEATURED_PER_LAB = 5


def _apply_featured(rows: list[dict]) -> None:
    """Attach a ``featured_models`` shortlist to each aggregator row: newest ``_FEATURED_PER_LAB`` per
    vendor by models.dev ``release_date`` (ranked within the row, never vs. today, so it is stable);
    ties keep curated order. Non-aggregators get an empty list and keep top-N behaviour."""
    try:
        from agent.models_dev import get_model_info
    except Exception:
        get_model_info = None  # type: ignore[assignment]

    for row in rows:
        slug = str(row.get("slug") or "").strip().lower()
        models = row.get("models") or []

        by_lab: dict[str, list[tuple[int, str, str]]] = {}  # only multi-lab aggregators get a shortlist
        for pos, model in enumerate(models):
            lab = model.split("/", 1)[0] if "/" in model else ""
            if not lab:  # no vendor prefix → single-namespace provider, not an aggregator
                by_lab = {}
                break
            date = ""
            if get_model_info is not None:
                info = get_model_info(slug, model) or get_model_info("openrouter", model)
                date = getattr(info, "release_date", "") if info else ""
            by_lab.setdefault(lab, []).append((pos, date, model))

        if len(by_lab) < 2:
            row["featured_models"] = []
            continue

        featured: list[str] = []
        for entries in by_lab.values():
            # Newest release_date first; earlier list position breaks ties (sole key when undated).
            ranked = sorted(entries, key=lambda e: (e[1], -e[0]), reverse=True)
            featured.extend(model for _pos, _date, model in ranked[:_FEATURED_PER_LAB])
        order = {m: i for i, m in enumerate(models)}  # keep the row's model order for stable rendering
        row["featured_models"] = sorted(featured, key=lambda m: order[m])


def _apply_custom_aliases(rows: list[dict]) -> None:
    """Attach the accepted identity set to each user-defined row: ``model.options`` reports the canonical
    ``custom:<key>`` while rows carry the bare key as ``slug``, so GUI exact-match never finds the row.

    GUI pickers compare the two to decide which row is active; exact equality never matches for custom
    providers (#87035). Exposing ``aliases`` — every current and legacy spelling from
    :func:`hermes_cli.providers.custom_provider_aliases` — lets the frontend do a membership check instead.
    """
    from hermes_cli.providers import custom_provider_aliases

    for row in rows:
        if not row.get("is_user_defined"):
            continue
        try:
            row["aliases"] = sorted(
                custom_provider_aliases(str(row.get("name", "")), str(row.get("slug", ""))))
        except Exception:
            continue


# ─── Internal: row post-processing ──────────────────────────────────────


def _provider_auth_hint(slug: str) -> tuple[str, str]:
    """``(auth_type, key_env)`` for a canonical provider (``("api_key", "")`` when unregistered)."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get(slug)
    auth_type = cfg.auth_type if cfg else "api_key"
    key_env = cfg.api_key_env_vars[0] if (cfg and cfg.api_key_env_vars) else ""
    return auth_type, key_env


def _row(slug: str, name: str, is_current: bool, **extra: Any) -> dict:
    return {"slug": slug, "name": name, "is_current": is_current, "is_user_defined": False, **extra}


def _canonical_row(entry, cur: str, **extra: Any) -> dict:
    from hermes_cli.models import _PROVIDER_LABELS

    return _row(entry.slug, _PROVIDER_LABELS.get(entry.slug, entry.label), entry.slug.lower() == cur, **extra)


def _append_unconfigured_rows(
    rows: list[dict], ctx: ConfigContext, *, current_only: bool = False,
) -> list[dict]:
    """Empty setup skeletons for canonical providers missing from ``rows`` — except the *current* one:
    if config.yaml still points at it but credentials are gone, keep a row carrying the saved model so
    GUI pickers don't silently snap to another provider."""
    from hermes_cli.models import CANONICAL_PROVIDERS, _model_requires_account_discovery

    seen = {r["slug"].lower() for r in rows}
    cur = (ctx.current_provider or "").lower()
    cur_model = str(ctx.current_model or "").strip()
    extras: list[dict] = []
    for entry in CANONICAL_PROVIDERS:
        if entry.slug.lower() in seen:
            continue
        if current_only and entry.slug.lower() != cur:
            continue
        if entry.slug.lower() == cur:
            saved_model = "" if _model_requires_account_discovery(entry.slug, cur_model) else cur_model
            auth_type, key_env = _provider_auth_hint(entry.slug)
            tail = (
                "Astra requires successful account-scoped model discovery."
                if cur_model and not saved_model else "Showing the saved model only."
            )
            warning = (
                f"Configured provider missing usable credentials; paste {key_env} to reactivate. {tail}"
                if auth_type == "api_key" and key_env
                else f"Configured provider is not authenticated; run `hermes model` to reactivate. {tail}"
            )
            extras.append(_canonical_row(
                entry, cur, models=[saved_model] if saved_model else [], total_models=1 if saved_model else 0,
                source="configured-current", authenticated=False, auth_type=auth_type, key_env=key_env,
                warning=warning,
            ))
            continue
        extras.append(_canonical_row(entry, cur, models=[], total_models=0, source="canonical"))
    return extras




def _filter_explicit_provider_rows(rows: list[dict], ctx: ConfigContext) -> list[dict]:
    """Keep only rows backed by explicit user configuration — ``list_authenticated_providers`` also
    discovers ambient credentials (e.g. GitHub CLI -> Copilot) Desktop chat pickers must not show."""
    from hermes_cli.auth import is_provider_explicitly_configured

    current_slug = str(ctx.current_provider or "").strip().lower()

    def _is_explicit(row: dict, slug: str) -> bool:
        # Managed local models are explicit configuration by existence (gigabytes downloaded into the
        # machine-scoped dir); there is deliberately no config credential, so without the source clause
        # the row would only survive on the profile where Use was last clicked.
        if (row.get("is_user_defined") or (current_slug and slug == current_slug)
                or row.get("source") == "local-runtime"):
            return True
        if slug == "moa":
            # Virtual routing mode, not a configured provider: hide unless current (above) or the user
            # wrote an enabled preset into RAW config (the DEFAULT_CONFIG preset must not show MoA).
            return _raw_config_has_enabled_moa_preset()
        return (
            _provider_is_keyless(slug)  # zero-setup providers need no configuration at all
            # Anthropic OAuth (device flow / Claude Code) and external-process CLIs (copilot-acp) are
            # deliberate sign-ins that leave no trace in config/env; keep the rows discovery accepted.
            or is_provider_explicitly_configured(slug)
        )

    return [row for row in rows
            if (slug := str(row.get("slug", "")).strip().lower()) and _is_explicit(row, slug)]




def _provider_is_keyless(slug: str) -> bool:
    """True when the provider's Hermes overlay declares it keyless."""
    try:
        from hermes_cli.providers import HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS.get(slug)
        return bool(overlay is not None and getattr(overlay, "keyless", False))
    except Exception:
        return False




def _apply_picker_hints(rows: list[dict]) -> None:
    """Add ``authenticated``/``auth_type``/``key_env``/``warning`` per row."""
    for row in rows:
        if "authenticated" in row:
            continue
        # Skeleton rows (_append_unconfigured_rows) have empty `models` AND source="canonical".
        is_skeleton = row.get("source") == "canonical" and not row.get("models")
        row["authenticated"] = not is_skeleton
        if not is_skeleton or row.get("is_user_defined"):
            continue
        auth_type, key_env = _provider_auth_hint(row["slug"])
        row["auth_type"] = auth_type
        row["key_env"] = key_env
        row["warning"] = (f"paste {key_env} to activate" if auth_type == "api_key" and key_env
                          else f"run `hermes model` to configure ({auth_type})")


def _reorder_canonical(rows: list[dict]) -> list[dict]:
    """Canonical slugs in ``CANONICAL_PROVIDERS`` order, truly-custom rows last. Keys on slug membership,
    NOT ``is_user_defined`` — ``providers:`` config rows carry that flag even for canonical slugs."""
    from hermes_cli.models import CANONICAL_PROVIDERS

    order = {e.slug: i for i, e in enumerate(CANONICAL_PROVIDERS)}
    canon = sorted((r for r in rows if r["slug"] in order), key=lambda r: order[r["slug"]])
    extras = [r for r in rows if r["slug"] not in order]
    return canon + extras


def _apply_pricing(rows: list[dict], *, force_fresh_nous_tier: bool = False, cached_only: bool = False) -> None:
    """Set ``row["pricing"] = {model_id: {input, output, cache | None, free}}``; for Nous also
    ``free_tier`` (account is free-tier) and ``unavailable_models`` (paid models a free user can't pick).
    ``cached_only`` never hits the network: unknown Nous entitlement fails closed (``free_tier_pending``,
    all models locked) and missing pricing is marked ``pricing_pending``."""
    from hermes_cli.models_pricing import (
        _format_price_per_mtok,
        compute_sale_discount,
        get_pricing_for_provider,
    )
    from hermes_cli.models import (
        check_nous_free_tier,
        get_cached_nous_free_tier,
        partition_nous_models_by_tier,
    )

    nous_free_tier: Optional[bool] = None  # resolved once (cached in models.py for the TTL window)

    for row in rows:
        slug = str(row.get("slug", "")).lower()
        models = row.get("models") or []
        if not models:
            continue
        if row.get("free_tier_row"):
            # The free tier's one model has no Portal pricing and no entitlement to read: pricing
            # it would lock the only row a free-tier install can select.
            continue
        try:
            pricing_kwargs = {"cached_only": True} if cached_only else {}
            raw_pricing = get_pricing_for_provider(slug, **pricing_kwargs) or {}
        except Exception:
            raw_pricing = {}
        cached_nous_tier: Optional[bool] = None
        if slug == "nous" and cached_only:
            cached_nous_tier = get_cached_nous_free_tier()
            if cached_nous_tier is None:
                # Entitlement unknown: stay nonblocking but fail closed until the prewarm has populated
                # both caches, else a free account could briefly select paid models on first open.
                row["free_tier_pending"] = True
                row["unavailable_models"] = list(models)
                if not row.get("warning"):  # say why every model renders locked
                    row["warning"] = ("Checking Nous plan entitlement… models unlock on the "
                                      "next picker open or refresh.")
                continue
        if not raw_pricing:
            if slug == "nous":
                row["free_tier"] = bool(cached_nous_tier)
                row["pricing_pending"] = True
                row["unavailable_models"] = list(models) if cached_nous_tier else []
            continue

        formatted: dict[str, dict] = {}
        for mid in models:
            p = raw_pricing.get(mid)
            if not p:
                continue
            inp_raw, out_raw = p.get("prompt", ""), p.get("completion", "")
            cache_raw = p.get("input_cache_read", "")
            inp = _format_price_per_mtok(inp_raw) if inp_raw != "" else ""
            out = _format_price_per_mtok(out_raw) if out_raw != "" else ""
            entry: dict = {
                "input": inp, "output": out,
                "cache": _format_price_per_mtok(cache_raw) if cache_raw else None,
                "free": inp == "free" and out in ("free", ""),  # both input and output cost nothing
            }
            # Sale chrome is Nous Portal-only (other catalogs' nested pricing.original is ignored); free
            # models get flat -100% chrome, was_* only when the gateway served an original.
            if slug == "nous":
                sale = compute_sale_discount(inp_raw, out_raw, p.get("original"))
                if sale is not None:
                    discount_percent, was_prompt_raw, was_out_raw = sale
                    entry["discount_percent"] = discount_percent
                    for key, was_raw in (("was_input", was_prompt_raw), ("was_output", was_out_raw)):
                        if was_raw != "":
                            entry[key] = _format_price_per_mtok(was_raw)
            formatted[mid] = entry

        if formatted:
            row["pricing"] = formatted

        if slug == "nous":
            try:
                if nous_free_tier is None:
                    nous_free_tier = (cached_nous_tier if cached_only
                                      else check_nous_free_tier(force_fresh=force_fresh_nous_tier))
                row["free_tier"] = bool(nous_free_tier)
                row["unavailable_models"] = (
                    partition_nous_models_by_tier(list(models), raw_pricing, free_tier=True)[1]
                    if nous_free_tier else [])
            except Exception:  # tier detection failed — fail open (no gating)
                row["free_tier"] = False
                row["unavailable_models"] = []




def _prewarm_pricing_async(
    rows: list[dict], *, current_provider: str = "", current_base_url: str = "",
) -> Optional[Thread]:
    """Warm picker pricing caches without delaying the current payload (one worker per
    profile + endpoint scope; a live worker is reused)."""
    from hermes_constants import hermes_home_key
    from hermes_cli.models_pricing import pricing_cache_scope

    slugs = {str(row.get("slug") or "").lower() for row in rows if row.get("slug")}
    endpoint_scope = tuple(sorted(
        (slug, pricing_cache_scope(slug, current_provider=current_provider, current_base_url=current_base_url))
        for slug in slugs))
    prewarm_key = (hermes_home_key(), endpoint_scope)

    with _pricing_prewarm_lock:
        current = _pricing_prewarm_threads.get(prewarm_key)
        if current is not None and current.is_alive():
            return current
        # The worker mutates only private copies; the pricing helpers populate shared process caches.
        worker_rows = [{**row, "models": list(row.get("models") or [])} for row in rows]

        def _worker() -> None:
            try:
                _apply_pricing(worker_rows)
            finally:
                with _pricing_prewarm_lock:
                    if _pricing_prewarm_threads.get(prewarm_key) is current_thread():
                        _pricing_prewarm_threads.pop(prewarm_key, None)

        thread = Thread(target=copy_context().run, args=(_worker,),
                        name="hermes-picker-pricing-prewarm", daemon=True)
        _pricing_prewarm_threads[prewarm_key] = thread
        thread.start()
        return thread
