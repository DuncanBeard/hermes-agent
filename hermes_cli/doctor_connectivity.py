"""API connectivity probes for ``hermes doctor`` (split out of ``doctor.py``).

Every probe is a pure function: one HTTP/SDK call returning a ``ProbeResult`` with the row(s) to
print and issue strings to append. No printing inside workers — the caller prints in submission order.
"""

from __future__ import annotations

import concurrent.futures
import functools
from typing import NamedTuple

from hermes_cli.colors import Colors, color



class ProbeResult(NamedTuple):
    label: str
    lines: list  # [(glyph, label, detail)]
    issues: list


_GLYPH = {"ok": ("✓", Colors.GREEN), "warn": ("⚠", Colors.YELLOW), "fail": ("✗", Colors.RED)}


def _row(name: str, status: str, detail: str = "", issues: list | None = None, label: str | None = None) -> ProbeResult:
    glyph, col = _GLYPH[status]
    return ProbeResult(name, [(color(glyph, col), name if label is None else label, color(detail, Colors.DIM) if detail else "")],
                       list(issues or []))


def _skip(name: str) -> ProbeResult:
    return ProbeResult(name, [], [])


def _probe_provider(descriptor) -> ProbeResult:
    from hermes_cli.auth import is_provider_explicitly_configured
    from hermes_cli.config import get_env_value_prefer_dotenv, load_config
    from hermes_cli.models import copilot_default_headers, fetch_api_models
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent.command_token_source import materialize_probe_api_key

    name, slug = descriptor.label, descriptor.slug
    if slug == "custom":
        model = load_config().get("model") or {}
        selected = str(model.get("provider") or "") if isinstance(model, dict) else ""
        if selected.startswith("custom:"):
            slug = selected
    # Never provision a guest or borrow ambient credentials merely to run doctor.
    if not (is_provider_explicitly_configured(slug) or any(
            get_env_value_prefer_dotenv(key) for key in descriptor.api_key_env_vars)):
        return _skip(name)
    try:
        runtime = resolve_runtime_provider(requested=slug)
        headers = {**(runtime.get("default_headers") or {}), **(runtime.get("extra_headers") or {})}
        if slug == "copilot":
            headers = {**copilot_default_headers(), **headers}
        models = fetch_api_models(
            materialize_probe_api_key(runtime.get("api_key")), runtime.get("base_url"),
            timeout=10, api_mode=runtime.get("api_mode"), headers=headers or None)
        if models is not None:
            return _row(name, "ok", f"({len(models)} models)")
        return _row(name, "warn", "(couldn't verify model endpoint)")
    except Exception as exc:
        return _row(name, "warn", f"({exc})", [f"Check {name} configuration"])


def build_probes() -> list:
    """Use the same supported provider catalog as setup and the model picker."""
    from hermes_cli.provider_catalog import provider_catalog
    return [(d.label, functools.partial(_probe_provider, d)) for d in provider_catalog()]


def run_probes(probes: list) -> list:
    """Run independent probes concurrently, preserving display order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="doctor-probe") as ex:
        return [f.result() for f in [ex.submit(fn) for _, fn in probes]]
