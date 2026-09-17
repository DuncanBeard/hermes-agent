"""Supported model account identities, independent of config and wire protocols.

Selectors (``auto``/None) and configured bare endpoint names must be resolved
before applying this boundary. A custom prefix does not prove an endpoint exists.
"""

SUPPORTED_BUILTIN_PROVIDERS = frozenset({"copilot", "nous", "custom"})


class UnsupportedProviderError(ValueError):
    """An explicit model provider is not supported by this build."""


def is_supported_provider_id(value: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in SUPPORTED_BUILTIN_PROVIDERS or (
        normalized.startswith("custom:") and bool(normalized.partition(":")[2].strip())
    )


def require_supported_provider(value: str) -> str:
    if not is_supported_provider_id(value):
        raise UnsupportedProviderError(
            f"Unsupported model provider {value!r}. Choose Copilot, Nous, or a custom endpoint "
            "with 'hermes model'; configure an endpoint explicitly to use custom:<name>. "
            "Existing credentials and history have not been changed."
        )
    return value.strip().lower()
