"""URL normalization shared by Gemini speech service tools, not inference routing."""
import re
from typing import Optional
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_API_VERSION_SEGMENT = re.compile(r"^v\d+(?:alpha|beta)?\d*$", re.IGNORECASE)


def normalize_gemini_base_url(base_url: Optional[str]) -> str:
    """Gemini native base URL with the API version segment guaranteed. Google's own client treats the
    base as a host root and appends the version itself, so users configure ``GEMINI_BASE_URL`` (or a
    proxy root like ``http://localhost:4000/gemini``) that way; our request builders expect
    ``{base}/models/{model}:generateContent`` — without ``/v1beta`` that is a guaranteed 404. Trailing
    slashes and an ``/openai`` suffix are stripped; an existing version segment (``v1``, ``v1beta``,
    ``v1alpha``, ...) is kept; empty input returns ``DEFAULT_GEMINI_BASE_URL``. Only the LAST path
    segment is inspected, so ``.../v1beta/extra`` still gets ``/v1beta`` appended; this does not
    decide routing (see ``is_native_gemini_base_url``)."""
    trimmed = str(base_url or "").strip().rstrip("/")
    trimmed = re.sub(r"/openai\Z", "", trimmed, flags=re.IGNORECASE).rstrip("/")
    if not trimmed:
        return DEFAULT_GEMINI_BASE_URL
    if _API_VERSION_SEGMENT.match(trimmed.rsplit("/", 1)[-1]):
        return trimmed
    return f"{trimmed}/v1beta"
