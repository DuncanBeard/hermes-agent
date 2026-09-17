"""Shared Responses-API streaming runtime for Copilot and custom endpoints."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current
from agent.sdk_transform_bypass import bypass_sdk_request_transform
from agent.usage_anchor import set_usage_anchor

logger = logging.getLogger(__name__)
_codex_watchdog_state_var: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "codex_watchdog_state", default=None
)


def _call_guarded(fn: Callable | None, fail_msg: str, *fail_args: Any, args: tuple = (), kwargs: dict | None = None):
    """Invoke an optional display/debug callback; a buggy hook must never tear down the turn."""
    if fn is None:
        return
    try:
        fn(*args, **(kwargs or {}))
    except Exception:
        logger.debug(fail_msg, *fail_args, exc_info=True)


def _codex_request_failure_details(error: BaseException) -> tuple[int | None, str]:
    """(serialized request bytes, exception class chain); the buffered ``httpx.Request`` content
    on OpenAI connection errors gives the exact byte count without logging payloads or URLs."""
    request_body_bytes: int | None = None
    exception_classes: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        exception_classes.append(type(current).__name__)
        if request_body_bytes is None:
            content = None
            with suppress(Exception):
                content = getattr(getattr(current, "request", None), "content", None)
            if isinstance(content, str):
                request_body_bytes = len(content.encode("utf-8"))
            elif isinstance(content, (bytes, bytearray, memoryview)):
                request_body_bytes = len(content)
        implicit_chain = current.__cause__ is None and not current.__suppress_context__
        current = current.__context__ if implicit_chain else current.__cause__
    return request_body_bytes, " <- ".join(exception_classes)


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """Field access for attr-style (SDK objects) and dict (raw JSON) events/items."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


_CODEX_PROGRESS_DELTA_TYPES = frozenset({
    "response.output_text.delta", "response.reasoning_summary_text.delta", "response.text.delta",
    "response.audio.delta", "response.function_call_arguments.delta", "response.reasoning_text.delta",
    "response.refusal.delta",
})


def _codex_event_has_content(event: Any) -> bool:
    """Whether a Codex Responses event carries substantive forward progress.

    Lifecycle/keepalive frames and empty structural deltas prove transport
    liveness, but do not mean the model has begun producing its response.
    """
    event_type = _event_field(event, "type")
    if event_type in _CODEX_PROGRESS_DELTA_TYPES:
        return bool(_event_field(event, "delta"))
    if event_type == "response.output_item.added":
        item = _event_field(event, "item")
        return "function_call" in str(_event_field(item, "type") or "") and any(
            bool(_event_field(item, field)) for field in ("id", "call_id", "name", "arguments"))
    return False


def _raise_stream_error(event: Any) -> None:
    """Raise ``_StreamErrorEvent`` from a ``type=error`` SSE frame. The spec puts code/message/param at the
    top level, but the SDK and several proxies nest them under ``error``; read top-level first, then the envelope."""
    from run_agent import _StreamErrorEvent
    nested = _event_field(event, "error")

    def _error_field(name: str) -> Any:
        value = _event_field(event, name)
        return _event_field(nested, name) if value is None and nested is not None else value
    raw_message = _error_field("message")
    message = (str(raw_message) if raw_message is not None else "stream emitted error event").strip() or "stream emitted error event"
    raise _StreamErrorEvent(message, code=_error_field("code"), param=_error_field("param"))


def _message_phase(item: Any) -> str | None:
    phase = _event_field(item, "phase", None)
    return phase.strip().lower() if isinstance(phase, str) else None


def _output_text_of(item: Any) -> str:
    """Concatenated ``output_text`` parts of a message item ("" if content is not a list)."""
    content_parts = _event_field(item, "content", [])
    parts = content_parts if isinstance(content_parts, list) else []
    return "".join(
        str(_event_field(part, "text", "") or "") for part in parts if _event_field(part, "type", "") == "output_text"
    ).strip()


class _CodexResponseAssembler:
    """Assemble a Response-shaped ``SimpleNamespace`` from raw Responses SSE events.

    Only ``usage`` / ``status`` / ``id`` are read from the terminal frame — never ``response.output``. Output
    items come from ``output_item.done``, or are synthesized from text deltas, or settled from function calls
    announced via ``output_item.added`` but never confirmed (some backends omit per-item done events on success)."""

    has_tool_calls = first_delta_fired = saw_terminal = False
    next_output_sequence = 0
    active_message_phase: str | None = None
    # Reasoning summary parts carry no separator; a summary_index change is where the blank line belongs.
    active_summary_index: Any = None
    terminal_status: str = "completed"
    terminal_usage = terminal_response_id = terminal_incomplete_details = terminal_error = None
    # terminal_status defaults to "completed", so settlement needs an explicitly observed response.completed frame.
    saw_response_completed = False

    def __init__(self, *, model, on_text_delta, on_reasoning_delta, on_commentary_message, on_first_delta):
        self.model, self.on_text_delta, self.on_reasoning_delta = model, on_text_delta, on_reasoning_delta
        self.on_commentary_message, self.on_first_delta = on_commentary_message, on_first_delta
        self.output_items: List[Any] = []
        # output_index / first-observed sequence per output item, in lockstep, so settled pending calls merge
        # back in stream order.
        self.output_indexes, self.output_sequences = [], []
        self.text_deltas, self.commentary_text_deltas = [], []
        # pending_function_calls: announced-but-unconfirmed function calls keyed by item id. announced_output_order:
        # first-observed (sequence, output_index) per announced item id so a later .done keeps its announced position.
        self.pending_function_calls: Dict[str, Dict[str, Any]] = {}
        self.announced_output_order: Dict[str, tuple] = {}

    def _safe(self, cb: Callable | None, label: str, *args: Any) -> None:
        _call_guarded(cb, f"Codex stream {label} raised", args=args)

    def _on_item_added(self, event: Any, event_type: str) -> None:
        item = _event_field(event, "item")
        item_type = _event_field(item, "type", "")
        self.active_message_phase = _message_phase(item) if item_type == "message" else None
        if self.active_message_phase == "commentary":
            self.commentary_text_deltas = []
        # Record first-observed ordering for EVERY announced item; .done must reuse it or a mixed
        # announced/pending stream without output_index values reorders the calls.
        item_id = str(_event_field(item, "id", ""))
        if item_id and item_id not in self.announced_output_order:
            self.announced_output_order[item_id] = (self.next_output_sequence, _event_field(event, "output_index"))
            self.next_output_sequence += 1
        if "function_call" in str(item_type):
            self.has_tool_calls = True
            if item_id:
                announced_sequence, announced_index = self.announced_output_order[item_id]
                self.pending_function_calls[item_id] = {
                    "item": item, "arguments": str(_event_field(item, "arguments", "") or ""),
                    "output_index": announced_index, "sequence": announced_sequence,
                }

    def _on_text_delta(self, event: Any, event_type: str) -> None:
        delta_text = _event_field(event, "delta", "")
        if not delta_text:
            return
        # Harmony commentary/analysis text is mid-turn narration, never the final answer: route to the
        # reasoning callback, keep only the item for replay.
        if self.active_message_phase == "commentary":
            self.commentary_text_deltas.append(delta_text)
            # Legacy fallback when no first-class commentary consumer is installed.
            if self.on_commentary_message is None:
                self._safe(self.on_reasoning_delta, "on_reasoning_delta", delta_text)
        elif self.active_message_phase == "analysis":
            self._safe(self.on_reasoning_delta, "on_reasoning_delta", delta_text)
        else:
            self.text_deltas.append(delta_text)
            if self.has_tool_calls:
                return
            if not self.first_delta_fired:
                self.first_delta_fired = True
                self._safe(self.on_first_delta, "on_first_delta")
            self._safe(self.on_text_delta, "on_text_delta", delta_text)

    def _on_refusal_delta(self, event: Any, event_type: str) -> None:
        # ``response.refusal.delta``: the model declined and streams its explanation on the refusal
        # channel instead of output_text. It is answer text — a refusal-only stream must not end
        # with zero content and "did not emit a terminal response". The done item's ``refusal``
        # part is read by the normalizer; the deltas cover backends that omit the done item.
        refusal_text = _event_field(event, "delta", "")
        if isinstance(refusal_text, str) and refusal_text:
            self.text_deltas.append(refusal_text)

    def _on_function_call(self, event: Any, event_type: str) -> None:
        self.has_tool_calls = True
        pending = self.pending_function_calls.get(str(_event_field(event, "item_id", "")))
        if pending is None:
            return  # the item itself lands on output_item.done
        if "delta" in event_type:
            pending["arguments"] += _event_field(event, "delta", "") or ""
        elif event_type.endswith("function_call_arguments.done"):
            # Authoritative for the accumulated string; an explicit "" (zero-arg call) counts, only a
            # missing field keeps the streamed deltas.
            if (done_args := _event_field(event, "arguments", None)) is not None:
                pending["arguments"] = str(done_args)

    def _on_reasoning_delta(self, event: Any, event_type: str) -> None:
        reasoning_text = _event_field(event, "delta", "")
        if not reasoning_text or self.on_reasoning_delta is None:
            return
        summary_index = _event_field(event, "summary_index")
        if summary_index is not None:
            if self.active_summary_index is not None and summary_index != self.active_summary_index:
                reasoning_text = f"\n\n{reasoning_text}"
            self.active_summary_index = summary_index
        self._safe(self.on_reasoning_delta, "on_reasoning_delta", reasoning_text)

    def _on_item_done(self, event: Any, event_type: str) -> None:
        done_item = _event_field(event, "item")
        if done_item is None:
            return
        self.output_items.append(done_item)
        # Reuse the announced position when known (fresh tail sequence for unannounced items); the .done
        # event's own output_index wins over the announced one.
        done_id = str(_event_field(done_item, "id", ""))
        announced_sequence, announced_index = self.announced_output_order.get(done_id, (None, None))
        if announced_sequence is None:
            announced_sequence, self.next_output_sequence = self.next_output_sequence, self.next_output_sequence + 1
        self.output_indexes.append(_event_field(event, "output_index", announced_index))
        self.output_sequences.append(announced_sequence)
        # Confirmed by the authoritative done event; never settle it twice.
        self.pending_function_calls.pop(done_id, None)
        if _message_phase(done_item) == "commentary" and self.on_commentary_message is not None:
            commentary_text = "".join(self.commentary_text_deltas).strip() or _output_text_of(done_item)
            if commentary_text:
                self._safe(self.on_commentary_message, "on_commentary_message", commentary_text)
            self.commentary_text_deltas = []

    def _on_terminal(self, event: Any, event_type: str) -> bool:
        self.saw_terminal = True
        resp_obj = _event_field(event, "response")
        if resp_obj is not None:
            self.terminal_usage, self.terminal_response_id = _event_field(resp_obj, "usage"), _event_field(resp_obj, "id")
            rstatus = _event_field(resp_obj, "status")
            if isinstance(rstatus, str):
                self.terminal_status = rstatus
            if event_type == "response.incomplete":
                self.terminal_incomplete_details = _event_field(resp_obj, "incomplete_details")
            elif event_type == "response.failed":
                self.terminal_error = _event_field(resp_obj, "error")
        self.saw_response_completed = self.saw_response_completed or event_type == "response.completed"
        self.terminal_status = self.terminal_status or event_type.removeprefix("response.")
        return True

    # Exact-type handlers first, then substring-matched ones in priority order. ``error`` frames
    # carry the provider's real failure reason; raise so the credential pool + classifier see the body.
    _EXACT_HANDLERS = {
        "error": lambda self, event, event_type: _raise_stream_error(event),
        "response.output_item.added": _on_item_added, "response.output_item.done": _on_item_done,
        "response.completed": _on_terminal, "response.incomplete": _on_terminal, "response.failed": _on_terminal,
        "response.refusal.delta": _on_refusal_delta,
    }
    _FUZZY_HANDLERS = (
        (lambda t: "output_text.delta" in t, _on_text_delta), (lambda t: "function_call" in t, _on_function_call),
        (lambda t: "reasoning" in t and "delta" in t, _on_reasoning_delta),
    )

    def feed(self, event: Any) -> bool:
        """Process one event; True when the stream hit a terminal frame."""
        event_type = _event_field(event, "type", "")
        event_type = event_type if isinstance(event_type, str) else ""
        handler = self._EXACT_HANDLERS.get(event_type) or next((h for m, h in self._FUZZY_HANDLERS if m(event_type)), None)
        return bool(handler(self, event, event_type)) if handler is not None else False

    def _settled_output(self) -> List[Any]:
        """Merge .done items with settled pending calls, keeping stream order."""
        indexed = list(zip(self.output_indexes, self.output_sequences, self.output_items))
        for pending in self.pending_function_calls.values():
            item = pending["item"]
            indexed.append((pending.get("output_index"), pending["sequence"], SimpleNamespace(
                type="function_call", id=_event_field(item, "id", None), call_id=_event_field(item, "call_id", None),
                name=_event_field(item, "name", None), status="completed",
                # Empty/whitespace arguments become "{}" so zero-delta calls stay executable; malformed
                # non-empty JSON passes through untouched.
                arguments=(pending["arguments"] or "").strip() or "{}",
            )))
        # output_index is optional: protocol order only when every entry has one, else wire order.
        if all(entry[0] is not None for entry in indexed):
            with suppress(TypeError):  # non-comparable index values: keep wire order
                indexed.sort(key=lambda entry: entry[0])
        else:
            indexed.sort(key=lambda entry: entry[1])
        return [entry[2] for entry in indexed]

    def result(self) -> SimpleNamespace:
        # With only plain text deltas (no tool calls), synthesize one message item.
        output: List[Any] = list(self.output_items)
        if not output and self.text_deltas and not self.has_tool_calls:
            content = [SimpleNamespace(type="output_text", text="".join(self.text_deltas))]
            output = [SimpleNamespace(type="message", role="assistant", status="completed", content=content)]
        # Done items stay authoritative; settlement only fills the gap left by backends that omit
        # per-item done events on a successful completion.
        if self.pending_function_calls and self.saw_response_completed:
            output = self._settled_output()
        # No terminal frame AND no usable content = truncated / rejected stream.
        if not self.saw_terminal and not output:
            raise RuntimeError("Codex Responses stream did not emit a terminal response")
        return SimpleNamespace(
            output=output, output_text="".join(self.text_deltas), usage=self.terminal_usage, status=self.terminal_status,
            id=self.terminal_response_id, model=self.model, incomplete_details=self.terminal_incomplete_details,
            error=self.terminal_error)


def _consume_codex_event_stream(
    event_iter: Any, *, model: str, on_text_delta=None, on_reasoning_delta=None, on_commentary_message=None,
    on_first_delta=None, on_event=None, interrupt_check=None,
) -> SimpleNamespace:
    """Consume a Codex Responses SSE stream into a Response-shaped ``SimpleNamespace`` (see
    :class:`_CodexResponseAssembler`; ``status`` is ``completed`` when the stream ended with content but no
    terminal frame; ``model`` comes from kwargs).

    Callbacks: ``on_text_delta`` per output_text delta, suppressed once a function_call is seen;
    ``on_reasoning_delta`` for reasoning and ``phase=analysis`` deltas (also commentary without a commentary
    callback); ``on_commentary_message`` once per completed ``phase=commentary`` message, before any following
    tool item; ``on_first_delta`` one-shot; ``on_event`` every event before any processing; ``interrupt_check()``
    True breaks the loop and may raise ``TimeoutError`` / ``InterruptedError`` for request retirement that
    must not become a partial final response."""
    assembler = _CodexResponseAssembler(model=model, on_text_delta=on_text_delta, on_reasoning_delta=on_reasoning_delta,
                                        on_commentary_message=on_commentary_message, on_first_delta=on_first_delta)
    for event in event_iter:
        if on_event is not None:
            try:
                on_event(event)
            except (TimeoutError, InterruptedError):
                raise  # watchdog / cancellation control flow must propagate
            except Exception:
                logger.debug("Codex stream on_event hook raised", exc_info=True)
        if (interrupt_check is not None and interrupt_check()) or assembler.feed(event):
            break
    return assembler.result()


def _sanitize_consumer_codex_request(agent: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Drop fields the ChatGPT OAuth Codex endpoint rejects, at the final wire boundary (after Relay /
    middleware / ``request_overrides``): a late ``prompt_cache_retention``, top-level or nested in
    ``extra_body``, would otherwise HTTP 400 a valid follow-up."""
    sanitized = dict(request)
    # getattr: run_codex_stream is also driven with stand-in agents carrying only the attrs a path needs.
    backend_predicate = getattr(agent, "_is_codex_backend", None)
    if not (callable(backend_predicate) and bool(backend_predicate())):
        return sanitized
    dropped_from = ["top-level"] if "prompt_cache_retention" in sanitized else []
    sanitized.pop("prompt_cache_retention", None)
    # Copy before editing (caller's mapping must not mutate); drop when emptied.
    extra_body = sanitized.get("extra_body")
    if isinstance(extra_body, dict) and "prompt_cache_retention" in extra_body:
        sanitized["extra_body"] = {k: v for k, v in extra_body.items() if k != "prompt_cache_retention"}
        if not sanitized["extra_body"]:
            sanitized.pop("extra_body")
        dropped_from.append("extra_body")
    if dropped_from:
        logger.warning("Dropped unsupported prompt_cache_retention at consumer Codex wire boundary (model=%s, via %s).",
                       sanitized.get("model", getattr(agent, "model", "unknown")), ", ".join(dropped_from))
    return sanitized


def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta=None):
    """One streaming Responses API request over raw ``responses.create(stream=True)`` events."""
    import httpx as _httpx
    from openai import APIConnectionError as _APIConnectionError
    from agent import relay_llm
    transport_errors = (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ReadError, _httpx.ConnectError, ConnectionError)
    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")
    max_stream_retries, model = 1, api_kwargs.get("model")
    # Accumulate streamed text so callers / compat shims can read it.
    agent._codex_streamed_text_parts: list = []
    # Retirement token for THIS request (installed by ``interruptible_api_call``). A watchdog that kills the
    # connection clears the agent-level token, so a worker still draining frames can tell it was retired.
    # ``None`` = no watchdog; every check passes.
    watchdog_state = _codex_watchdog_state_var.get()
    request_token = (
        watchdog_state.token
        if watchdog_state is not None
        else getattr(agent, "_active_codex_stream_request_token", None)
    )
    # Delta-sink claim for the CURRENT physical attempt (None until the stream opens).
    writer_token = {"value": None}

    def _request_is_current() -> bool:
        return request_token is None or getattr(agent, "_active_codex_stream_request_token", None) is request_token

    def _fenced(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        """Wrap a callback so a retired request's late frames never reach the agent."""
        return lambda value: fn(value) if _request_is_current() else None

    def _on_text_delta(text: str) -> None:
        agent._codex_streamed_text_parts.append(text)
        agent._fire_stream_delta(text)

    def _on_event(event: Any) -> None:  # TTFB/activity touch — once per SSE event.
        now = time.time()
        has_progress = _codex_event_has_content(event)
        if watchdog_state is not None:
            with watchdog_state.lock:
                if watchdog_state.retry_started_ts is not None:
                    watchdog_state.retry_started_ts = None
                    watchdog_state.last_progress_ts = None
                watchdog_state.last_event_ts = now
                if has_progress:
                    watchdog_state.last_progress_ts = now
        agent._touch_activity("receiving stream response")

    def _interrupt_or_superseded() -> bool:
        # A retired request must NOT break out of the consume loop (that returns a partial ``final`` with
        # status "completed"); raise so the watchdog's TimeoutError is seen.
        if not _request_is_current():
            raise TimeoutError("Codex Responses stream request retired before terminal response")
        return bool(agent._interrupt_requested)

    def _open_codex_stream(next_api_kwargs: dict[str, Any]):
        from hermes_cli.providers import is_actual_route

        if is_actual_route(
            getattr(agent, "provider", ""),
            str(getattr(active_client, "base_url", "") or ""),
        ):
            raise ValueError(
                "Actual requests require Chat Completions; refusing to call /responses."
            )
        stream_kwargs = _sanitize_consumer_codex_request(agent, next_api_kwargs)
        stream_kwargs["stream"] = True
        return active_client.responses.create(**bypass_sdk_request_transform(stream_kwargs))

    def _log_failure(exc: BaseException) -> None:
        request_body_bytes, exception_chain = _codex_request_failure_details(exc)
        logger.warning("Codex Responses request failed: serialized_request_body_bytes=%s stream_opened=%s "
                       "exception_chain=%s model=%s", "unknown" if request_body_bytes is None else request_body_bytes,
                       str(writer_token["value"] is not None).lower(), exception_chain, getattr(agent, "model", "unknown"))

    def _codex_stream_created(_raw_stream: Any) -> None:
        # Claim the delta sink for THIS attempt; a newer attempt supersedes this token.
        writer_token["value"] = claim_stream_writer(agent)

    def _accept_codex_chunk(_chunk: Any) -> bool:
        token = writer_token["value"]
        if token is None or stream_writer_is_current(agent, token):
            return True
        logger.warning("Codex streaming attempt superseded by a newer stream; stopping consumption to preserve "
                       "the single-writer invariant (model=%s).", api_kwargs.get("model", "unknown"))
        return False

    def _drain_for_finalizer(event_stream: Any) -> None:
        # ``final`` is already assembled; draining only lets Relay run its finalizer. A transport error
        # here must NOT discard the completed, already-billed response.
        try:
            for _ignored in event_stream:
                pass
        except (*transport_errors, _APIConnectionError) as exc:
            if not isinstance(exc, transport_errors):
                _log_failure(exc)
            logger.warning("Codex Responses stream transport finalization failed after a terminal response was already "
                           "received; returning the completed response instead of retrying. %s error=%s",
                           agent._client_log_context(), exc)

    def _close_event_stream(event_stream: Any) -> None:
        close_fn = getattr(event_stream, "close", None)  # None while connect never succeeded
        try:
            if callable(close_fn):
                close_fn()
        except Exception:
            # A failed close can leave this connection checked out of the httpx pool while the caller
            # reuse-caches the client; poison the slot so close really closes the pool. ``client is None``
            # is the shared primary client — never force-shut.
            if client is not None:
                agent._abort_request_openai_client(active_client, reason="codex_stream_close_failed")
    show_commentary = getattr(agent, "show_commentary", True)
    wants_commentary = getattr(agent, "interim_assistant_callback", None) is not None and show_commentary
    on_commentary_message = _fenced(lambda text: agent._fire_streamed_codex_commentary(text)) if wants_commentary else None
    call_role = ("delegated" if getattr(agent, "is_subagent", False)
                 else "fallback" if int(getattr(agent, "_fallback_index", 0) or 0) > 0 else "primary")
    for attempt in range(max_stream_retries + 1):
        if not _request_is_current():
            raise TimeoutError("Codex Responses stream request retired before retry")
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before Codex stream retry")
        if attempt > 0 and watchdog_state is not None and watchdog_state.phase_aware:
            # A physical reconnect has its own no-event TTFB phase. Its first parsed
            # event clears this marker and starts a fresh model-progress phase.
            with watchdog_state.lock:
                watchdog_state.retry_started_ts = time.time()
        intercepted_events: list = []
        writer_token["value"] = event_stream = None
        try:
            try:
                event_stream = relay_llm.stream(
                    dict(api_kwargs), _open_codex_stream,
                    session_id=str(getattr(agent, "session_id", "") or ""),
                    name=str(getattr(agent, "provider", "") or "codex"), model_name=str(model or ""),
                    finalizer=lambda: _consume_codex_event_stream(list(intercepted_events), model=model),
                    on_stream_created=_codex_stream_created, on_chunk=intercepted_events.append,
                    chunk_adapter=lambda chunk: chunk, accept_chunk=_accept_codex_chunk,
                    completed_response_predicate=lambda r: bool(hasattr(r, "output") and not hasattr(r, "__iter__")),
                    metadata={"api_mode": "codex_responses", "call_role": call_role, "retry_count": attempt,
                              "api_request_id": getattr(agent, "_current_api_request_id", None)},
                    defer_logical_completion=True,
                )
                final = _consume_codex_event_stream(
                    event_stream, model=model, on_text_delta=_fenced(_on_text_delta),
                    on_reasoning_delta=_fenced(lambda text: agent._fire_reasoning_delta(text)),
                    on_commentary_message=on_commentary_message, on_first_delta=on_first_delta,
                    on_event=_fenced(_on_event), interrupt_check=_interrupt_or_superseded,
                )
            except transport_errors as exc:
                if attempt >= max_stream_retries:
                    _log_failure(exc)
                    raise
                logger.debug(
                    "Codex Responses stream connect failed (attempt %s/%s); retrying. %s error=%s" if event_stream is None
                    else "Codex Responses stream transport failed mid-iteration (attempt %s/%s); retrying. %s error=%s",
                    attempt + 1, max_stream_retries + 1, agent._client_log_context(), exc,
                )
                continue
            except RuntimeError:
                # "No terminal response"; Relay may still hold a finalizer-assembled response.
                if event_stream is not None and event_stream.final_response is not None:
                    return event_stream.final_response
                raise
            except _APIConnectionError as exc:
                _log_failure(exc)
                raise
            if not agent._interrupt_requested:
                _drain_for_finalizer(event_stream)
            if final.status in {"incomplete", "failed"}:
                logger.warning("Codex Responses stream terminal status=%s "
                               "(incomplete_details=%s, error=%s, streamed_chars=%d). %s",
                               final.status, final.incomplete_details, final.error,
                               sum(len(p) for p in agent._codex_streamed_text_parts), agent._client_log_context())
            return final
        finally:
            _close_event_stream(event_stream)


__all__ = [
    "run_codex_stream", "_consume_codex_event_stream",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Backward-compatible alias for the unified event-driven path.

    Historically this was the fallback when the SDK's high-level
    ``responses.stream(...)`` helper raised on shape drift.  The primary
    path now does exactly what the fallback did, so this just forwards.
    Kept as a public symbol because tests and a small number of call sites
    still reference it by name.
    """
    return run_codex_stream(agent, api_kwargs, client=client)
# ---- END PLUGIN-COMPAT ----
