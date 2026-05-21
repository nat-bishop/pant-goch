"""Retry wrapper for the engine model — eval-only resilience.

Under concurrent playtest load the local gemma server (via inferno) intermittently
returns HTTP 500 ("Internal Server Error") — KV/VRAM pressure when many long
sessions run at once, made worse when the 35B judge model is co-resident. Without
a retry that single blip raises `ModelHTTPError` out of `play_session`, and the
WHOLE session — often 10+ scenes deep — is dropped from the corpus. Three separate
runs lost 9–14 of 16 sessions this way.

Per-call retry (here) beats per-session retry: a deep session makes ~100 engine
calls, so absorbing a transient 500 at the call level lets the session survive,
rather than re-running the entire playthrough. Production (the SSE backend) keeps
the bare model — this wrapper is wired in only by `evals.run`.
"""
from __future__ import annotations

import asyncio
import random
import sys
from contextlib import asynccontextmanager

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.wrapper import WrapperModel

# Transient HTTP statuses worth retrying. 500 is the one the llama/inferno endpoint
# emits under load; the rest cover gateway/rate-limit blips for completeness.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RetryModel(WrapperModel):
    """Wrap an engine model, retrying transient 5xx/429 with exponential backoff."""

    def __init__(self, wrapped, *, max_attempts: int = 5, base_delay: float = 0.5):
        super().__init__(wrapped)
        self._max_attempts = max_attempts
        self._base_delay = base_delay

    @staticmethod
    def _transient(e: Exception) -> bool:
        return isinstance(e, ModelHTTPError) and e.status_code in _RETRYABLE_STATUS

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(self._base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3))

    async def request(self, messages, model_settings, model_request_parameters):
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self.wrapped.request(
                    messages, model_settings, model_request_parameters
                )
            except Exception as e:  # noqa: BLE001 — re-raised unless transient
                if not self._transient(e) or attempt == self._max_attempts:
                    raise
                await self._backoff(attempt)

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):
        # Retry only the stream ENTRY (the POST that may 500 before any tokens
        # arrive). Once tokens are flowing, a mid-stream failure propagates — the
        # consumer has already seen partial output, so re-yielding a fresh stream
        # would corrupt it.
        for attempt in range(1, self._max_attempts + 1):
            cm = self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            )
            try:
                response_stream = await cm.__aenter__()
            except Exception as e:  # noqa: BLE001
                if not self._transient(e) or attempt == self._max_attempts:
                    raise
                await self._backoff(attempt)
                continue
            try:
                yield response_stream
            finally:
                await cm.__aexit__(*sys.exc_info())
            return
