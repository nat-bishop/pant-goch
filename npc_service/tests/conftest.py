import json
import os
from typing import Iterable

# Keep Logfire console output off during tests so pytest captures stay clean.
# Override with `NPC_LOGFIRE_CONSOLE=true uv run pytest -s ...` when debugging.
# Must run before `npc_service.main` is imported, since `_BOOT_SETTINGS` reads
# env vars eagerly at module import — pytest auto-loads conftest.py first.
os.environ.setdefault("NPC_LOGFIRE_CONSOLE", "false")

import httpx
import pytest


# ───────────────────────── live-llama integration fixtures ─────────────────────────

LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"
LLAMA_TOKENIZE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/tokenize"


@pytest.fixture(scope="module")
def live_llama() -> None:
    """Skip the module if llama-server isn't reachable. /health 200 means
    the model slot is loaded — /v1/models would 200 too early on cold boot."""
    try:
        r = httpx.get(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health", timeout=2.0)
        if r.status_code != 200:
            pytest.skip(f"llama-server health returned {r.status_code}")
    except httpx.HTTPError as e:
        pytest.skip(f"llama-server not reachable at {LLAMA_HOST}:{LLAMA_PORT}: {e}")


# ───────────────────────── SSE helper ─────────────────────────


def consume_sse(response_cm) -> Iterable[dict]:
    """Iterate `data: {...}` events from a streaming TestClient response."""
    with response_cm as r:
        assert r.status_code == 200, r.text
        buffer = ""
        for chunk in r.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, _, buffer = buffer.partition("\n\n")
                for line in block.split("\n"):
                    if line.startswith("data: "):
                        try:
                            yield json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue


# ───────────────────────── token-counter stub ─────────────────────────


@pytest.fixture(autouse=True)
def _stub_llama_token_counter(request, monkeypatch):
    """Replace `make_llama_token_counter` with a stub that always reports 0
    tokens, so the chat agent's history-compaction processor never makes the
    /tokenize HTTP call (llama-server isn't running in unit tests).

    Tests marked `@pytest.mark.integration` bypass the stub — they run
    against a real llama-server and need the real /tokenize endpoint.
    Tests that exercise the compactor directly (e.g. test_history_compaction.py)
    inject their own counter and never import `main`, so the patch is a no-op
    for them — running the autouse hook anyway keeps the rules simple.
    """
    if request.node.get_closest_marker("integration"):
        return  # integration tests use the real counter
    from npc_service import main as _main

    async def _zero(_messages):
        return 0

    def _factory(_tokenize_url):
        return _zero

    monkeypatch.setattr(_main, "make_llama_token_counter", _factory)
