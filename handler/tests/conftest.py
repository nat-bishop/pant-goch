import json
import os
from typing import Iterable

# Keep Logfire console output off during tests so pytest captures stay clean.
# Override with `HANDLER_LOGFIRE_CONSOLE=true uv run pytest -s ...` when debugging.
# Must run before `handler.main` is imported, since `_BOOT_SETTINGS` reads
# env vars eagerly at module import — pytest auto-loads conftest.py first.
os.environ.setdefault("HANDLER_LOGFIRE_CONSOLE", "false")

import httpx
import pytest

from handler.config import LLAMA_BASE_URL, LLAMA_HOST, LLAMA_PORT


# ───────────────────────── live-llama integration fixtures ─────────────────────────


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
