# handler

FastAPI sidecar for HANDLER. Owns the deterministic engine (`spine.py`, `cycle.py`) and three stateless Pydantic AI agents (`agents.py`) on a single managed `llama-server` (Gemma 4 E4B Q8_0), and streams the cycle to the React frontend over SSE.

See the top-level `README.md` for the game design, project layout, run/test commands, and troubleshooting. See `../CLAUDE.md` for engine invariants and the file-by-file map.

## Quick start

```bash
cd handler
uv sync
uv run pytest               # 49 unit tests, no llama needed
uv run python -m handler    # boot sidecar (auto-spawns llama-server)
```

The sidecar listens on `:8000`. Open `<http://localhost:5173>` after also booting the Vite dev server (`make web` from the repo root).
