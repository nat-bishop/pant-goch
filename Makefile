MODEL       := models/ggml-org-gemma-4-E4B-it-Q8_0.gguf
MODEL_URL   := https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
LLAMA_BIN   := vendor/llama.cpp/llama-server.exe
LLAMA_HOST  := 127.0.0.1
LLAMA_PORT  := 8080

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "HANDLER v2 targets:"
	@echo "  make sidecar             Run the FastAPI sidecar (spawns its own llama-server)"
	@echo "  make sidecar-inferno     Run the sidecar against the inferno gateway (skips bundled llama)"
	@echo "  make web                 Run the Vite dev server"
	@echo "  make dev                 Run sidecar (inferno) + web together via concurrently"
	@echo "  make cli                 Run the text-only CLI playthrough (uses inferno)"
	@echo "  make download-model      Fetch the Gemma 4 E4B Q8_0 GGUF (~8 GB)"
	@echo "  make test                Run pytest (unit suite, skips live-llama integration)"
	@echo "  make test-integration    Run pytest including the live-llama integration test"
	@echo "  make typecheck           Run TS typecheck on handler/web/"
	@echo "  make build               Production build of handler/web/"
	@echo "  make kill                Force-kill stragglers on :8000/:8080/:5173-5176"
	@echo "  make proof               Image proofing station (Gemini backdrop generation UI) on :8765"
	@echo "  make clean               Remove __pycache__"
	@echo ""
	@echo "Browser dev flow: 'make dev' (one terminal, inferno-routed) — open http://localhost:5173"
	@echo "Or: 'make sidecar' (terminal 1) + 'make web' (terminal 2) for the bundled-llama path"

$(MODEL):
	mkdir -p models
	curl -L -o $(MODEL) $(MODEL_URL)

.PHONY: download-model
download-model: $(MODEL)

.PHONY: sidecar
sidecar:
	cd handler && uv run uvicorn handler.main:app --reload --port 8000

# Inferno mode: skip bundled llama, route pydantic-ai at inferno's /v1.
# Requires inferno running on :7770. See handler/CLAUDE.md (when written) or
# the root CLAUDE.md "Inferno gateway" section for env var details.
.PHONY: sidecar-inferno
sidecar-inferno:
	cd handler && SMALLVILLE_USE_INFERNO=1 uv run uvicorn handler.main:app --reload --port 8000

.PHONY: web
web:
	cd handler/web && npm run dev

# Spawn sidecar (inferno-routed) + Vite dev server together via concurrently.
# Ctrl-C kills both (--kill-others). Backend env var is set here, not in the
# npm script, so the package.json stays portable (no cross-env dep needed).
.PHONY: dev
dev:
	cd handler/web && SMALLVILLE_USE_INFERNO=1 npm run dev:all

# Text-only CLI playthrough — stdin pick/walk, paced prose to stdout, log
# entries indented after each scene. Routes at inferno by default.
.PHONY: cli
cli:
	cd handler && SMALLVILLE_USE_INFERNO=1 uv run python -m handler

.PHONY: test
test:
	cd handler && uv run pytest -q --ignore=tests/test_live_integration.py

.PHONY: test-integration
test-integration:
	cd handler && uv run pytest -q

.PHONY: typecheck
typecheck:
	cd handler/web && npm run typecheck

.PHONY: build
build:
	cd handler/web && npm run build

.PHONY: clean
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Image proofing station — parses handler/world/backdrops/IMAGES.md, shows
# every plate with its prompt + canonical preview, lets the author re-roll a
# new variant per entry (Gemini flash or pro). Variants land in
# handler/world/backdrops/_variants/<id>/ so the existing canonical is never
# clobbered. Requires GEMINI_API_KEY in env.
.PHONY: proof
proof:
	cd handler && uv run python scripts/proof_server.py

# Two-pass kill — single-pass misses multiprocessing-fork orphans that inherit
# listening sockets. Exits non-zero if any port is still bound after the kill.
#
# Why Python (not inline PowerShell): when make's SHELL is bash (git bash on
# Windows), bash eats `$var` references in the recipe before PowerShell ever
# sees them, so every variable in the previous inline PowerShell script
# collapsed to empty (`Write-Host "killed pid  on :"` etc.) and the brace
# parser then died with "Unexpected token ')'". Python via `uv run` dodges
# the whole shell-quoting + reserved-`$pid` mess — and `uv run` is already
# the project's standard invocation.
.PHONY: kill
kill:
	cd handler && uv run python scripts/kill_dev_ports.py
