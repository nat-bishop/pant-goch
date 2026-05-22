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
	@echo "  make playtest            Run simulated Gemini-player playthroughs + invariant report"
	@echo "  make typecheck           Run TS typecheck on web/"
	@echo "  make build               Production build of web/"
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
	uv run uvicorn handler.main:app --reload --reload-dir src --port 8000

# Inferno mode: skip bundled llama, route pydantic-ai at inferno's /v1.
# Requires inferno running on :7770. See CLAUDE.md "Inferno gateway"
# section for env var details.
.PHONY: sidecar-inferno
sidecar-inferno:
	SMALLVILLE_USE_INFERNO=1 uv run uvicorn handler.main:app --reload --reload-dir src --port 8000

.PHONY: web
web:
	cd web && npm run dev

# Spawn sidecar (inferno-routed) + Vite dev server together via concurrently.
# Ctrl-C kills both (--kill-others). Backend env var is set here, not in the
# npm script, so the package.json stays portable (no cross-env dep needed).
.PHONY: dev
dev:
	cd web && SMALLVILLE_USE_INFERNO=1 npm run dev:all

# Text-only CLI playthrough — stdin pick/walk, paced prose to stdout, log
# entries indented after each scene. Routes at inferno by default.
.PHONY: cli
cli:
	SMALLVILLE_USE_INFERNO=1 uv run python -m handler

.PHONY: test
test:
	uv run pytest -q --ignore=tests/test_live_integration.py

.PHONY: test-integration
test-integration:
	uv run pytest -q

# Simulated playtest harness — a Gemini player drives many sessions in parallel
# against the configured engine (inferno by default), each scored against the
# invariant suite. Requires GOOGLE_API_KEY in env. See EVALS.md.
.PHONY: playtest
playtest:
	SMALLVILLE_USE_INFERNO=1 uv run python -m evals.run

# LLM-as-judge pass over a recorded run. Reads dumped artifacts only (no llama
# session); defaults to local Qwen via inferno. RUN=<artifact dir> required.
# e.g. make judge RUN=evals/artifacts/20260520-161349
.PHONY: judge
judge:
	SMALLVILLE_USE_INFERNO=1 uv run python -m evals.judge_run --run $(RUN)

# Phase-1 delta regression eval — frozen turn contexts run against the live
# phase1 agent K times each, scored by deterministic assertions on the emitted
# delta (trust/guard/attention). Fast, focused, no full playthrough. Seeded
# with the attention-spam regression. Override samples with K=<n>.
.PHONY: eval-deltas
eval-deltas:
	SMALLVILLE_USE_INFERNO=1 uv run python -m evals.phase1_deltas --k $(or $(K),12)

.PHONY: typecheck
typecheck:
	cd web && npm run typecheck

.PHONY: build
build:
	cd web && npm run build

.PHONY: clean
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Image proofing station — parses world/backdrops/IMAGES.md, shows every
# plate with its prompt + canonical preview, lets the author re-roll a new
# variant per entry (Gemini flash or pro). Variants land in
# world/backdrops/_variants/<id>/ so the existing canonical is never
# clobbered. Requires GEMINI_API_KEY in env.
.PHONY: proof
proof:
	uv run python scripts/proof_server.py

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
	uv run python scripts/kill_dev_ports.py
