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
	@echo "  make cli                 Run the text-only CLI playthrough (uses inferno)"
	@echo "  make download-model      Fetch the Gemma 4 E4B Q8_0 GGUF (~8 GB)"
	@echo "  make test                Run pytest (unit suite, skips live-llama integration)"
	@echo "  make test-integration    Run pytest including the live-llama integration test"
	@echo "  make typecheck           Run TS typecheck on handler/web/"
	@echo "  make build               Production build of handler/web/"
	@echo "  make kill                Force-kill stragglers on :8000/:8080/:5173-5176"
	@echo "  make clean               Remove __pycache__"
	@echo ""
	@echo "Browser dev flow: 'make sidecar' (terminal 1) + 'make web' (terminal 2), open http://localhost:5173"

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

# Two-pass kill — single-pass misses multiprocessing-fork orphans that inherit
# listening sockets. Exits non-zero if any port is still bound after the kill.
.PHONY: kill
kill:
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "& { \
	  $$ports = 8000,8080,5173,5174,5175,5176; \
	  foreach ($$pass in 1..2) { \
	    foreach ($$p in $$ports) { \
	      $$pids = (Get-NetTCPConnection -LocalPort $$p -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique; \
	      foreach ($$pid in $$pids) { try { Stop-Process -Id $$pid -Force -ErrorAction Stop; Write-Host \"killed pid $$pid on :$$p\" } catch {} } \
	    } \
	    Start-Sleep -Milliseconds 300 \
	  } \
	  $$still = @(); foreach ($$p in $$ports) { if (Get-NetTCPConnection -LocalPort $$p -ErrorAction SilentlyContinue) { $$still += $$p } } \
	  if ($$still.Count) { Write-Host \"still bound: $$($$still -join ',')\"; exit 1 } else { Write-Host 'all clear' } \
	}"
