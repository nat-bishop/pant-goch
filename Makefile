MODEL       := models/ggml-org-gemma-4-E4B-it-Q8_0.gguf
MODEL_URL   := https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
LLAMA_BIN   := vendor/llama.cpp/llama-server.exe
LLAMA_HOST  := 127.0.0.1
LLAMA_PORT  := 8080

# Eval llama (Qwen 3.6 35B-A3B MoE) — only needed by future LLM-judge phases
# of the playtest. The MVP playtest is programmatic-only and doesn't require
# the eval llama. Kept here for when judges return.
EVAL_MODEL      := models/Qwen3.6-35B-A3B-Q8_0.gguf
EVAL_MODEL_URL  := https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q8_0.gguf

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "HANDLER targets:"
	@echo "  make run            Boot sidecar (spawns llama) + web dev server in parallel"
	@echo "  make download-model      Fetch the Gemma 4 E4B Q8_0 GGUF (~8 GB)"
	@echo "  make download-eval-model Fetch the Qwen 3.6 35B-A3B Q8_0 GGUF (~37 GB)"
	@echo "  make llama               Run engine llama-server (Gemma on :8080)"
	@echo "  make llama-eval          Run eval llama-server (Qwen 35B-A3B on :8081)"
	@echo "  make sidecar             Run Python sidecar, externally-managed llama (terminal 2)"
	@echo "  make web                 Run Vite dev server (terminal 3)"
	@echo "  make test                Run pytest"
	@echo "  make typecheck           Run TS typecheck on web/"
	@echo "  make playtest            Run the option-picker playtest (requires sidecar running)"
	@echo "  make clean               Remove __pycache__ and logs/*.log"
	@echo "  make kill-zombies        Force-kill stragglers on :8000/:8080/:5173-5176"
	@echo "  make mission-check       Validate handler/missions/*.yaml against the schema"

# `run` boots sidecar and web in parallel via `concurrently` (npm). Ctrl-C
# propagates through concurrently → tree-kill → both children die. The
# sidecar spawns llama-server itself (LAUNCH_LLAMA=true default) and its
# Job Object (launcher.py) ensures llama dies when the sidecar dies.
.PHONY: run
run: $(MODEL)
	@echo "Starting sidecar (spawns llama) and web in parallel. Browser: http://localhost:5173"
	cd web && npm run dev:all

$(MODEL):
	mkdir -p models
	curl -L -o $(MODEL) $(MODEL_URL)

.PHONY: download-model
download-model: $(MODEL)

$(EVAL_MODEL):
	mkdir -p models
	curl -L -o $(EVAL_MODEL) $(EVAL_MODEL_URL)

.PHONY: download-eval-model
download-eval-model: $(EVAL_MODEL)

# Boots llama-server via handler.run_llama so the argv comes from the same
# `launcher.build_command` the sidecar uses. Don't reintroduce a hardcoded
# llama-server invocation here — it will drift from Settings.
.PHONY: llama
llama: $(MODEL)
	cd handler && uv run python -m handler.run_llama

# Eval llama, routed through the same launcher.py with `--role eval`. Used
# by future LLM-judge phases of the playtest.
.PHONY: llama-eval
llama-eval: $(EVAL_MODEL)
	cd handler && uv run python -m handler.run_llama --role eval

.PHONY: sidecar
sidecar:
	cd handler && LAUNCH_LLAMA=false uv run python -m handler

.PHONY: web
web:
	cd web && npm run dev

.PHONY: test
test:
	cd handler && uv run pytest

# Mission YAML schema check — loads every handler/missions/*.yaml and
# surfaces validator errors with the file path. Cheap (no llama, no SSE).
.PHONY: mission-check
mission-check:
	cd handler && uv run python -c "from pathlib import Path; from handler.mission_loader import load_mission; [print(f'OK {p.name}: {load_mission(p).codename}') for p in sorted(Path('missions').glob('*.yaml'))]"

# Playtest harness. Requires the sidecar to be running (boots its own llama
# by default). Probes /health before starting. Programmatic evaluators only
# for MVP — LLM judges come back when prose-quality scoring is worth it.
.PHONY: playtest
playtest:
	cd handler && PYTHONIOENCODING=utf-8 uv run python -u -m scripts.playtest

.PHONY: typecheck
typecheck:
	cd web && npm run typecheck

.PHONY: clean
clean:
	rm -f logs/*.log
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: kill-zombies
kill-zombies:
	@powershell -Command 'Get-NetTCPConnection -LocalPort 8000,8080,8081,5173,5174,5175,5176 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $$_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Host "killed any processes holding :8000/:8080/:8081/:5173-5176"'
