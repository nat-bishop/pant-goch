MODEL       := models/ggml-org-gemma-4-E4B-it-Q8_0.gguf
MODEL_URL   := https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
LLAMA_BIN   := vendor/llama.cpp/llama-server.exe
LLAMA_HOST  := 127.0.0.1
LLAMA_PORT  := 8080

# Eval llama: Qwen 3.6 35B-A3B (MoE, ~3B active) on mainline llama.cpp.
# Powers the synthetic-player agent and every LLM judge in the playtester.
# Different family from Gemma (the SUT) to mitigate same-family preference
# leakage in judging — see arxiv 2502.01534. Uses the same vendored
# llama-server binary as the engine; ik_llama.cpp would offer ~25% better
# MoE throughput but adds a from-source build step on Windows we don't
# need yet.
EVAL_MODEL      := models/Qwen3.6-35B-A3B-Q8_0.gguf
EVAL_MODEL_URL  := https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q8_0.gguf
EVAL_LLAMA_BIN  := vendor/llama.cpp/llama-server.exe
EVAL_LLAMA_PORT := 8081

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Inbox targets:"
	@echo "  make run             Boot sidecar (spawns llama) + web dev server in parallel"
	@echo "  make download-model       Fetch the Gemma 4 E4B Q8_0 GGUF (~8 GB)"
	@echo "  make download-eval-model  Fetch the Qwen 3.6 35B-A3B Q8_0 GGUF (~37 GB)"
	@echo "  make llama                Run engine llama-server (Gemma on :8080, single-user KV)"
	@echo "  make llama-playtest       Run engine llama with bumped KV pool for the playtester (~2M ctx, 64 concurrent)"
	@echo "  make llama-eval           Run eval llama-server (Qwen 35B-A3B on :8081, ik_llama.cpp, ~2M ctx)"
	@echo "  make sidecar              Run Python sidecar, no llama   (terminal 2)"
	@echo "  make web                  Run Vite dev server            (terminal 3)"
	@echo "  make test                 Run pytest"
	@echo "  make playtest             Run the LLM playtester (requires both llamas up)"
	@echo "  make lint-scenario   Lint npc_service/scenario.yaml (fast; spec rules only)"
	@echo "  make analyze-scenario  Run ASP-backed reachability analyzer (clingo; sub-second on 13b)"
	@echo "  make typecheck       Run TS typecheck on web/"
	@echo "  make clean           Remove __pycache__ and logs/*.log"
	@echo "  make kill-zombies    Force-kill stragglers on :8000/:8080/:5173-5176 (recovery)"

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

# Boots llama-server via npc_service.run_llama so the argv comes from the
# same `launcher.build_command` the sidecar uses. Don't reintroduce a
# hardcoded llama-server invocation here — it will drift from Settings
# (this is exactly how `-np` got stuck at 8 while config.py said 128).
.PHONY: llama
llama: $(MODEL)
	cd npc_service && uv run python -m npc_service.run_llama

# Engine llama, sized for the playtester at --parallel 64. Bumps `-np` to
# 64 (matches max concurrency) and lets launcher.py auto-size `-c` to
# `n_parallel × llama_ctx = 64 × 32k = 2M`. Single-session llama_ctx stays
# 32k, so the chat agent's history-compaction threshold is unaffected.
# KV cost ≈ 64 × 32k × ~4 KB/tok (Gemma MQA) ≈ ~8 GB VRAM. The default
# `--no-kv-unified` (see config.py:llama_kv_unified) means llama-server
# divides `-c` by `-np` per slot — each slot gets the full 32k budget.
.PHONY: llama-playtest
llama-playtest: $(MODEL)
	cd npc_service && LLAMA_N_PARALLEL=64 uv run python -m npc_service.run_llama

# Eval llama: Qwen 3.6 35B-A3B (MoE) on mainline llama.cpp. Direct invocation —
# launcher.build_command is Gemma-specific (--jinja, --reasoning on,
# --reasoning-format deepseek). Qwen wants different flags. Mirrors the
# engine's `-c 32768 -np 128 --kv-unified` shape so single-call latency
# stays sub-second; the unified KV pool dynamically allocates across
# concurrent sessions.
.PHONY: llama-eval
llama-eval: $(EVAL_MODEL)
	$(EVAL_LLAMA_BIN) -m $(EVAL_MODEL) --host $(LLAMA_HOST) --port $(EVAL_LLAMA_PORT) -c 32768 --jinja -ngl 99 -np 128 --kv-unified

.PHONY: sidecar
sidecar:
	cd npc_service && LAUNCH_LLAMA=false uv run python -m npc_service

.PHONY: web
web:
	cd web && npm run dev

.PHONY: test
test:
	cd npc_service && uv run pytest

# Scenario linter. Runs against `npc_service/scenario.yaml` by default.
# Loads the scenario via the Pydantic models (so `_cross_references`
# parse errors surface first), then applies the author-facing checks the
# model doesn't catch on its own. Fast — spec rules only. The state-space
# analyzer is opt-in via `--analyzer`; run `make analyze-scenario` to
# invoke it as its own target. See `npc_service/scripts/lint_rules.md`
# for what each check does and why.
.PHONY: lint-scenario
lint-scenario:
	cd npc_service && uv run python scripts/lint_scenario.py scenario.yaml

# Static reachability analyzer (ASP-backed via clingo). Computes which
# beats are reachable from boot, per-quality value bounds, and orphan/
# bound-mismatch findings that the spec lint can't see. Sub-second on
# the production scenario thanks to clingo's brave-mode consequence
# reasoning — see DendryScope (Robison & Mateas, AIIDE 2023) for the
# encoding precedent. Run before commit / in CI alongside `lint-scenario`.
.PHONY: analyze-scenario
analyze-scenario:
	cd npc_service && uv run python scripts/analyze_scenario.py scenario.yaml

# LLM playtester. Probes /health on BOTH llama servers (engine on :8080,
# eval on :8081) before booting; errors clearly if either isn't up. Boot
# them in separate terminals first via `make llama-playtest` (bumped KV
# pool) and `make llama-eval`. Installs the optional pydantic-evals dep on
# first run via `uv sync --extra playtest`. PYTHONIOENCODING=utf-8 keeps
# pydantic-evals' rich-rendered evaluation summary from crashing on the
# Windows cp1252 console when it prints `✓` / `✗` glyphs. `python -u`
# disables stdout buffering so progress is visible while the run is in
# flight (default-buffered subprocesses don't flush until exit).
.PHONY: playtest
playtest:
	cd npc_service && uv sync --extra playtest && PYTHONIOENCODING=utf-8 uv run python -u -m scripts.playtest

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
