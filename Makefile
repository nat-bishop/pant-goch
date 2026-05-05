MODEL       := models/ggml-org-gemma-4-E4B-it-Q8_0.gguf
MODEL_URL   := https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
LLAMA_BIN   := vendor/llama.cpp/llama-server.exe
LLAMA_HOST  := 127.0.0.1
LLAMA_PORT  := 8080

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Inbox targets:"
	@echo "  make run             Boot sidecar (spawns llama) + web dev server in parallel"
	@echo "  make download-model  Fetch the Gemma 4 E4B Q8_0 GGUF (~8 GB)"
	@echo "  make llama           Run llama-server               (terminal 1)"
	@echo "  make sidecar         Run Python sidecar, no llama   (terminal 2)"
	@echo "  make web             Run Vite dev server            (terminal 3)"
	@echo "  make test            Run pytest"
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

.PHONY: llama
llama: $(MODEL)
	$(LLAMA_BIN) -m $(MODEL) --host $(LLAMA_HOST) --port $(LLAMA_PORT) -c 8192 --jinja --reasoning off -ngl 99

.PHONY: sidecar
sidecar:
	cd npc_service && LAUNCH_LLAMA=false uv run python -m npc_service

.PHONY: web
web:
	cd web && npm run dev

.PHONY: test
test:
	cd npc_service && uv run pytest

.PHONY: typecheck
typecheck:
	cd web && npm run typecheck

.PHONY: clean
clean:
	rm -f logs/*.log
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: kill-zombies
kill-zombies:
	@powershell -Command 'Get-NetTCPConnection -LocalPort 8000,8080,5173,5174,5175,5176 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $$_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Host "killed any processes holding :8000/:8080/:5173-5176"'
