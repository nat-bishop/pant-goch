MODEL       := models/gemma-4-E4B-it-Q8_0.gguf
MODEL_URL   := https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
LLAMA_BIN   := vendor/llama.cpp/llama-server.exe
LLAMA_HOST  := 127.0.0.1
LLAMA_PORT  := 8080

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Smallville targets:"
	@echo "  make run             Single-bootstrap: Godot spawns Python spawns llama-server"
	@echo "  make dev             Print the 3-terminal dev workflow"
	@echo "  make download-model  Fetch the Gemma 4 E4B Q8_0 GGUF (~8 GB)"
	@echo "  make sync            uv sync inside npc_service/"
	@echo "  make llama           Run llama-server               (terminal 1)"
	@echo "  make sidecar         Run Python sidecar, no llama   (terminal 2)"
	@echo "  make godot           Run Godot with --no-sidecar    (terminal 3)"
	@echo "  make health          curl the sidecar /health endpoint"
	@echo "  make smoke           POST a sample action to the sidecar"
	@echo "  make clean           Remove __pycache__ and logs/*.log"

.PHONY: run
run:
	godot --path godot_project

.PHONY: dev
dev:
	@echo "Open three terminals and run:"
	@echo "  1) make llama"
	@echo "  2) make sidecar"
	@echo "  3) make godot"

$(MODEL):
	mkdir -p models
	curl -L -o $(MODEL) $(MODEL_URL)

.PHONY: download-model
download-model: $(MODEL)

.PHONY: sync
sync:
	cd npc_service && uv sync

.PHONY: llama
llama: $(MODEL)
	$(LLAMA_BIN) -m $(MODEL) --host $(LLAMA_HOST) --port $(LLAMA_PORT) -c 8192 --jinja --reasoning off -ngl 99

.PHONY: sidecar
sidecar:
	cd npc_service && PYTHONPATH=.. LAUNCH_LLAMA=false uv run python -m npc_service

.PHONY: godot
godot:
	godot --path godot_project --no-sidecar

.PHONY: health
health:
	curl -s http://127.0.0.1:8000/health

.PHONY: smoke
smoke:
	curl -N -X POST http://127.0.0.1:8000/action \
	  -H "Content-Type: application/json" \
	  -d '{"type":"empathize","detail":""}'

.PHONY: clean
clean:
	rm -f logs/*.log
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
