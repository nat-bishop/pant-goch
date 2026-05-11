"""Standalone llama-server entry point used by `make llama` and `make llama-eval`.

Exists so the Makefile and the sidecar share one definition of llama-server's
argv (`launcher.build_command`). Without this, the Makefile's hardcoded flags
drift from `config.py` — e.g., `-np 8` in Make vs. `LLAMA_N_PARALLEL=128` in
config.py, which silently bottlenecks the playtester.

`--role engine` (default) boots the engine llama from `resolve_config`.
`--role eval` boots the eval llama (Qwen on :8081) from `resolve_eval_config`.
Both go through the same `build_command`; the only family-specific axis is
the reasoning flags (engine emits them, eval doesn't — see launcher.py).

Stdio is inherited so llama's logs stream to the foreground terminal and
Ctrl-C goes straight to llama. No Job Object needed here — Python is just
parked in subprocess.run() until llama exits, and the terminal's signal
handling closes both together.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .launcher import build_command, resolve_config, resolve_eval_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Boot llama-server (engine or eval).")
    parser.add_argument(
        "--role",
        choices=("engine", "eval"),
        default="engine",
        help="Which llama to boot. `engine` (default) is the SUT (Gemma 4 on :8080); "
             "`eval` is the playtester's synthetic-player + judge model (Qwen 35B-A3B on :8081).",
    )
    args = parser.parse_args(argv)

    cfg = resolve_eval_config() if args.role == "eval" else resolve_config()
    cmd = build_command(cfg)
    print(f"# [{args.role}] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
