"""Standalone llama-server entry point used by `make llama`.

Exists so the Makefile and the sidecar share one definition of llama-server's
argv (`launcher.build_command`). Without this, the Makefile's hardcoded flags
drift from `Settings` — e.g., `-np 8` in Make vs. `llama_n_parallel=128` in
config.py, which silently bottlenecks the playtester.

Stdio is inherited so llama's logs stream to the foreground terminal and
Ctrl-C goes straight to llama. No Job Object needed here — Python is just
parked in subprocess.run() until llama exits, and the terminal's signal
handling closes both together.
"""
from __future__ import annotations

import subprocess
import sys

from .launcher import build_command, resolve_config


def main() -> int:
    cfg = resolve_config()
    cmd = build_command(cfg)
    print(f"# {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
