import contextlib
import sys

# Force UTF-8 on stdout/stderr before importing anything that may log. Windows
# consoles default to cp1252, and Logfire's console exporter (via rich) writes
# box-drawing glyphs like U+2502 that crash the cp1252 encoder mid-span-export.
# `make playtest` solves this with PYTHONIOENCODING=utf-8; doing it here covers
# every launch path (`make run`, `make sidecar`, direct `python -m handler`).
with contextlib.suppress(AttributeError):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn

from .config import Settings


def main() -> None:
    cfg = Settings()
    uvicorn.run(
        "handler.main:app",
        host="127.0.0.1",
        port=cfg.handler_port,
        log_level=cfg.handler_log_level,
    )


if __name__ == "__main__":
    main()
