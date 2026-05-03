import os

# Keep Logfire console output off during tests so pytest captures stay clean.
# Override with `NPC_LOGFIRE_CONSOLE=true uv run pytest -s ...` when debugging.
# Must run before `npc_service.main` is imported, since `_BOOT_SETTINGS` reads
# env vars eagerly at module import — pytest auto-loads conftest.py first.
os.environ.setdefault("NPC_LOGFIRE_CONSOLE", "false")
