"""Test config: silence Logfire console output and prevent any accidental
llama-server spawn during unit tests."""
import os

# Match v1's conftest: keep pytest output uncluttered.
os.environ.setdefault("HANDLER_LOGFIRE_CONSOLE", "false")
