"""Test config: silence Logfire console output and prevent any accidental
llama-server spawn during unit tests."""
import os

# Match v1's conftest: keep pytest output uncluttered.
os.environ.setdefault("HANDLER_LOGFIRE_CONSOLE", "false")
# Tag any spans this suite emits as env:test so they're filterable out of
# real playthroughs in the Logfire UI (see main.py logfire.configure).
os.environ.setdefault("HANDLER_ENV", "test")
