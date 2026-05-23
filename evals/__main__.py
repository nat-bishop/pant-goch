"""`python -m evals` → the unified Typer eval CLI.

Repo-root invocation (`uv run python -m evals <cmd>`) puts `evals/` on the path,
the same way the old `python -m evals.run` entry did — so this works without the
top-level `evals` package being part of the installed wheel.
"""
from .cli import app

if __name__ == "__main__":
    app()
