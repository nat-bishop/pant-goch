"""Blinded A/B human rater — a tiny local server behind a static page.

    uv run python scripts/ab_rater.py --run evals/artifacts/ab/<ts>

Serves `ab_rater.html`, reads the run's blinded `pairs.jsonl`, and appends each
verdict to `human_ratings.jsonl` in the same dir (a browser can't write files, so
this minimal backend persists them — same shape as scripts/proof_server.py). The
arm behind each A/B is hidden from you (it lives only in `pairs.key.json`); rerun
`evals play --experiment <name> --human` afterwards to merge your ratings into the
report (human win-rate + agreement with the LLM judge).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 and abort on a non-ascii glyph; force UTF-8
# so any logged transcript/persona text (or the startup line) can't crash the server.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

HTML_PATH = Path(__file__).resolve().parent / "ab_rater.html"

app = FastAPI(title="HANDLER A/B rater")
RUN_DIR: Path = Path()  # set in main()


class Rating(BaseModel):
    pair_id: str
    choice: str  # "A" | "B" | "tie"


def _pairs() -> list[dict]:
    path = RUN_DIR / "pairs.jsonl"
    if not path.exists():
        raise HTTPException(404, f"no pairs.jsonl in {RUN_DIR}")
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _rated_ids() -> set[str]:
    path = RUN_DIR / "human_ratings.jsonl"
    if not path.exists():
        return set()
    return {json.loads(ln)["pair_id"]
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/pairs")
def api_pairs() -> JSONResponse:
    pairs = _pairs()
    rated = _rated_ids()
    return JSONResponse({
        "run": str(RUN_DIR),
        "total": len(pairs),
        "rated": sorted(rated),
        "pairs": [p for p in pairs if p["pair_id"] not in rated],
    })


@app.post("/api/rate")
def api_rate(rating: Rating) -> JSONResponse:
    if rating.choice not in ("A", "B", "tie"):
        raise HTTPException(400, "choice must be A, B, or tie")
    # Append-only; idempotency is the caller's concern (the UI skips rated pairs).
    with (RUN_DIR / "human_ratings.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"pair_id": rating.pair_id, "choice": rating.choice}) + "\n")
    done = len(_rated_ids())
    return JSONResponse({"ok": True, "rated": done, "total": len(_pairs())})


def main() -> None:
    global RUN_DIR
    ap = argparse.ArgumentParser(description="HANDLER blinded A/B human rater")
    ap.add_argument("--run", type=Path, required=True, help="A/B run dir (with pairs.jsonl)")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    RUN_DIR = args.run.resolve()
    if not (RUN_DIR / "pairs.jsonl").exists():
        raise SystemExit(f"no pairs.jsonl in {RUN_DIR} — run `evals play --experiment <name>` first")

    import uvicorn
    print(f"rating {RUN_DIR}  ->  http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
