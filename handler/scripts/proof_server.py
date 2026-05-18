"""Proofing server for handler/world image assets.

A small FastAPI app that serves a gallery of every entry in IMAGES.md alongside
its current canonical preview and any rebuilt variants. The author can trigger
a regeneration (gemini flash or pro) per entry; new images land in the
`_variants/<id>/` subfolder so the existing canonical is never clobbered.

Run with `make proof` (or `uv run python handler/scripts/proof_server.py`).
The server listens on http://127.0.0.1:8765 by default; set PROOF_PORT to
override.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# Sibling import: scripts/ isn't a package, so add it to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_image as gi  # noqa: E402

REPO_ROOT = gi.REPO_ROOT
HTML_PATH = Path(__file__).resolve().parent / "proof_index.html"

app = FastAPI(title="Pant Goch backdrop proofs")


class RebuildRequest(BaseModel):
    id: str
    model: str  # "flash" | "pro"
    prompt: str | None = None  # if omitted, assemble from manifest
    references: list[str] = []   # ids of OTHER plates whose canonical PNG
                                 # bytes should ride along as style anchors


class PromoteRequest(BaseModel):
    id: str
    filename: str  # variant filename to copy into the canonical slot


# ----- helpers --------------------------------------------------------------

def _variant_records(entry: gi.ImageEntry) -> list[dict]:
    d = gi.variants_dir(entry)
    if not d.exists():
        return []
    records = []
    for p in sorted(d.glob("*.png")):
        stem = p.stem
        model_short = "unknown"
        iso = ""
        if "__" in stem:
            stamp_str, _, model_short = stem.partition("__")
            try:
                when = dt.datetime.strptime(stamp_str, "%Y%m%d-%H%M%S")
                iso = when.isoformat(timespec="seconds")
            except ValueError:
                pass
        records.append(
            {
                "url": f"/img/variants/{entry.id}/{p.name}",
                "filename": p.name,
                "model": model_short,
                "when": iso,
                "size_bytes": p.stat().st_size,
            }
        )
    records.sort(key=lambda r: r["when"], reverse=True)
    return records


def _canonical_record(entry: gi.ImageEntry) -> dict:
    p = gi.canonical_path(entry)
    return {
        "url": f"/img/canonical/{entry.id}?t={int(p.stat().st_mtime)}"
        if p.exists()
        else None,
        "exists": p.exists(),
        "path": str(p.relative_to(REPO_ROOT)) if p.exists() else None,
    }


def _load_entries() -> tuple[str, list[gi.ImageEntry]]:
    return gi.parse_manifest(gi.MANIFEST.read_text(encoding="utf-8"))


_client = None


def _get_client():
    """Lazily build (and cache) the Gemini client.

    google-genai uses httpx under the hood, which is thread-safe — so the one
    client object is shared across FastAPI's threadpool workers. This avoids
    paying the client-construction cost on every rebuild and means concurrent
    rebuilds don't serialize on client setup.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(500, "GEMINI_API_KEY is not set")
        from google import genai
        _client = genai.Client(api_key=api_key)
    return _client


# ----- API ------------------------------------------------------------------

@app.get("/api/manifest")
def api_manifest():
    global_aesthetic, entries = _load_entries()
    out = []
    for e in entries:
        out.append(
            {
                "id": e.id,
                "filename": e.filename,
                "subject": e.subject,
                "composition": e.composition,
                "required_props": e.required_props,
                "avoid": e.avoid,
                "prompt_seed": e.prompt_seed,
                "prompt": gi.build_prompt(global_aesthetic, e),
                "canonical": _canonical_record(e),
                "variants": _variant_records(e),
            }
        )
    return {
        "global_aesthetic": global_aesthetic,
        "models": gi.MODELS,
        "entries": out,
    }


@app.post("/api/rebuild")
def api_rebuild(req: RebuildRequest):
    if req.model not in gi.MODELS:
        raise HTTPException(400, f"unknown model '{req.model}'")
    global_aesthetic, entries = _load_entries()
    by_id = {e.id: e for e in entries}
    if req.id not in by_id:
        raise HTTPException(404, f"unknown id '{req.id}'")
    entry = by_id[req.id]

    client = _get_client()
    prompt = req.prompt if (req.prompt and req.prompt.strip()) else gi.build_prompt(global_aesthetic, entry)

    reference_bytes: list[bytes] = []
    resolved_refs: list[str] = []
    for ref_id in req.references[:14]:  # API caps at 14
        if ref_id == entry.id or ref_id not in by_id:
            continue
        ref_entry = by_id[ref_id]
        p = gi.canonical_path(ref_entry)
        if not p.exists():
            continue
        reference_bytes.append(p.read_bytes())
        resolved_refs.append(ref_id)

    try:
        png_bytes = gi.generate_one(
            client, gi.MODELS[req.model], prompt, references=reference_bytes
        )
    except Exception as exc:
        raise HTTPException(500, f"generation failed: {exc}")

    out = gi.variant_path(entry, req.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    # Sidecar: save the exact prompt + reference list so each variant is
    # self-documenting and reproducible.
    sidecar = prompt
    if resolved_refs:
        sidecar += "\n\n---\nreferences: " + ", ".join(resolved_refs)
    out.with_suffix(".txt").write_text(sidecar, encoding="utf-8")

    return {
        "id": entry.id,
        "variant": {
            "url": f"/img/variants/{entry.id}/{out.name}",
            "filename": out.name,
            "model": req.model,
            "when": dt.datetime.now().isoformat(timespec="seconds"),
            "size_bytes": len(png_bytes),
            "references": resolved_refs,
        },
    }


@app.post("/api/promote")
def api_promote(req: PromoteRequest):
    """Copy a variant into the canonical slot.

    Before overwriting, archive the current canonical (if any) into
    `_variants/<id>/archived-<timestamp>.png` so it is never lost.
    """
    _, entries = _load_entries()
    by_id = {e.id: e for e in entries}
    if req.id not in by_id:
        raise HTTPException(404, f"unknown id '{req.id}'")
    entry = by_id[req.id]

    fname = Path(req.filename).name  # strip any path
    if not fname.endswith(".png"):
        raise HTTPException(400, "filename must be a .png")
    src = gi.variants_dir(entry) / fname
    if not src.exists():
        raise HTTPException(404, f"variant not found: {fname}")

    canonical = gi.canonical_path(entry)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical.exists():
        archive_name = f"archived-{dt.datetime.now():%Y%m%d-%H%M%S}.png"
        archive_dst = gi.variants_dir(entry) / archive_name
        archive_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical, archive_dst)
    shutil.copy2(src, canonical)
    return {"id": entry.id, "canonical": _canonical_record(entry)}


# ----- static image serving -------------------------------------------------

@app.get("/img/canonical/{name}")
def img_canonical(name: str):
    if name == "map":
        p = gi.MAP_PATH
    else:
        p = gi.BACKDROPS_DIR / f"{name}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/img/variants/{entry_id}/{filename}")
def img_variant(entry_id: str, filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400)
    if not filename.endswith(".png"):
        raise HTTPException(400)
    if entry_id == "map":
        d = gi.MAP_VARIANTS_DIR
    else:
        d = gi.VARIANTS_DIR / entry_id
    p = d / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


# ----- HTML -----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_PATH.exists():
        return HTMLResponse(
            "<h1>proof_index.html is missing</h1>",
            status_code=500,
        )
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


def main() -> int:
    import uvicorn

    port = int(os.environ.get("PROOF_PORT", "8765"))
    print(f"Proofs at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
