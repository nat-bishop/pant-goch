r"""Generate location backdrops and the map for world/ from IMAGES.md.

The manifest at world/backdrops/IMAGES.md is the source of truth. Each
`### \`<name>.png\`` block declares Status / Subject / Composition / Required
props / Avoid / Prompt seed. We assemble a per-image prompt as:

    global_aesthetic + prompt_seed + composition + required_props + avoid

and feed it to Gemini's image-generation model. Outputs land at
world/backdrops/{id}.png, except `map.png` which goes to world/map.png.

Existing files are never overwritten unless --force is passed. Use --only to
regenerate a subset, and --dry-run to inspect the assembled prompts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Surface GEMINI_API_KEY from .env if present (the project keeps it there;
# requiring shell export every session would be friction).
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

MANIFEST = REPO_ROOT / "world" / "backdrops" / "IMAGES.md"
BACKDROPS_DIR = REPO_ROOT / "world" / "backdrops"
VARIANTS_DIR = BACKDROPS_DIR / "_variants"
MAP_PATH = REPO_ROOT / "world" / "map.png"
MAP_VARIANTS_DIR = REPO_ROOT / "world" / "_map_variants"

MODELS = {
    "flash": "gemini-3.1-flash-image-preview",
    "pro": "gemini-3-pro-image-preview",
}
DEFAULT_MODEL = "flash"

# Closest fixed aspect ratio to both the 16:9 backdrops (2560×1440) and the
# 1.833:1 map (2816×1536). 2K @ 16:9 yields 2752×1536; the author can crop.
ASPECT_RATIO = "16:9"
IMAGE_SIZE = "2K"


@dataclass
class ImageEntry:
    id: str
    filename: str
    subject: str
    composition: str
    required_props: str
    avoid: str
    prompt_seed: str


_FIELDS = ("Subject", "Composition", "Required props", "Avoid", "Prompt seed")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _strip_italic(text: str) -> str:
    text = text.strip()
    if text.startswith("*") and text.endswith("*"):
        text = text[1:-1].strip()
    return text


def parse_manifest(text: str) -> tuple[str, list[ImageEntry]]:
    """Return (global_aesthetic, entries) from the manifest text."""
    m = re.search(r"##\s+Global aesthetic\s*\n(.*?)(?=\n---|\n##\s)", text, re.S)
    if not m:
        raise ValueError("Could not find '## Global aesthetic' section in IMAGES.md")
    global_aesthetic = m.group(1).strip()

    block_re = re.compile(
        r"^###\s+`([^`]+\.png)`.*?\n(.*?)(?=^###\s|^##\s|\Z)",
        re.S | re.M,
    )
    field_res = {
        name: re.compile(
            rf"-\s+\*\*{re.escape(name)}:\*\*\s*(.+?)(?=\n-\s+\*\*|\n\s*---|\n##\s|\Z)",
            re.S,
        )
        for name in _FIELDS
    }

    entries: list[ImageEntry] = []
    for bm in block_re.finditer(text):
        filename = bm.group(1)
        body = bm.group(2)
        values: dict[str, str] = {}
        for name, pat in field_res.items():
            fm = pat.search(body)
            if not fm:
                raise ValueError(f"Missing **{name}:** in entry `{filename}`")
            values[name] = _collapse(fm.group(1))
        entries.append(
            ImageEntry(
                id=filename[: -len(".png")],
                filename=filename,
                subject=values["Subject"],
                composition=values["Composition"],
                required_props=values["Required props"],
                avoid=values["Avoid"],
                prompt_seed=_strip_italic(values["Prompt seed"]),
            )
        )
    if not entries:
        raise ValueError("No `### \\`*.png\\`` entries found in IMAGES.md")
    return global_aesthetic, entries


def build_prompt(global_aesthetic: str, entry: ImageEntry) -> str:
    return (
        f"{global_aesthetic}\n\n"
        f"{entry.prompt_seed}\n\n"
        f"Composition: {entry.composition}\n\n"
        f"Required props: {entry.required_props}\n\n"
        f"Avoid: {entry.avoid}"
    )


def canonical_path(entry: ImageEntry) -> Path:
    if entry.filename == "map.png":
        return MAP_PATH
    return BACKDROPS_DIR / entry.filename


def variants_dir(entry: ImageEntry) -> Path:
    if entry.filename == "map.png":
        return MAP_VARIANTS_DIR
    return VARIANTS_DIR / entry.id


def variant_path(entry: ImageEntry, model_short: str, when: dt.datetime | None = None) -> Path:
    when = when or dt.datetime.now()
    stamp = when.strftime("%Y%m%d-%H%M%S")
    return variants_dir(entry) / f"{stamp}__{model_short}.png"


def _extract_png_bytes(response) -> bytes:
    for cand in response.candidates or []:
        if not cand.content or not cand.content.parts:
            continue
        for part in cand.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data
    raise RuntimeError("Response contained no image data")


def generate_one(
    client,
    model: str,
    prompt: str,
    references: list[bytes] | None = None,
) -> bytes:
    """Generate an image. `references` are raw PNG bytes used as style anchors.

    Gemini recommends including previously-generated images alongside the text
    prompt to maintain visual consistency across a series — up to 14 reference
    images on flash, slightly fewer on pro. We pass them as inline-data Parts
    in the contents list.
    """
    from google.genai import types

    contents: list = [prompt]
    for ref in references or []:
        contents.append(
            types.Part(inline_data=types.Blob(mime_type="image/png", data=ref))
        )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=ASPECT_RATIO,
                image_size=IMAGE_SIZE,
            ),
        ),
    )
    return _extract_png_bytes(response)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(MODELS),
        default=DEFAULT_MODEL,
        help=f"flash = {MODELS['flash']} (default, fast iteration); "
             f"pro = {MODELS['pro']} (final quality)",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated ids to (re)generate (e.g. tan_y_coed,knighton,map). "
             "Default: every entry whose output file is missing.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the canonical {id}.png when it already exists.",
    )
    p.add_argument(
        "--variant",
        action="store_true",
        help="Always save to the _variants/<id>/ subdirectory with a "
             "{timestamp}__{model}.png filename. Never touches the canonical "
             "file. Use this when you want to re-roll an image without "
             "destroying the existing one.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assembled prompt for each selected entry without "
             "calling the API.",
    )
    args = p.parse_args()

    manifest_text = MANIFEST.read_text(encoding="utf-8")
    global_aesthetic, entries = parse_manifest(manifest_text)
    by_id = {e.id: e for e in entries}

    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            print(
                f"error: unknown id(s): {', '.join(unknown)}\n"
                f"known: {', '.join(sorted(by_id))}",
                file=sys.stderr,
            )
            return 2
        selected = [by_id[w] for w in wanted]
    elif args.variant:
        selected = entries
    else:
        selected = [e for e in entries if not canonical_path(e).exists()]

    if not selected:
        print("Nothing to do — every output file already exists. "
              "Use --only or --force to regenerate.")
        return 0

    model_id = MODELS[args.model]
    print(f"Model:    {model_id}")
    print(f"Selected: {', '.join(e.id for e in selected)} ({len(selected)})")
    print()

    if args.dry_run:
        for entry in selected:
            if args.variant:
                out = variants_dir(entry).relative_to(REPO_ROOT)
                target = f"{out}/<timestamp>__{args.model}.png"
            else:
                target = str(canonical_path(entry).relative_to(REPO_ROOT))
            print(f"=== {entry.id} → {target} ===")
            print(build_prompt(global_aesthetic, entry))
            print()
        return 0

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("error: GEMINI_API_KEY is not set in the environment.", file=sys.stderr)
        return 2

    from google import genai
    client = genai.Client(api_key=api_key)

    failures = 0
    for entry in selected:
        if args.variant:
            out = variant_path(entry, args.model)
        else:
            out = canonical_path(entry)
            if out.exists() and not args.force:
                print(f"- skip {entry.id}: {out.relative_to(REPO_ROOT)} exists "
                      f"(pass --force to overwrite, or --variant to save a copy)")
                continue
        prompt = build_prompt(global_aesthetic, entry)
        start = time.monotonic()
        try:
            png_bytes = generate_one(client, model_id, prompt)
        except Exception as exc:
            failures += 1
            print(f"x {entry.id}: {exc}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png_bytes)
        elapsed = time.monotonic() - start
        print(f"+ generated {out.relative_to(REPO_ROOT)} in {elapsed:.1f}s")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
