# Backdrop proofing — iteration with Claude

A small Gemini-backed pipeline for the location backdrops + map in `world/`.
You author and review; **Claude edits the prompts** when you ask.

## The loop

1. `make proof` → open http://127.0.0.1:8765
2. Review the plates. Click any image for fullscreen (opens in a new tab).
3. **Don't like one?** Tell Claude in chat — e.g.
   *"Tan-y-coed is too sunny. Push the prompt seed toward late dusk and emphasise the Land Rover."*
   Claude edits `world/backdrops/IMAGES.md`.
4. Refresh the proof page (it re-reads the manifest), then click **pull a new proof** on that plate. The new image lands as a variant; the canonical isn't touched.
5. When a variant is right, click **use as canonical →** on it. The old canonical is preserved in `_variants/`.

## Files

| Path | What |
|---|---|
| `world/backdrops/IMAGES.md` | **Source of truth.** Per-plate Subject / Composition / Required props / Avoid / Prompt seed. Plus a top-of-file `## Global aesthetic` block that prepends every prompt. |
| `world/backdrops/{id}.png` | Canonical plate the game serves. |
| `world/backdrops/_variants/{id}/*.png` | Every re-roll. Each has a `.txt` sidecar with the exact prompt + references used. |
| `world/map.png` and `_map_variants/` | Same shape, for the parish map. |
| `scripts/generate_image.py` | CLI generator. `--only id1,id2`, `--variant`, `--force`, `--dry-run`, `--model flash|pro`. |
| `scripts/proof_server.py` + `proof_index.html` | The proofing UI. |

## Editing the prompt — phrases that work with Claude

Each ###-block in `IMAGES.md` has six fields. Tell Claude which to change.

| What you want | Say something like |
|---|---|
| Tweak the poetic prompt seed | *"In the_lane, rewrite the prompt seed to emphasise the wet tarmac and add a single rook on the fingerpost."* |
| Add/remove a required prop | *"Add a small parish noticeboard to the_post_office required props."* |
| Loosen a constraint | *"Drop the 'no cars' line from knighton Avoid — there should be parked cars on the high street."* |
| Change the overall mood | *"In the Global aesthetic block, push palette toward bluer dusk and away from ochres."* — this affects every subsequent generation. |
| Reword for tone | *"Make the prompt seed for st_caradocs more Jem-Southam, less cinematic."* |

After Claude saves the file, refresh `http://127.0.0.1:8765` and rebuild the affected plates.

## References (style anchors) for consistency

Each plate in the proof UI shows thumbnails of every other canonical plate under **style anchors**. Click one (or several) to lock its aesthetic onto the next rebuild — Gemini reads them as reference images alongside your prompt.

Use this when a series feels inconsistent. The typical workflow:

1. Pick one plate you love. Call it the anchor.
2. On every other plate, select the anchor (and optionally one or two more) as references.
3. Rebuild. The new variants inherit the anchor's lighting, palette, mood.

You can stack up to 14 references per plate (Gemini's cap). Selection persists per plate in `localStorage` — refresh-safe.

## Models

- **flash** = `gemini-3.1-flash-image-preview` — fast (~20s/plate), good for iteration. Default.
- **pro** = `gemini-3-pro-image-preview` — slower, more obedient. Use when a plate is dialled in and you want the final-quality pass.

Per-plate selector lives next to the rebuild button.

## CLI cheats

For batch work outside the UI:

```bash
# fill in any missing canonicals
uv run python scripts/generate_image.py

# re-roll one plate as a variant (preserves canonical)
uv run python scripts/generate_image.py --only tan_y_coed --variant

# what would I send? (no API call)
uv run python scripts/generate_image.py --only tan_y_coed --dry-run

# do everything fresh as variants, in pro
uv run python scripts/generate_image.py --variant --model pro
```

`GEMINI_API_KEY` is read from `.env` at the repo root (auto-loaded).
