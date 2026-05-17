# Port the Stage frontend to handler/web

A finished UI prototype sits in this folder. Your job is to make `handler/web/` look and behave like it.

## Run the reference

```
cd design && python -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/mockup-stage/
```

Click through it. Hover the dotted-underline phrases. Switch the LOCATION / MAP / NOTES tabs in the left pane. Pick a "leave for" choice and watch the walking interstitial. The mockup is the source of truth for layout, motion, typography, and behaviour.

## How to use it

- `styles.css` — every colour, font, and dimension. The CSS variables at the top are the design tokens.
- `app.jsx` — the React patterns you need: two-pane grid, image cross-fade, hover popup, mode tabs, walking interstitial, page-turn scroll. Read top-to-bottom; the file is small.
- `index.html` — the Google Fonts URL (Cormorant Garamond + Old Standard TT + Courier Prime).

## What to do

Restructure `handler/web/src/` so the real frontend renders the same composition driven by the live SSE backend. Read `handler/CLAUDE.md` first for the engine constraints you must respect (two-stage emission, server-side word-pacing, etc.). The mockup's data layer (`scene-data.js`, `window.GLOSSES`) is stub — wire to the real `api.ts` and store instead.

## On glosses

The mockup fakes them with pre-authored `[phrase|gloss_id]` markup. The real `ProsePart.prose` has no annotation field and Phase 2 emits no markup. 
Think about how we could do this and where it would live. The simplest iteration is just to have a list of facts somewhere, and maybe have alias's on those facts, and when that prose appears in the text we do a model that appear on hover.

Interview me about any parts which you are not sure on.