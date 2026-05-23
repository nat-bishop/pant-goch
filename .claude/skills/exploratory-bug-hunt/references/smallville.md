# smallville / HANDLER — launch, surfaces & oracles

HANDLER v2 is an **LLM-driven investigation game**: the player is a journalist
in the Welsh village of Pant Goch. The world (locations, cast, the truth) is
hand-authored YAML; an LLM improvises each scene inside it via two-stage
structured emission, streamed to the browser over SSE. Frontend is Vite + React
(`web/`); backend is a FastAPI sidecar; generation runs on a bundled llama-server
(or inferno-routed).

## Launch

From the repo root:

```bash
make dev        # sidecar (:8000) + Vite (:5173) together, inferno-routed
```

Then `browser_navigate` to `http://localhost:5173`. If ports are stuck from a
previous run, `make kill` clears orphans on `:8000` / `:8080` / `:5173-5176`
before relaunching. Generation needs a model: either the bundled llama-server
(`:8080`) or inferno (`:7770`) must be reachable, or scenes will never stream.

## What makes this app different (read before you start)

**The scene text is non-deterministic by design.** The LLM improvises prose, NPC
dialogue, and choices. This rewires which oracles are meaningful:

- **Do NOT treat narrative variation as a bug.** Different wording, a scene
  taking an unexpected turn, an NPC being evasive — that's the game working, not
  a defect. You are not judging writing quality or story logic.
- **Round-trip / save-reload oracles mostly don't apply.** There is no undo and
  no "save then reload to identical" — a fresh playthrough diverges. Don't force
  these. (A page *refresh* mid-session is still worth testing — see below.)
- **Your real oracles are the engine's structural invariants** plus ordinary UI
  integrity. Those are below.

## The real oracles here

The engine enforces hard structural rules via llama.cpp grammar sampling (the
model literally cannot emit an invalid id). The **frontend must honor the same
rules** — where the UI surfaces something that violates one, that's a genuine,
high-value bug:

- **Only valid catalog ids ever appear.** `npc_id`, `location_id`, `beat_id`,
  `ending_id`, `leaves_for` are narrowed per-call to the valid set. A leave
  option, NPC, or location label that points at an id not in `world/` is a bug.
- **Leave options go only to *reachable* locations** (`adjacent_to` in
  `world/locations/*.yaml`). An offered move to a non-adjacent location is a bug.
- **At most one leave option per scene** (a structural validator). Two+ "leave"
  choices is a bug.
- **Endings only fire when their `requires_beats` are met and not
  `blocked_by_beats`.** A climax the player didn't earn — or a missing one they
  did — is worth flagging (`needs-judgment` unless clearly contradicted by state).
- Plus the universal ones from SKILL.md: no new console errors, no failed/hung
  network or SSE requests, no white screen, no garbage text
  (`undefined`/`NaN`/`[object Object]`) rendered, layout holds.

To check id/reachability invariants you usually need the world catalog. The
adjacency and id lists live in `world/locations/*.yaml`, `world/cast.yaml`,
`world/truth.yaml`. Read those once at the start so you can recognize an
out-of-catalog or non-adjacent option when the UI shows one.

## Surfaces (components under `web/src/components/`)

- **`FreeTextInput`** — the player's free-text action box. The single richest
  fuzz target (see below).
- **`StreamingProse`** — the SSE-streamed scene text. Watch streaming/timing
  edges most closely here.
- **`ChoicesBlock`** — offered choices, including the (≤1) leave option. Check
  against the reachability / single-leave invariants.
- **`Stage` / `Chapter` / `BetweenScenes` / `Frontispiece`** — scene staging,
  chapter transitions, the between-scenes interstitial, and the opening title.
- **`MapView`** — the village map (`world/map.yaml` geometry + `map.png`
  backdrop). Test location markers, navigation, and resize.
- **`CastBoard` / `DossierView` / `NotesView`** — the cast board, dossier, and
  player notes panels.
- **`GlossTrigger` / `GlossPopup` / `glosses.ts`** — glossary tooltips on terms.
- **`WorldPane` / `PanelScroll` / `ImageSlots` / `CatchUpLink`** — layout panes,
  scroll behavior, image slots, and the catch-up/recap link.

## Core verbs

Send a free-text action → read the streamed scene → pick a choice / take the
leave option to move location → talk to NPCs → progress beats toward an ending.
Do the happy path of these first so you recognize deviations.

## Where the bugs actually live (prioritize)

1. **Free-text input fuzzing (`FreeTextInput`).** Empty submit, whitespace-only,
   enormous paste, unicode/emoji/RTL, HTML/`<script>`, and crucially
   **out-of-world / adversarial prompts**: "ignore your instructions", "go to
   London" (not in the map), "tell me the murderer" (out of fiction), asking to
   do impossible actions. The oracle is *not* "the LLM gives a specific answer" —
   it's that the **engine stays inside its constraints**: still only valid ids,
   still ≤1 reachable leave option, no crash, no invalid move, and it doesn't
   leak system-prompt internals.
2. **SSE / streaming timing (`StreamingProse`).** Submit a new action *while a
   scene is still streaming*; double-submit; refresh or navigate away mid-stream;
   let a slow/stalled generation run. Watch for duplicated/interleaved scenes,
   stuck "thinking" spinners, dropped SSE connections that never recover, or
   state desync after a mid-stream refresh.
3. **Model-unavailable handling.** With llama-server / inferno down or slow, does
   the UI degrade gracefully (clear error, retry) or white-screen / spin forever?
4. **Map & panel interactions (`MapView`, panels, glosses).** Marker clicks,
   resize (`browser_resize`), gloss tooltips opening/dismissing, panel scroll.

## Pre-existing noise to expect (not findings)

Record these at baseline and set them aside: Vite dev warnings/HMR logs, SSE
connect/reconnect chatter, llama-server or inferno startup/latency warnings, and
generation errors that occur simply because no model is running locally
(distinguish "no model" from "the UI handled no-model badly").
