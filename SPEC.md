# SPEC — Debug Menu Redesign

> **Status:** draft (v0 scope) · **Owner:** Nat · **Date:** 2026-05-07
> **Scope:** Replace the current DEV-only `DebugPanel` (which renders raw `tick_decision` SSE events) with a structured, multi-panel debug surface that supports pacing inspection, beat-predicate forensics, and player-canon visibility. Replay and inline agent-trace are explicitly deferred — Logfire covers historical agent activity, and rewind hasn't proven necessary yet.

---

## Terminology

- **Tick** — one pass through `_tick()` (`main.py:847`). Triggered by a player `/send` or once at boot for the welcome. **Not time-based** — there is no background scheduler. Each tick fires at most one beat. Pacing is action-counter / quality driven.
- **Thread** — an email thread keyed to one NPC (e.g., `aria`, `cpo`). The player can reply on any thread; one tick can fire a beat on a *different* thread than the one the player replied on (cross-thread first-fires; see `main.py:911-914`).
- **Beat fire** — the engine renders a beat, emitting one or more emails. A beat carries an `npc:` field, so the sender is determined by the beat, not by the trigger thread.

---

## 1. Goals

1. **Make pacing legible.** "Why is `creature_grievance` at 4?" and "what's stalling `recognition_scene`?" should be answerable in seconds, not by tailing logs.
2. **Make beat eligibility explainable.** "Why didn't `recognition_scene` fire?" should resolve to a specific failed predicate atom, not a guess.
3. **Make player canon visible.** What's in `state.player_canon`, what got dropped, and which extractor candidates failed the literal-quote anchor check.
4. **Stay read-only.** Inspection is the contract — no force-fire, no quality edits, no canon mutations. Authoring tooling lives elsewhere.

### Non-goals

- Mutating live state (force-fire beats, edit qualities, set canon entries).
- **Session replay / rewind.** Defer until concrete usage shows it's needed.
- **Inline agent trace UI.** Logfire is already wired up; building a parallel renderer for the same data is duplication.
- Cross-session comparison or A/B-style diffs.
- Production-user visibility — this is DEV-only and ships behind `import.meta.env.DEV`.
- Persistence across server restarts.

---

## 2. Audience

A single power user (the dev who is also the author and the playtester). The UI is **information-dense by default** with progressive disclosure for raw payloads (JSON drawers, expandable tick rows). No role switcher; no curated subset views.

---

## 3. Architecture overview

```
┌────────────────────────── browser (DEV builds only) ──────────────────────────┐
│                                                                                │
│   ┌─ Pacing (primary) ─┐  ┌─ Secondary tabs ────────────────────────────┐      │
│   │  Sparkline grid    │  │ Tick log │ Canon │ Beats                    │      │
│   │  → click → drill   │  └──────────────────────────────────────────────┘      │
│   └────────────────────┘                                                        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────── FastAPI ──────────────────────────────────────────┐
│                                                                                │
│   GameState + TickDecision (existing)                                          │
│   - tick_decision SSE stream (existing)                                        │
│   - GET /debug/scenario (new) — full beat/predicate definitions                │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

The panel is a pure **derived view** over data the engine already produces — `TickDecision` SSE events for tick history, `state.player_canon` from `/state`, and a new (read-only) endpoint that ships the scenario beat/predicate definitions so the beat inspector can render predicate trees.

---

## 4. Primary view — Pacing

The default view when the debug menu opens.

### 4.1 Overview: sparkline grid

A compact grid, one row per quality. Each row shows:

| Element       | Behavior                                                                 |
| ------------- | ------------------------------------------------------------------------ |
| Quality name  | Plain text, left-aligned.                                                |
| Sparkline     | SVG polyline; x = tick index, y = quality value normalized to row range. |
| Current value | Right-aligned numeric.                                                   |
| Last delta    | `+1 ↑`, `-2 ↓`, `±0` — colored (green/red/neutral).                      |
| Click target  | Whole row. Drills into §4.2 for that quality.                            |

Hovering anywhere on the sparkline reveals a tooltip with `(tick, value, cause)` for that x position.

> **Why sparklines as the default:** they answer "shape of the session" in one viewport. Tables and timelines are better for forensic reading but slower for first impressions.

### 4.2 Drill-down: cause attribution

When a sparkline row is clicked, the quality's full history fills the panel:

```
creature_grievance                                    current: 4.0
─────────────────────────────────────────────────────────────
T3   +0.5   classify: cold        classifier → creature_grievance
T4   +0.5   classify: cold        classifier → creature_grievance
T5   +0.5   classify: cold        classifier → creature_grievance
T7   +2.5   bargain_scene         literal, when: classify=refused (floor 4.0)
─────────────────────────────────────────────────────────────
BLOCKED ON THIS QUALITY
  aria.recognition_scene    needs ≥ 5.0   (1.0 short)
  aria.bargain_betrayed     needs ≥ 6.0   (2.0 short)
```

The **blocked-on** section is computed by scanning every unfired beat's `entry_when` for atoms that reference this quality and would currently fail. It answers "what is gated on this number?" without needing to grep the YAML.

---

## 5. Secondary panels

### 5.1 Tick log

Chronological list of every `TickDecision` in the session, newest first. Each row collapsed by default; expand to reveal the full JSON:

- `eligibility_report`: list of beats considered, with per-beat fired-status and predicate verdict.
- `selection_method`: `none` | `sole` | `continue` | `director`.
- `director_rationale`: present only when method = `director`.
- `effects_applied`: literal effects + classify writes + extractor promotions, with before/after values.
- `fired`: beat id + render destination (chat-stream | epilogue) + the beat's `thread` (which may differ from the trigger thread — cross-thread first-fires are common).
- `trigger`: thread the player replied on (or `"boot"` for the boot tick) — surfaces when a CPO beat fires off an Aria reply, etc.

Replaces the current raw `tick_decision` SSE rendering. Engine-dev essential.

### 5.2 Player canon ledger

Live snapshot of `state.player_canon`:

| Column         | Meaning                                                              |
| -------------- | -------------------------------------------------------------------- |
| `key`          | Canon key (e.g., `name`, `motivation`).                              |
| `value`        | Current value.                                                       |
| `quote`        | The verbatim player-text substring that anchored this entry.         |
| `salience`     | 0.0–1.0 score from extractor.                                        |
| `origin_tick`  | Tick when the entry was set or last replaced.                        |
| `decay_age`    | `turns_since_origin` — used to compute the chat agent's top-K view.  |

A second section shows **dropped candidates from the last N=20 player turns** — extractor outputs that failed the literal-quote anchor check. Useful for debugging "why did Aria not pick up that I told her my name?" cases.

### 5.3 Beat inspector — "why didn't X fire?"

Pick any beat (search/dropdown). The panel renders:

1. **Header**: beat id, `kind` (`one_shot` | `scene`), `npc`, `thread`, `ends_session`.
2. **Fire status**: in `fired_beat_ids`? if scene, how many fires so far?
3. **`entry_when` evaluation** — atom by atom, against current state:
   - ✅ green for atoms that pass.
   - ❌ red for atoms that fail, with the current value beside the required value.
   - For nested `And_`/`Or_`/`Not_`, indent and render the same way.
4. **`exit_when` evaluation** (scenes only) — same atom-by-atom rendering.
5. **Effects preview** — what would mutate if this beat fired right now.

Example:

```
aria.recognition_scene  (kind: scene, npc: aria, thread: aria)
  fired_beat_ids: false (0 fires)
  entry_when:
    And_:
      ✅ *calibration_done
      ❌ QualityAtLeast(creature_grievance, 5.0)   cur=4.0   (1.0 short)
      ✅ Not_(BeatFired(aria.recognition_scene))
```

---

## 6. Frontend layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Inbox (chat surface, unchanged)                                    │
│                                                                     │
│  ┌─ Debug menu (DEV-only, collapsible) ──────────────────────────┐   │
│  │  [Pacing]  Tick log │ Canon │ Beats                           │   │
│  │  ─────────────────────────────────────────────────────────────│   │
│  │  Sparkline grid                                                │   │
│  │  creator_affect      ▁▂▃▅▇█▇   1.2   +0.2↑                      │   │
│  │  creature_grievance  ▂▂▃▃▄▅▆   4.0   +2.5↑                      │   │
│  │  ...                                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

- Tabs at top of debug menu select among **Pacing** (default) / **Tick log** / **Canon** / **Beats**.
- Selected tab persists in `localStorage` so it survives reloads during dev iteration.

---

## 7. Acceptance criteria

- [ ] Sparkline grid renders one row per quality and updates live as ticks fire.
- [ ] Clicking a sparkline row shows §4.2 cause attribution + blocked-on list.
- [ ] Tick log shows every `TickDecision` with expandable JSON.
- [ ] Canon ledger lists current entries plus the last 20 dropped candidates.
- [ ] Beat inspector resolves any beat to a per-atom predicate verdict (entry_when + exit_when).
- [ ] None of the new code ships in production builds (`!import.meta.env.DEV` short-circuits the panel).

---

## 8. Open questions

1. **Tooltip cost.** Computing per-tick cause for every sparkline pixel may be expensive. If perf bites, downsample to "one cause label per tick band" rather than per pixel.
2. **Beat inspector for `Or_` predicates.** Atom-by-atom rendering for `Or_` is ambiguous — do we show every branch's status, or just the first satisfying branch? Default to *every branch*, since the debug intent is comprehensive.
3. **Beat data delivery.** Ship the full scenario shape to the frontend (one `GET /debug/scenario` call on boot) vs. a per-beat eval endpoint. Leaning toward shipping the whole scenario — it's <100KB and the frontend can re-evaluate atoms locally as `/state` updates.

---

## 9. Out-of-scope follow-ups

These came up during design but are explicitly deferred:

- **Replay / rewind.** Heavy state coupling; revisit when there's concrete usage signal that "I want to undo" is a frequent need.
- **Inline agent trace.** Logfire already serves this; building a duplicate UI is busywork.
- **Authoring mutations.** Force-fire, quality editor, canon editor — belong in a separate authoring tool.
- **Session export/import.** Useful for bug reports; revisit once v0 is in steady use.
- **Branching timelines.** Big scope jump; not justified by current pain.
- **Production-user debug surface.** Different audience, different vocabulary.
