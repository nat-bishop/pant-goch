---
name: exploratory-bug-hunt
description: >-
  Methodically drive a running web app through the Playwright MCP browser to
  discover bugs and edge cases with no predefined scenario — an autonomous QA
  "breaker" session. Use whenever the user wants to find bugs by actually using
  the app (not by reading code or writing one known test): "try to break it",
  "poke around and see what's broken", "do an exploratory QA pass", "find edge
  cases", "fuzz the UI", "stress test the frontend", "see if clicking around
  throws errors", or "use Playwright to test the app generally". Also use when
  the user gestures at the app being flaky without naming a specific repro.
  Default is FIND AND REPORT only — produce a findings report, do not edit code
  unless the prompt explicitly asks to fix. Do NOT use this for writing one
  known-scenario Playwright spec, for unit/component tests, or for reproducing a
  single bug the user already described — those are targeted, not exploratory.
---

# Exploratory Bug Hunt

Drive a live web app through the Playwright MCP browser, deliberately try things
that tend to break apps, and watch for anything that goes wrong. There's no
script and usually no one telling you the "correct" behavior — so the central
problem is **noticing**. Random clicking finds shallow bugs and stops. The whole
point of this skill is to make you systematic instead.

The trick to finding bugs without a spec: you can't assert behavior you don't
know, so anchor on two things you *can* know — a **baseline** (what a fresh load
looks like) and a handful of **universal invariants** that hold for any correct
app. A bug is a deviation from one of those.

## Posture

- **Report only, by default.** Produce a findings report; do not edit code. This
  keeps the session safe to run unattended. Only fix if the prompt explicitly
  opts in ("and fix what you find") — see the last section.
- **Steerable.** If the prompt names a focus ("hammer the input box", "the map
  is flaky"), weight exploration there after a quick baseline sweep. If it's
  broad, cover the whole app — breadth before depth.

## The universal invariants (your oracles)

After interactions, these are how you spot a bug without knowing intent. A
violation is almost always a real finding:

- **No *new* console errors/warnings** vs the baseline.
- **No failed network requests** (4xx/5xx, hung, CORS).
- **No white screen, crash, freeze, or React error overlay.**
- **Round-trips invert** — undo after redo returns to the prior state; save then
  reload matches; create then delete returns to empty. This is the most
  productive oracle *where actions are reversible* — but some apps have no such
  operations (a streamed, non-deterministic app has nothing to invert). The
  project reference will tell you which oracles actually apply here.
- **No rendered garbage** — `undefined`, `NaN`, `[object Object]`,
  `Invalid Date`, raw `i18n.keys` visible in the UI.
- **Layout holds** — no overflow, off-screen elements, overlapping text, or a
  spinner that never resolves.
- **Controls do something** — a button with no state/network/console effect is
  suspicious.

Every finding must cite real evidence (a console line, a network entry, a
screenshot). Don't invent bugs to look busy. If you suspect a behavioral bug but
can't be sure without knowing intent, still log it — mark it `needs-judgment`.

## Loop

1. **Launch & baseline.** Get the app running and `browser_navigate` to it. In
   this repo (smallville / HANDLER), read `references/smallville.md` for the
   launch sequence, the surfaces, and — importantly — which oracles apply to an
   LLM-driven app. Take a `browser_snapshot` (the a11y tree — your map) and a
   screenshot, then record the **pre-existing** console warnings and network
   requests. Distinguishing "new" from "already there" is what makes the report
   trustworthy. Open a findings file (e.g. `bug-hunt-findings.md`) and write the
   baseline noise under "Pre-existing (not bugs)".

2. **Map.** From the snapshot, list the views, panels, and interactive elements,
   plus the app's 2–6 core *verbs* (the main things it lets you do). Write this
   as a checklist in the findings file so coverage is trackable and exploration
   is systematic rather than random.

3. **Provoke.** Do each verb's happy path first (so you know it works), then push
   the edges. Work through these categories, ticking off the map:
   - **Malformed input** — empty, whitespace, very long, unicode/emoji,
     negative/zero/huge numbers, HTML/script, stray spaces.
   - **Lifecycle & persistence** — refresh mid-action, navigate away and back,
     save→reload→compare, create→delete→empty, does an unsaved draft survive a
     reload (and should it?).
   - **Undo/redo round-trips** — do, undo, verify exact return; redo; chain
     several and unwind both ways. (Skip if the app has no such operations.)
   - **Concurrency & timing** — double-click submits, spam a button, fire two
     async ops at once, interact before loading finishes, cancel mid-flight.
   - **Interaction edges** — drag to an invalid target / drop in empty space, tab
     order + Enter/Escape, resize small and large (`browser_resize`).
   - **Failure handling** — block the network or stop the backend and see if the
     UI degrades gracefully or white-screens; feed invalid IDs into routes.

   You won't exhaust this every run — timebox each area, and prioritize the core
   verbs and whatever the prompt steered toward.

4. **Check, after every action.** This discipline is what separates QA from
   clicking: `browser_console_messages` and `browser_network_requests` for new
   errors, a fresh snapshot/screenshot for visual breakage, the relevant
   round-trip oracle, and `browser_evaluate` for invariants you can't see (scan
   the DOM for overflow or garbage text, or read app state if exposed). When
   something fires, capture evidence *immediately* — state is ephemeral — and
   note the exact steps from a known starting point so the repro replays.

5. **Log as you go.** Append one entry per distinct bug (dedupe — same root cause
   hit three ways is one finding). Write to the file continuously, not at the
   end: long sessions get summarized and in-context findings vanish. Per entry:
   severity (`critical` > `major` > `minor` > `polish`) and confidence
   (`definite` / `likely` / `needs-judgment`, independent of severity), area,
   numbered repro from a known state, expected vs actual, and evidence.

6. **Report & stop.** Finish with a summary table sorted by severity, a coverage
   note (what you exercised and what you didn't reach, so the user knows the
   blind spots), the full entries, and the pre-existing noise set aside. State
   the count plainly ("6 findings: 1 critical, 2 major, 3 minor"). In report-only
   mode, do not fix anything.

Keep one browser; reload to a clean state before starting a new area. If a flow
isn't yielding after a few varied attempts, log what you have and move on —
don't loop on the same failing action.

## If the prompt opts into fixing

Fix in severity order. Auto-fix the clear, low-risk ones (null guards, missing
`await`, off-by-one, a typo'd key, an unhandled empty state); **pause and ask**
before anything that changes intended behavior or touches shared logic. **Verify
each fix against its own repro** — re-run the exact Playwright steps and confirm
the bug is gone with no new console/network errors; a fix you didn't reproduce
then re-check is a guess. Report what you fixed, what you left, and why.
