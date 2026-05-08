// Every terminal beat unfired, predicate trees show concrete failing
// atoms with shortBy-like notes. Stress-test for the Beats inspector
// renderer.

import type { TickDecision } from "../../api";

export const allBlockedTicks: TickDecision[] = [
  {
    in_response_to: "msg_initial",
    eligible: [],
    selection: "none",
    chosen_beat_id: null,
    chosen_rationale: null,
    quality_deltas: [],
    fired_beat_id: null,
    scene_turn: null,
    canon_writes: [],
    canon_drops: [],
    player_canon: {},
    fact_check: null,
    classifier_history: {},
    fired_beat_ids: ["aria.welcome"],
    beats_evaluation: [
      {
        beat_id: "aria.welcome", status: "fired", is_scene: false,
        entry_traces: [], exit_traces: [],
      },
      {
        beat_id: "aria.calibration_scene", status: "blocked", is_scene: true,
        entry_traces: [{ passed: true, label: "beat_fired(aria.welcome)", children: [] }],
        exit_traces: [{
          passed: false,
          label: "or(player_canon(name) — not yet | replies(aria)>=4 got 1)",
          children: [
            { passed: false, label: "player_canon(name) — not yet", children: [] },
            { passed: false, label: "replies(aria)>=4 got 1", children: [] },
          ],
        }],
      },
      {
        beat_id: "aria.bargain_scene", status: "blocked", is_scene: true,
        entry_traces: [
          { passed: false, label: "beat_fired(aria.calibration_scene) — not yet", children: [] },
          { passed: true, label: "not(beat_fired(aria.recognition_scene))", children: [
            { passed: false, label: "beat_fired(aria.recognition_scene) — not yet", children: [] },
          ]},
        ],
        exit_traces: [{ passed: false, label: "bargain_replies>=5 got 0", children: [] }],
      },
      {
        beat_id: "aria.recognition_scene", status: "blocked", is_scene: true,
        entry_traces: [
          { passed: false, label: "beat_fired(aria.calibration_scene) — not yet", children: [] },
          { passed: false, label: "creature_grievance>=5 got 0", children: [] },
        ],
        exit_traces: [{ passed: false, label: "recognition_replies>=3 got 0", children: [] }],
      },
      {
        beat_id: "cpo.dispatch_first", status: "blocked", is_scene: false,
        entry_traces: [{ passed: false, label: "beat_fired(aria.bargain_scene) — not yet", children: [] }],
        exit_traces: [],
      },
      {
        beat_id: "aria.complete_integration", status: "blocked", is_scene: false,
        entry_traces: [
          { passed: false, label: "beat_fired(aria.recognition_scene) — not yet", children: [] },
          { passed: false, label: "creator_affect>=3 got 0", children: [] },
        ],
        exit_traces: [],
      },
      {
        beat_id: "aria.complete_exile", status: "blocked", is_scene: false,
        entry_traces: [
          { passed: false, label: "beat_fired(aria.bargain_scene) — not yet", children: [] },
          { passed: false, label: "creature_grievance>=6 got 0", children: [] },
          { passed: true, label: "not(beat_fired(aria.complete_integration))", children: [
            { passed: false, label: "beat_fired(aria.complete_integration) — not yet", children: [] },
          ]},
        ],
        exit_traces: [],
      },
    ],
  },
];
