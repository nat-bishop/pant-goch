// One tick where the extractor proposed two candidates and both were
// rejected: one for anchor_fail (quote not literal), one for
// below_salience (extractor confidence too low). Verifies the Canon
// dropped-candidates section renders both reason variants.

import type { TickDecision } from "../../api";

export const droppedCanonTicks: TickDecision[] = [
  {
    in_response_to: "msg_drop",
    eligible: [{
      beat_id: "aria.calibration_scene", status: "eligible",
      blocked_by: null, is_scene: true, predicate_tree: null, predicate_trace: null,
    }],
    selection: "continue",
    chosen_beat_id: "aria.calibration_scene",
    chosen_rationale: null,
    quality_deltas: [],
    fired_beat_id: "aria.calibration_scene",
    scene_turn: 2,
    canon_writes: [],
    canon_drops: [
      {
        candidate: {
          key: "relationship",
          summary: "married",
          literal_text_quote: "I have a partner",
          salience: 0.31,
          category: "person",
        },
        reason: "below_salience",
      },
      {
        candidate: {
          key: "hometown",
          summary: "Detroit",
          literal_text_quote: "I'm from up north",
          salience: 0.62,
          category: "place",
        },
        reason: "anchor_fail",
      },
    ],
    player_canon: {
      name: {
        summary: "Marcus Chen",
        literal_text_quote: "I'm Marcus Chen, the new associate",
        salience: 0.92,
        category: "name",
        turn_of_origin: 2,
      },
    },
    fact_check: null,
    classifier_history: { creator_affect: ["earnest"] },
    fired_beat_ids: ["aria.welcome"],
    beats_evaluation: [
      { beat_id: "aria.welcome", status: "fired", is_scene: false, entry_traces: [], exit_traces: [] },
      {
        beat_id: "aria.calibration_scene", status: "eligible", is_scene: true,
        entry_traces: [{ passed: true, label: "beat_fired(aria.welcome)", children: [] }],
        exit_traces: [{ passed: false, label: "or(...)", children: [] }],
      },
      {
        beat_id: "aria.bargain_scene", status: "blocked", is_scene: true,
        entry_traces: [
          { passed: false, label: "beat_fired(aria.calibration_scene) — not yet", children: [] },
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
          { passed: true, label: "not(beat_fired(aria.complete_integration))", children: [] },
        ],
        exit_traces: [],
      },
    ],
  },
];
