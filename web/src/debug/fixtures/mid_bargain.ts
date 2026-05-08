// The "demo" fixture. Mid-game state matching the mockup:
//   - calibration retired (canon[name] set)
//   - bargain scene firing (2 fires so far)
//   - cpo.dispatch_first fired
//   - canon has name + motivation
//   - several classifier verdicts stacked
//
// tickHistory is newest-first (matches the runtime store shape).

import type { TickDecision } from "../../api";

const beatsT8: TickDecision["beats_evaluation"] = [
  { beat_id: "aria.welcome", status: "fired", is_scene: false, entry_traces: [], exit_traces: [] },
  {
    beat_id: "aria.calibration_scene", status: "fired", is_scene: true,
    entry_traces: [{ passed: true, label: "beat_fired(aria.welcome)", children: [] }],
    exit_traces: [{ passed: true, label: "or(player_canon(name)=Marcus Chen | replies(aria)>=4 got 5)", children: [
      { passed: true, label: "player_canon(name)=Marcus Chen", children: [] },
      { passed: true, label: "replies(aria)>=4 got 5", children: [] },
    ]}],
  },
  {
    beat_id: "aria.bargain_scene", status: "eligible", is_scene: true,
    entry_traces: [
      { passed: true, label: "beat_fired(aria.calibration_scene)", children: [] },
      { passed: true, label: "not(beat_fired(aria.recognition_scene))", children: [
        { passed: false, label: "beat_fired(aria.recognition_scene) — not yet", children: [] },
      ]},
    ],
    exit_traces: [{ passed: false, label: "bargain_replies>=5 got 2", children: [] }],
  },
  {
    beat_id: "aria.recognition_scene", status: "blocked", is_scene: true,
    entry_traces: [
      { passed: true, label: "beat_fired(aria.calibration_scene)", children: [] },
      { passed: false, label: "creature_grievance>=5 got 4", children: [] },
    ],
    exit_traces: [{ passed: false, label: "recognition_replies>=3 got 0", children: [] }],
  },
  {
    beat_id: "cpo.dispatch_first", status: "fired", is_scene: false,
    entry_traces: [{ passed: true, label: "beat_fired(aria.bargain_scene)", children: [] }],
    exit_traces: [],
  },
  {
    beat_id: "aria.complete_integration", status: "blocked", is_scene: false,
    entry_traces: [
      { passed: false, label: "beat_fired(aria.recognition_scene) — not yet", children: [] },
      { passed: false, label: "creator_affect>=3 got 1.2", children: [] },
    ],
    exit_traces: [],
  },
  {
    beat_id: "aria.complete_exile", status: "blocked", is_scene: false,
    entry_traces: [
      { passed: true, label: "beat_fired(aria.bargain_scene)", children: [] },
      { passed: false, label: "creature_grievance>=6 got 4", children: [] },
      { passed: true, label: "not(beat_fired(aria.complete_integration))", children: [
        { passed: false, label: "beat_fired(aria.complete_integration) — not yet", children: [] },
      ]},
    ],
    exit_traces: [],
  },
];

const playerCanonT8: TickDecision["player_canon"] = {
  name: {
    summary: "Marcus Chen",
    literal_text_quote: "I'm Marcus Chen, the new associate",
    salience: 0.92,
    category: "name",
    turn_of_origin: 2,
  },
  motivation: {
    summary: "needs to prove himself",
    literal_text_quote: "I came here to prove I deserve a seat at this table",
    salience: 0.78,
    category: "event",
    turn_of_origin: 4,
  },
};

export const midBargainTicks: TickDecision[] = [
  // T8 — bargain_scene fire #2 (newest)
  {
    in_response_to: "msg_t8",
    eligible: [{ beat_id: "aria.bargain_scene", status: "eligible", blocked_by: null, is_scene: true, predicate_tree: null, predicate_trace: null }],
    selection: "continue",
    chosen_beat_id: "aria.bargain_scene",
    chosen_rationale: null,
    quality_deltas: [
      { quality: "creature_grievance", before: 1.5, after: 4.0, source: "effect:aria.bargain_scene:creature_grievance" },
      { quality: "bargain_replies", before: 1, after: 2, source: "effect:aria.bargain_scene:bargain_replies" },
    ],
    fired_beat_id: "aria.bargain_scene",
    scene_turn: 2,
    canon_writes: [],
    canon_drops: [],
    player_canon: playerCanonT8,
    fact_check: null,
    classifier_history: { bargain: ["refused", "engaged"], creator_affect: ["heated", "concerned", "clinical"] },
    fired_beat_ids: ["aria.welcome", "aria.calibration_scene", "cpo.dispatch_first"],
    beats_evaluation: beatsT8,
  },
  // T7 — bargain_scene fire #1
  {
    in_response_to: "msg_t7",
    eligible: [{ beat_id: "aria.bargain_scene", status: "eligible", blocked_by: null, is_scene: true, predicate_tree: null, predicate_trace: null }],
    selection: "sole",
    chosen_beat_id: "aria.bargain_scene",
    chosen_rationale: null,
    quality_deltas: [
      { quality: "bargain_replies", before: 0, after: 1, source: "effect:aria.bargain_scene:bargain_replies" },
    ],
    fired_beat_id: "aria.bargain_scene",
    scene_turn: 1,
    canon_writes: [],
    canon_drops: [],
    player_canon: playerCanonT8,
    fact_check: null,
    classifier_history: { bargain: ["engaged"], creator_affect: ["concerned", "clinical"] },
    fired_beat_ids: ["aria.welcome", "aria.calibration_scene", "cpo.dispatch_first"],
    beats_evaluation: beatsT8,
  },
  // T6 — cpo dispatch
  {
    in_response_to: "msg_t6",
    eligible: [
      { beat_id: "aria.bargain_scene", status: "eligible", blocked_by: null, is_scene: true, predicate_tree: null, predicate_trace: null },
      { beat_id: "cpo.dispatch_first", status: "eligible", blocked_by: null, is_scene: false, predicate_tree: null, predicate_trace: null },
    ],
    selection: "director",
    chosen_beat_id: "cpo.dispatch_first",
    chosen_rationale:
      "introduce institutional voice mid-bargain stall — player needs friction-from-elsewhere before re-engaging Aria",
    quality_deltas: [
      { quality: "creator_affect", before: 0.5, after: 1.0, source: "effect:cpo.dispatch_first:creator_affect" },
    ],
    fired_beat_id: "cpo.dispatch_first",
    scene_turn: null,
    canon_writes: [],
    canon_drops: [],
    player_canon: playerCanonT8,
    fact_check: null,
    classifier_history: { creator_affect: ["concerned"] },
    fired_beat_ids: ["aria.welcome", "aria.calibration_scene"],
    beats_evaluation: beatsT8,
  },
  // T4 — calibration_scene fire #3 (canon name extracted)
  {
    in_response_to: "msg_t4",
    eligible: [{ beat_id: "aria.calibration_scene", status: "eligible", blocked_by: null, is_scene: true, predicate_tree: null, predicate_trace: null }],
    selection: "continue",
    chosen_beat_id: "aria.calibration_scene",
    chosen_rationale: null,
    quality_deltas: [
      { quality: "creator_affect", before: 1.0, after: 0.4, source: "classify:creator_affect" },
    ],
    fired_beat_id: "aria.calibration_scene",
    scene_turn: 3,
    canon_writes: [{
      key: "motivation",
      summary: "needs to prove himself",
      literal_text_quote: "I came here to prove I deserve a seat at this table",
      salience: 0.78,
      category: "event",
    }],
    canon_drops: [],
    player_canon: playerCanonT8,
    fact_check: null,
    classifier_history: { creator_affect: ["earnest"] },
    fired_beat_ids: ["aria.welcome"],
    beats_evaluation: beatsT8,
  },
  // T2 — calibration_scene fire #1, name canonized
  {
    in_response_to: "msg_t2",
    eligible: [{ beat_id: "aria.calibration_scene", status: "eligible", blocked_by: null, is_scene: true, predicate_tree: null, predicate_trace: null }],
    selection: "sole",
    chosen_beat_id: "aria.calibration_scene",
    chosen_rationale: null,
    quality_deltas: [
      { quality: "creator_affect", before: 0.5, after: 1.0, source: "classify:creator_affect" },
    ],
    fired_beat_id: "aria.calibration_scene",
    scene_turn: 1,
    canon_writes: [{
      key: "name",
      summary: "Marcus Chen",
      literal_text_quote: "I'm Marcus Chen, the new associate",
      salience: 0.92,
      category: "name",
    }],
    canon_drops: [],
    player_canon: { name: playerCanonT8.name! },
    fact_check: null,
    classifier_history: { creator_affect: ["warm"] },
    fired_beat_ids: ["aria.welcome"],
    beats_evaluation: beatsT8,
  },
  // T1 — boot welcome
  {
    in_response_to: null,
    eligible: [{ beat_id: "aria.welcome", status: "eligible", blocked_by: null, is_scene: false, predicate_tree: null, predicate_trace: null }],
    selection: "sole",
    chosen_beat_id: "aria.welcome",
    chosen_rationale: null,
    quality_deltas: [
      { quality: "creator_affect", before: 0.0, after: 0.5, source: "effect:aria.welcome:creator_affect" },
    ],
    fired_beat_id: "aria.welcome",
    scene_turn: null,
    canon_writes: [],
    canon_drops: [],
    player_canon: {},
    fact_check: null,
    classifier_history: {},
    fired_beat_ids: ["aria.welcome"],
    beats_evaluation: beatsT8,
  },
];
