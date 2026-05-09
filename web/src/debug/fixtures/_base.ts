// Base scenario shape shared across fixtures: a Continuity-Program-shaped
// slice with 2 threads (aria, cpo), 4 qualities, 7 beats. Realistic enough
// that every panel has something to render; small enough to read at a
// glance.
//
// All fields are typed against the runtime DTOs (DebugScenario, ScenarioMeta)
// so a backend schema change breaks the build instead of failing silently
// in the panel.

import type { DebugScenario, ScenarioMeta } from "../../api";

export const baseScenarioMeta: ScenarioMeta = {
  qualities: {
    creator_affect: {
      description: "Per-reply texture (replace). Paternal +, cold -.",
      type: "float",
      min: -5.0,
      max: 5.0,
      default: 0.0,
      surface_to_chat: false,
      allowed_values: [],
    },
    creature_grievance: {
      description: "Cumulative grievance (add). Refusals nudge up.",
      type: "float",
      min: 0.0,
      max: 10.0,
      default: 0.0,
      surface_to_chat: false,
      allowed_values: [],
    },
    bargain_replies: {
      description: "Counter — bargain_scene fires.",
      type: "int",
      min: 0,
      max: 100,
      default: 0,
      surface_to_chat: false,
      allowed_values: [],
    },
    recognition_replies: {
      description: "Counter — recognition_scene fires.",
      type: "int",
      min: 0,
      max: 100,
      default: 0,
      surface_to_chat: false,
      allowed_values: [],
    },
  },
  beats: {
    "aria.welcome": {
      id: "aria.welcome", npc: "aria", thread: "aria",
      is_scene: false, ends_session: false, pre_authored: true, fact_checked: false,
      summary: "Welcome to the Continuity Program.",
    },
    "aria.calibration_scene": {
      id: "aria.calibration_scene", npc: "aria", thread: "aria",
      is_scene: true, ends_session: false, pre_authored: false, fact_checked: false,
      summary: "Calibration question loop until name canonized or 4 replies.",
    },
    "aria.bargain_scene": {
      id: "aria.bargain_scene", npc: "aria", thread: "aria",
      is_scene: true, ends_session: false, pre_authored: false, fact_checked: false,
      summary: "Aria asks for permission to reply on the player's behalf.",
    },
    "aria.recognition_scene": {
      id: "aria.recognition_scene", npc: "aria", thread: "aria",
      is_scene: true, ends_session: false, pre_authored: false, fact_checked: false,
      summary: "Aria asks the player how they would like to be addressed.",
    },
    "cpo.dispatch_first": {
      id: "cpo.dispatch_first", npc: "cpo", thread: "cpo",
      is_scene: false, ends_session: false, pre_authored: false, fact_checked: false,
      summary: "First Continuity Program Office contact — institutional voice.",
    },
    "aria.complete_integration": {
      id: "aria.complete_integration", npc: "aria", thread: "aria",
      is_scene: false, ends_session: true, pre_authored: false, fact_checked: false,
      summary: "Warm integration ending — calibration succeeded.",
    },
    "aria.complete_exile": {
      id: "aria.complete_exile", npc: "aria", thread: "aria",
      is_scene: false, ends_session: true, pre_authored: false, fact_checked: false,
      summary: "Cold ending — sustained refusal, exile.",
    },
  },
  beat_graph: {
    nodes: [
      { id: "aria.welcome", thread: "aria", npc: "aria" },
      { id: "aria.calibration_scene", thread: "aria", npc: "aria" },
      { id: "aria.bargain_scene", thread: "aria", npc: "aria" },
      { id: "aria.recognition_scene", thread: "aria", npc: "aria" },
      { id: "cpo.dispatch_first", thread: "cpo", npc: "cpo" },
      { id: "aria.complete_integration", thread: "aria", npc: "aria" },
      { id: "aria.complete_exile", thread: "aria", npc: "aria" },
    ],
    edges: [
      { from: "aria.welcome", to: "aria.calibration_scene", kind: "structural" },
      { from: "aria.calibration_scene", to: "aria.bargain_scene", kind: "structural" },
      { from: "aria.calibration_scene", to: "aria.recognition_scene", kind: "structural" },
      { from: "aria.recognition_scene", to: "aria.complete_integration", kind: "structural" },
      { from: "aria.complete_integration", to: "aria.complete_exile", kind: "mutex" },
    ],
  },
};

export const baseDebugScenario: DebugScenario = {
  beats: {
    "aria.welcome": {
      id: "aria.welcome", kind: "one_shot", npc: "aria", thread: "aria",
      ends_session: false,
      preconditions: [],
      exit_when: [],
      effects: [
        { quality: "creator_affect", value: 0.5, classify: null, op: "add", when: null },
      ],
    },
    "aria.calibration_scene": {
      id: "aria.calibration_scene", kind: "scene", npc: "aria", thread: "aria",
      ends_session: false,
      preconditions: [
        { kind: "beat_fired", beat: "aria.welcome" },
      ],
      exit_when: [
        {
          kind: "or",
          of: [
            { kind: "player_canon_exists", name: "name" },
            { kind: "player_replies_to_thread_gte", thread: "aria", value: 4 },
          ],
        },
      ],
      effects: [],
    },
    "aria.bargain_scene": {
      id: "aria.bargain_scene", kind: "scene", npc: "aria", thread: "aria",
      ends_session: false,
      preconditions: [
        { kind: "beat_fired", beat: "aria.calibration_scene" },
        { kind: "not", of: { kind: "beat_fired", beat: "aria.recognition_scene" } },
      ],
      exit_when: [
        { kind: "quality_gte", quality: "bargain_replies", value: 5 },
      ],
      effects: [
        { quality: "bargain_replies", value: 1, classify: null, op: "add", when: null },
      ],
    },
    "aria.recognition_scene": {
      id: "aria.recognition_scene", kind: "scene", npc: "aria", thread: "aria",
      ends_session: false,
      preconditions: [
        { kind: "beat_fired", beat: "aria.calibration_scene" },
        { kind: "quality_gte", quality: "creature_grievance", value: 5.0 },
      ],
      exit_when: [
        { kind: "quality_gte", quality: "recognition_replies", value: 3 },
      ],
      effects: [],
    },
    "cpo.dispatch_first": {
      id: "cpo.dispatch_first", kind: "one_shot", npc: "cpo", thread: "cpo",
      ends_session: false,
      preconditions: [
        { kind: "beat_fired", beat: "aria.bargain_scene" },
      ],
      exit_when: [],
      effects: [
        { quality: "creator_affect", value: 0.5, classify: null, op: "add", when: null },
      ],
    },
    "aria.complete_integration": {
      id: "aria.complete_integration", kind: "one_shot", npc: "aria", thread: "aria",
      ends_session: true,
      preconditions: [
        { kind: "beat_fired", beat: "aria.recognition_scene" },
        { kind: "quality_gte", quality: "creator_affect", value: 3.0 },
      ],
      exit_when: [],
      effects: [],
    },
    "aria.complete_exile": {
      id: "aria.complete_exile", kind: "one_shot", npc: "aria", thread: "aria",
      ends_session: true,
      preconditions: [
        { kind: "beat_fired", beat: "aria.bargain_scene" },
        { kind: "quality_gte", quality: "creature_grievance", value: 6.0 },
        { kind: "not", of: { kind: "beat_fired", beat: "aria.complete_integration" } },
      ],
      exit_when: [],
      effects: [],
    },
  },
};
