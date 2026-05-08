// Hover-label primitives: quality and beat names. Both render the bare
// identifier when no metadata is loaded, otherwise wrap in a tooltip with
// type/range/description (qualities) or npc/thread/flags/summary (beats).

import type { BeatMeta, ScenarioMeta } from "../../api";
import { DebugTooltip } from "./DebugTooltip";

export function QualityName({
  quality, scenarioMeta,
}: { quality: string; scenarioMeta: ScenarioMeta }) {
  const meta = scenarioMeta.qualities[quality];
  if (!meta || !meta.description) {
    return <span className="text-text">{quality}</span>;
  }
  return (
    <DebugTooltip
      wide
      content={
        <div className="space-y-0.5">
          <div className="font-mono text-text">{quality}</div>
          <div className="text-text-muted">type: {meta.type}{meta.min !== null && `, ${meta.min}–${meta.max}`}</div>
          {meta.surface_to_chat && (
            <div className="text-text-muted">surface_to_chat: true</div>
          )}
          <div className="mt-1 whitespace-pre-wrap text-text">{meta.description.trim()}</div>
        </div>
      }
    >
      <span className="text-text underline decoration-dotted decoration-text-muted/40 underline-offset-2">{quality}</span>
    </DebugTooltip>
  );
}

export function BeatName({
  beatId, scenarioMeta,
}: { beatId: string; scenarioMeta: ScenarioMeta }) {
  const meta = scenarioMeta.beats[beatId];
  if (!meta) {
    return <span className="text-text">{beatId}</span>;
  }
  return (
    <DebugTooltip
      wide
      content={<BeatTooltipBody meta={meta} />}
    >
      <span className="text-text underline decoration-dotted decoration-text-muted/40 underline-offset-2">{beatId}</span>
    </DebugTooltip>
  );
}

function BeatTooltipBody({ meta }: { meta: BeatMeta }) {
  return (
    <div className="space-y-0.5">
      <div className="font-mono text-text">{meta.id}</div>
      <div className="text-text-muted">npc: <span className="text-text">{meta.npc}</span></div>
      <div className="text-text-muted">thread: <span className="text-text">{meta.thread}</span></div>
      <div className="flex flex-wrap gap-1 text-text-muted">
        {meta.is_scene && <span>scene</span>}
        {meta.ends_session && <span className="text-meter-hot">terminal</span>}
        {meta.pre_authored && <span>pre-authored</span>}
        {meta.validate_facts && <span>fact-checked</span>}
      </div>
      {meta.summary && <div className="mt-1 whitespace-pre-wrap text-text">{meta.summary}</div>}
    </div>
  );
}
