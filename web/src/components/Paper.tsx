import { useMemo, useState } from "react";
import type { DeltaDescriptors } from "../api";
import { nowZulu } from "../lib/time";
import { useStore } from "../store";
import { CaveatBanner } from "./CaveatBanner";
import { OptionsList } from "./OptionsList";
import { StreamingText } from "./StreamingText";

type AuthPhase = "wait" | "ready" | "authorized" | "continue";

const BUTTON_LABEL: Record<AuthPhase, string> = {
  wait: "Authorize ▸",
  ready: "Authorize ▸",
  authorized: "Authorized ✓",
  continue: "Continue ▸",
};

// Stable empty fallback so the Zustand selector doesn't allocate a fresh
// reference on every read (same pattern as the old DispatchOverlay).
const EMPTY_DESCRIPTORS: DeltaDescriptors = Object.freeze({}) as DeltaDescriptors;

interface ResultLine {
  key: string;
  text: string;
}

/** Build the RESULT footer lines from the cycle's accumulated narrative
 * descriptors. Cover changes render as `LYNX: cover intact → cover degraded`
 * (using `agent_id` from the server payload); mission qualities render the
 * new descriptor as a standalone clause. */
function buildResultLines(descriptors: DeltaDescriptors): ResultLine[] {
  const lines: ResultLine[] = [];
  for (const [q, d] of Object.entries(descriptors)) {
    if (d.after === null) continue;
    if (d.agent_id !== null) {
      lines.push({
        key: q,
        text: d.before
          ? `${d.agent_id}: ${d.before} → ${d.after}`
          : `${d.agent_id}: ${d.after}`,
      });
    } else {
      lines.push({ key: q, text: d.after });
    }
  }
  // Cover lines first (most player-visible), then mission-scoped lines.
  return lines.sort(
    (a, b) =>
      (descriptors[a.key].agent_id ? 0 : 1) -
      (descriptors[b.key].agent_id ? 0 : 1),
  );
}

export function Paper() {
  const meta = useStore((s) => s.mission_meta);
  const cycle = useStore((s) => s.cycle);
  const authorize = useStore((s) => s.authorize);
  const continueCycle = useStore((s) => s.continueCycle);
  const [stampTimestamp, setStampTimestamp] = useState<string>("");

  if (!cycle || !meta) {
    return (
      <article className="paper-sheet">
        <CaveatBanner />
        <div className="paper-waiting">Waiting for cycle…</div>
      </article>
    );
  }

  const file =
    meta.agents[0]?.file_number ? `PF.${meta.agents[0].file_number}` : "PF.??-?";
  const headerTimestamp = nowZulu();

  const canAuth =
    cycle.awaiting_auth &&
    cycle.selected_option_index !== null &&
    !cycle.authorized;
  const phase: AuthPhase = cycle.awaiting_continue
    ? "continue"
    : cycle.authorized
      ? "authorized"
      : canAuth
        ? "ready"
        : "wait";
  const onClick =
    phase === "ready"
      ? () => {
          setStampTimestamp(nowZulu());
          void authorize();
        }
      : phase === "continue"
        ? () => void continueCycle()
        : undefined;

  const showFootHint =
    cycle.entry_archetype !== null && !cycle.entry_complete;

  return (
    <article
      className="paper-sheet"
      aria-label={`cycle ${cycle.index} brief`}
      data-has-selection={cycle.selected_option_index !== null}
      data-authorized={cycle.authorized}
    >
      <CaveatBanner />

      <div className="paper-doctype">
        <span>BRIEF</span>
        <span className="paper-doctype-meta">
          Op. {meta.codename} · {headerTimestamp}
        </span>
      </div>
      <div className="paper-rule64" />

      <div className="status-divider" />

      {/* Pre-authorize: options block. Post-authorize: log entry. */}
      {!cycle.authorized ? (
        <OptionsBlock
          stampTimestamp={stampTimestamp}
          onClick={onClick}
          buttonLabel={BUTTON_LABEL[phase]}
          authorized={cycle.authorized}
        />
      ) : (
        <EntryBlock
          stampTimestamp={stampTimestamp}
          onClick={onClick}
          buttonLabel={BUTTON_LABEL[phase]}
        />
      )}

      <div className="paper-foot">
        <span>
          {file} · cycle {cycle.index} of {meta.max_cycles} · p.1
        </span>
        {showFootHint && (
          <span className="paper-foot-hint">tap to skip reveal</span>
        )}
      </div>
    </article>
  );
}

function AuthFooter({
  stampTimestamp,
  onClick,
  buttonLabel,
  showStamp,
}: {
  stampTimestamp: string;
  onClick: (() => void) | undefined;
  buttonLabel: string;
  showStamp: boolean;
}) {
  return (
    <div className="auth">
      <div className="auth-line">
        <span className="paper-lbl">AUTHORIZED</span>
        <span className="auth-rule">
          {showStamp && (
            <span className="stamp stamp-auth">
              AUTHORIZED · {stampTimestamp || nowZulu()} · N.B.
            </span>
          )}
        </span>
      </div>
      <button
        type="button"
        className="paper-btn"
        disabled={onClick === undefined}
        onClick={onClick}
      >
        {buttonLabel}
      </button>
    </div>
  );
}

function OptionsBlock({
  stampTimestamp,
  onClick,
  buttonLabel,
  authorized,
}: {
  stampTimestamp: string;
  onClick: (() => void) | undefined;
  buttonLabel: string;
  authorized: boolean;
}) {
  return (
    <div className="opts-block">
      <div className="opts-label">
        OPERATIONS DESK
        <span className="block-sub"> · Courses of Action</span>
      </div>
      <OptionsList />
      <AuthFooter
        stampTimestamp={stampTimestamp}
        onClick={onClick}
        buttonLabel={buttonLabel}
        showStamp={authorized}
      />
    </div>
  );
}

function EntryBlock({
  stampTimestamp,
  onClick,
  buttonLabel,
}: {
  stampTimestamp: string;
  onClick: (() => void) | undefined;
  buttonLabel: string;
}) {
  const words = useStore((s) => s.cycle?.entry_words ?? []);
  const entryComplete = useStore((s) => s.cycle?.entry_complete ?? false);
  const descriptors = useStore(
    (s) => s.cycle?.delta_descriptors ?? EMPTY_DESCRIPTORS,
  );

  // Only populated once the entry has finished streaming — the player reads
  // the prose first, the mechanical summary second.
  const resultLines = useMemo(
    () => (entryComplete ? buildResultLines(descriptors) : []),
    [entryComplete, descriptors],
  );

  return (
    <div className="sop-block">
      <div className="sop-label">
        OPERATIONS DESK
        <span className="block-sub"> · Case-file entry</span>
      </div>
      <div className="sop-body">
        <StreamingText
          words={words}
          showCaret={!entryComplete}
          complete={entryComplete}
        />
      </div>
      {resultLines.length > 0 && (
        <div className="dispatch-result" aria-label="cycle result">
          <span className="dispatch-result-label">RESULT</span>
          <ul className="dispatch-result-lines">
            {resultLines.map((line) => (
              <li key={line.key}>{line.text}</li>
            ))}
          </ul>
        </div>
      )}
      <AuthFooter
        stampTimestamp={stampTimestamp}
        onClick={onClick}
        buttonLabel={buttonLabel}
        showStamp
      />
    </div>
  );
}
