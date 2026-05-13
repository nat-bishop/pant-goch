import { useEffect, useMemo, useRef, useState } from "react";
import type { DeltaDescriptors } from "../api";
import type { CycleState, LogEntryRecord } from "../store";
import { scrollToCycle } from "../lib/scroll";
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

interface ResultLine {
  key: string;
  text: string;
}

/** Build the RESULT footer lines from a cycle's accumulated narrative
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
  return lines.sort(
    (a, b) =>
      (descriptors[a.key].agent_id ? 0 : 1) -
      (descriptors[b.key].agent_id ? 0 : 1),
  );
}

export function Paper() {
  const meta = useStore((s) => s.mission_meta);
  const cycle = useStore((s) => s.cycle);
  const inbox = useStore((s) => s.inbox);
  const authorize = useStore((s) => s.authorize);
  const continueCycle = useStore((s) => s.continueCycle);
  const [stampTimestamp, setStampTimestamp] = useState<string>("");

  // Scroll the newly-opened cycle into view on transitions only (not every
  // render). rAF gives the new <section> a tick to mount before scrolling.
  const lastSeenCycle = useRef<number | null>(null);
  const cycleIndex = cycle?.index ?? null;
  useEffect(() => {
    if (cycleIndex === null) return;
    if (
      lastSeenCycle.current !== null &&
      cycleIndex !== lastSeenCycle.current
    ) {
      requestAnimationFrame(() => scrollToCycle(cycleIndex));
    }
    lastSeenCycle.current = cycleIndex;
  }, [cycleIndex]);

  if (!cycle || !meta) {
    return (
      <article className="paper-sheet">
        <CaveatBanner />
        <div className="paper-waiting">Waiting for cycle…</div>
      </article>
    );
  }

  // The case-file header timestamp anchors the session open time; it
  // doesn't drift with renders.
  const headerTimestamp = useMemo(() => nowZulu(), []);
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

  // The just-finished current cycle (between `entry_word is_last` and
  // `cycle_opened` for N+1) appears in both `inbox` and `cycle` — filter it
  // out of the past list so CurrentCycleSection owns its rendering. Memoized
  // because Paper re-renders on every streamed word.
  const pastCycles = useMemo(
    () => inbox.filter((e) => e.cycle_index < cycle.index),
    [inbox, cycle.index],
  );

  return (
    <article
      className="paper-sheet"
      aria-label={`case file ${meta.codename}`}
      data-has-selection={cycle.selected_option_index !== null}
      data-authorized={cycle.authorized}
    >
      <CaveatBanner />

      <div className="paper-doctype">
        <span>CASE FILE</span>
        <span className="paper-doctype-meta">
          Op. {meta.codename} · {headerTimestamp}
        </span>
      </div>
      <div className="paper-rule64" />
      <div className="status-divider" />

      {pastCycles.map((entry) => (
        <PastCycleSection key={entry.cycle_index} entry={entry} />
      ))}

      <CurrentCycleSection cycle={cycle} />

      <AuthFooter
        stampTimestamp={stampTimestamp}
        onClick={onClick}
        buttonLabel={BUTTON_LABEL[phase]}
        showStamp={cycle.authorized}
      />
    </article>
  );
}

function CycleDivider({
  index,
  archetype,
  title,
}: {
  index: number;
  archetype?: string;
  title?: string;
}) {
  const suffix = [archetype, title].filter(Boolean).join(" · ");
  return (
    <div className="cycle-divider" aria-hidden>
      <span className="cycle-divider-rule" />
      <span className="cycle-divider-label">
        DAY {index}
        {suffix ? ` · ${suffix}` : ""}
      </span>
      <span className="cycle-divider-rule" />
    </div>
  );
}

function ResultBlock({ lines }: { lines: ResultLine[] }) {
  if (lines.length === 0) return null;
  return (
    <div className="dispatch-result" aria-label="cycle result">
      <span className="dispatch-result-label">RESULT</span>
      <ul className="dispatch-result-lines">
        {lines.map((line) => (
          <li key={line.key}>{line.text}</li>
        ))}
      </ul>
    </div>
  );
}

function PastCycleSection({ entry }: { entry: LogEntryRecord }) {
  const lines = useMemo(
    () => buildResultLines(entry.delta_descriptors ?? {}),
    [entry.delta_descriptors],
  );
  return (
    <section
      id={`cycle-${entry.cycle_index}`}
      className="cycle-section past"
      aria-label={`cycle ${entry.cycle_index} archive`}
    >
      <CycleDivider
        index={entry.cycle_index}
        archetype={entry.archetype}
        title={entry.title}
      />
      <div className="sop-block">
        <div className="sop-label">
          OPERATIONS DESK
          <span className="block-sub"> · Case-file entry</span>
        </div>
        <div className="sop-body">{entry.text}</div>
        <ResultBlock lines={lines} />
      </div>
    </section>
  );
}

function CurrentCycleSection({ cycle }: { cycle: CycleState }) {
  const resultLines = useMemo(
    () =>
      cycle.entry_complete ? buildResultLines(cycle.delta_descriptors) : [],
    [cycle.entry_complete, cycle.delta_descriptors],
  );
  const authorizedTitle =
    cycle.selected_option_index !== null
      ? (cycle.options[cycle.selected_option_index]?.title ?? "")
      : "";

  return (
    <section
      id={`cycle-${cycle.index}`}
      className="cycle-section current"
      aria-label={`cycle ${cycle.index} brief`}
    >
      <CycleDivider
        index={cycle.index}
        archetype={cycle.entry_archetype ?? undefined}
        title={cycle.authorized ? authorizedTitle : undefined}
      />
      <div className="opts-block">
        <div className="opts-label">
          OPERATIONS DESK
          <span className="block-sub"> · Courses of Action</span>
        </div>
        <OptionsList />
      </div>

      {cycle.authorized && (
        <div className="sop-block">
          <div className="sop-label">
            OPERATIONS DESK
            <span className="block-sub"> · Case-file entry</span>
          </div>
          <div className="sop-body">
            <StreamingText
              words={cycle.entry_words}
              showCaret={!cycle.entry_complete}
              complete={cycle.entry_complete}
            />
          </div>
          <ResultBlock lines={resultLines} />
        </div>
      )}
    </section>
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
