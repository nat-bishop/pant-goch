import { useEffect, useMemo, useRef, useState } from "react";
import type { DeltaDescriptors } from "../api";
import type { DayState, LogEntryRecord } from "../store";
import { scrollToDay } from "../lib/scroll";
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

/** Build the RESULT footer lines from a day's accumulated narrative
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
  const day = useStore((s) => s.day);
  const inbox = useStore((s) => s.inbox);
  const authorize = useStore((s) => s.authorize);
  const continueDay = useStore((s) => s.continueDay);
  const [stampTimestamp, setStampTimestamp] = useState<string>("");

  // Scroll the newly-opened day into view on transitions only (not every
  // render). rAF gives the new <section> a tick to mount before scrolling.
  const lastSeenDay = useRef<number | null>(null);
  const dayIndex = day?.index ?? null;
  useEffect(() => {
    if (dayIndex === null) return;
    if (
      lastSeenDay.current !== null &&
      dayIndex !== lastSeenDay.current
    ) {
      requestAnimationFrame(() => scrollToDay(dayIndex));
    }
    lastSeenDay.current = dayIndex;
  }, [dayIndex]);

  // The case-file header timestamp anchors the session open time; it
  // doesn't drift with renders.
  const headerTimestamp = useMemo(() => nowZulu(), []);

  // The just-finished current day (between `entry_word is_last` and
  // `day_opened` for N+1) appears in both `inbox` and `day` — filter it
  // out of the past list so CurrentDaySection owns its rendering. Memoized
  // because Paper re-renders on every streamed word. Hoisted above the
  // early return so hook count stays stable across renders.
  const pastDays = useMemo(
    () => (day ? inbox.filter((e) => e.day_index < day.index) : []),
    [inbox, day],
  );

  if (!day || !meta) {
    return (
      <article className="paper-sheet">
        <CaveatBanner />
        <div className="paper-waiting">Waiting for day…</div>
      </article>
    );
  }

  const canAuth =
    day.awaiting_auth &&
    day.selected_option_index !== null &&
    !day.authorized;
  const phase: AuthPhase = day.awaiting_continue
    ? "continue"
    : day.authorized
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
        ? () => void continueDay()
        : undefined;

  return (
    <article
      className="paper-sheet"
      aria-label={`case file ${meta.codename}`}
      data-has-selection={day.selected_option_index !== null}
      data-authorized={day.authorized}
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

      {pastDays.map((entry) => (
        <PastDaySection key={entry.day_index} entry={entry} />
      ))}

      <CurrentDaySection day={day} />

      <AuthFooter
        stampTimestamp={stampTimestamp}
        onClick={onClick}
        buttonLabel={BUTTON_LABEL[phase]}
        showStamp={day.authorized}
      />
    </article>
  );
}

function DayDivider({
  index,
  action,
  title,
}: {
  index: number;
  action?: string;
  title?: string;
}) {
  const suffix = [action, title].filter(Boolean).join(" · ");
  return (
    <div className="day-divider" aria-hidden>
      <span className="day-divider-rule" />
      <span className="day-divider-label">
        DAY {index}
        {suffix ? ` · ${suffix}` : ""}
      </span>
      <span className="day-divider-rule" />
    </div>
  );
}

function ResultBlock({ lines }: { lines: ResultLine[] }) {
  if (lines.length === 0) return null;
  return (
    <div className="dispatch-result" aria-label="day result">
      <span className="dispatch-result-label">RESULT</span>
      <ul className="dispatch-result-lines">
        {lines.map((line) => (
          <li key={line.key}>{line.text}</li>
        ))}
      </ul>
    </div>
  );
}

function PastDaySection({ entry }: { entry: LogEntryRecord }) {
  const lines = useMemo(
    () => buildResultLines(entry.delta_descriptors ?? {}),
    [entry.delta_descriptors],
  );
  return (
    <section
      id={`day-${entry.day_index}`}
      className="day-section past"
      aria-label={`day ${entry.day_index} archive`}
    >
      <DayDivider
        index={entry.day_index}
        action={entry.action}
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

function CurrentDaySection({ day }: { day: DayState }) {
  const resultLines = useMemo(
    () =>
      day.entry_complete ? buildResultLines(day.delta_descriptors) : [],
    [day.entry_complete, day.delta_descriptors],
  );
  const authorizedTitle =
    day.selected_option_index !== null
      ? (day.options[day.selected_option_index]?.title ?? "")
      : "";

  return (
    <section
      id={`day-${day.index}`}
      className="day-section current"
      aria-label={`day ${day.index} brief`}
    >
      <DayDivider
        index={day.index}
        action={day.entry_action ?? undefined}
        title={day.authorized ? authorizedTitle : undefined}
      />
      <div className="opts-block">
        <div className="opts-label">
          OPERATIONS DESK
          <span className="block-sub"> · Courses of Action</span>
        </div>
        <OptionsList />
      </div>

      {day.authorized && (
        <div className="sop-block">
          <div className="sop-label">
            OPERATIONS DESK
            <span className="block-sub"> · Case-file entry</span>
          </div>
          <div className="sop-body">
            <StreamingText
              words={day.entry_words}
              showCaret={!day.entry_complete}
              complete={day.entry_complete}
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
