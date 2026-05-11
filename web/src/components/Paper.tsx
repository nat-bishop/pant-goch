import { useEffect, useState } from "react";
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

export function Paper() {
  const meta = useStore((s) => s.mission_meta);
  const cycle = useStore((s) => s.cycle);
  const authorize = useStore((s) => s.authorize);
  const continueCycle = useStore((s) => s.continueCycle);
  const [optsVisible, setOptsVisible] = useState(false);
  const [stampTimestamp, setStampTimestamp] = useState<string>("");

  // 480ms idle after the SoP completes — matches the design's stage 2→3
  // transition so the player reads State of Play before the options appear.
  useEffect(() => {
    if (!cycle) {
      setOptsVisible(false);
      return;
    }
    if (cycle.sop_complete) {
      const t = setTimeout(() => setOptsVisible(true), 480);
      return () => clearTimeout(t);
    }
    setOptsVisible(false);
  }, [cycle?.index, cycle?.sop_complete]);

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
    !cycle.sop_complete ||
    (cycle.dispatch_acting !== null && !cycle.dispatch_complete);

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

      <div className="sop-block">
        <div className="sop-label">STATE OF PLAY</div>
        <div className="sop-body">
          <StreamingText
            words={cycle.sop_words}
            showCaret={!cycle.sop_complete}
            complete={cycle.sop_complete}
          />
        </div>
      </div>

      <div className={`opts-block ${optsVisible ? "in" : ""}`}>
        <div className="opts-label">FOR AUTHORIZATION</div>
        <OptionsList visible={optsVisible} />

        <div className="auth">
          <div className="auth-line">
            <span className="paper-lbl">AUTHORIZED</span>
            <span className="auth-rule">
              {cycle.authorized && (
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
            {BUTTON_LABEL[phase]}
          </button>
        </div>
      </div>

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
