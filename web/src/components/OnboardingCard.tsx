import { useEffect } from "react";
import type { CastMember } from "../api";
import { useStore } from "../store";

// Role labels shown alongside each cast member in the roster panel.
const ROLE_LABEL: Record<CastMember["role"], string> = {
  agent: "operative",
  handler: "handler",
  boss: "boss",
  opposition: "opposition",
  asset: "asset",
};

// Per-role description shown next to a member's codename in the roster.
// YAML block scalars (`situation: |`) preserve interior newlines for author
// readability; collapse whitespace runs to single spaces so the row reads
// as one continuous clause.
function memberLine(m: CastMember): string {
  const collapsed = m.situation.trim().replace(/\s+/g, " ");
  return collapsed || `(${ROLE_LABEL[m.role]})`;
}

export function OnboardingCard() {
  const meta = useStore((s) => s.mission_meta);
  const dismissed = useStore((s) => s.onboardingDismissed);
  const dismiss = useStore((s) => s.dismissOnboarding);

  // Esc dismisses, like the dispatch archive modal.
  useEffect(() => {
    if (dismissed || !meta) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" || e.key === "Enter") dismiss();
    }
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [dismissed, meta, dismiss]);

  if (dismissed || !meta) return null;

  // Author the cast in a friendly priority for first-time reading: agents
  // (the people you spend), then assets (off-stage support), then handler
  // (who you are), then boss, then opposition. Within a role, mission YAML
  // order is preserved.
  const grouped: Record<CastMember["role"], CastMember[]> = {
    agent: [],
    asset: [],
    handler: [],
    boss: [],
    opposition: [],
  };
  for (const m of meta.cast) grouped[m.role].push(m);
  const orderedRoles: CastMember["role"][] = [
    "agent",
    "asset",
    "handler",
    "boss",
    "opposition",
  ];

  return (
    <div
      className="scrim heavy"
      role="dialog"
      aria-modal="true"
      aria-label="orientation"
    >
      <article className="warm-paper-card onboarding-card">
        <header className="onboarding-head">
          <span className="onboarding-tag">ORIENTATION</span>
          <span className="onboarding-codename">Op. {meta.codename}</span>
        </header>

        <section className="onboarding-section">
          <h3 className="onboarding-section-label">You are RAVEN</h3>
          <p className="onboarding-prose">
            You sit at a desk in a room without windows. Each cycle,{" "}
            <strong>Operations Desk</strong> drafts three or four courses of
            action and presents them to you. You read; you pick one; you
            authorize it. In the field, an operative carries the order out
            and files a dispatch — in their own voice — when the cycle
            closes. That is the entire verb.
          </p>
          <p className="onboarding-prose">
            You have {meta.max_cycles} cycles. The window closes when those
            cycles run out, or when an operative's cover collapses, or when
            the operation is completed. Doing nothing has a cost.
          </p>
        </section>

        {meta.opening_briefing.trim() && (
          <section className="onboarding-section onboarding-briefing">
            <h3 className="onboarding-section-label">Opening briefing</h3>
            <div className="onboarding-briefing-body">
              {meta.opening_briefing.trim()}
            </div>
          </section>
        )}

        <section className="onboarding-section">
          <h3 className="onboarding-section-label">Your people</h3>
          <ul className="onboarding-roster">
            {orderedRoles.flatMap((role) =>
              grouped[role].map((m) => (
                <li key={m.id} className={`roster-row roster-${role}`}>
                  <span className="roster-id">{m.id}</span>
                  <span className="roster-role">{ROLE_LABEL[role]}</span>
                  <span className="roster-note">{memberLine(m)}</span>
                </li>
              )),
            )}
          </ul>
        </section>

        <section className="onboarding-section">
          <h3 className="onboarding-section-label">The meters</h3>
          <ul className="onboarding-legend">
            <li>
              <strong>Cycles left</strong> — turns before the window closes.
              When it hits zero the operation is closed without resolution.
            </li>
            <li>
              <strong>Cover</strong> — an operative's exposure. Each agent
              has theirs. At zero, they're lost.
            </li>
            <li>
              <strong>Operation posture</strong> — where the mission stands
              in narrative terms (papers not yet surfaced, papers in hand,
              safe-house confirmed, exfil running). Operations Desk reports
              it cycle to cycle; you don't track it numerically.
            </li>
          </ul>
        </section>

        <footer className="onboarding-foot">
          <button type="button" className="paper-btn" onClick={dismiss}>
            Begin — cycle 1 ▸
          </button>
        </footer>
      </article>
    </div>
  );
}
