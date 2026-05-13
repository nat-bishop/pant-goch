import { useEffect } from "react";
import { useStore } from "../store";
import { PacedReveal } from "./PacedReveal";
import { usePacedWords } from "../hooks/usePacedWords";

// Option B (see _mockups/handler-onboarding-options.html): orientation
// surface holds only the briefing in voice plus one signpost line.
// Roster and meter glossary live off this surface (dossier panel /
// HUD tooltips).
export function OnboardingCard() {
  const meta = useStore((s) => s.mission_meta);
  const dismissed = useStore((s) => s.onboardingDismissed);
  const dismiss = useStore((s) => s.dismissOnboarding);

  const briefing = meta?.opening_briefing?.trim() ?? "";
  const PACE_MS = 80;
  const { tokens, flushed, complete, flush } = usePacedWords(briefing, PACE_MS);

  useEffect(() => {
    if (dismissed || !meta) return;
    function onKey(e: KeyboardEvent) {
      // Esc skips orientation outright. Enter flushes if still streaming,
      // otherwise dismisses — so a player hammering Enter doesn't blow
      // past the briefing they haven't read.
      if (e.key === "Escape") {
        dismiss();
      } else if (e.key === "Enter") {
        if (complete) dismiss();
        else flush();
      }
    }
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [dismissed, meta, dismiss, complete, flush]);

  if (dismissed || !meta) return null;

  return (
    <div
      className="scrim heavy"
      role="dialog"
      aria-modal="true"
      aria-label="orientation"
    >
      <article
        className="warm-paper-card onboarding-card"
        onClickCapture={flush}
      >
        <header className="onboarding-head">
          <span className="onboarding-tag">ORIENTATION</span>
          <span className="onboarding-codename">Op. {meta.codename}</span>
        </header>

        <div className="onboarding-briefing-body">
          <PacedReveal tokens={tokens} flushed={flushed} paceMs={PACE_MS} />
        </div>

        <footer className="onboarding-foot">
          <span className="onboarding-foot-note">
            Operations Desk will draft your options. You authorize one.
          </span>
          <button
            type="button"
            className="paper-btn"
            onClick={dismiss}
            disabled={!complete}
          >
            Begin — day 1 ▸
          </button>
        </footer>
      </article>
    </div>
  );
}
