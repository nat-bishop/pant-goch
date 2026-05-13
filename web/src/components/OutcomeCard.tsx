import { isWinOutcome } from "../api";
import { useStore } from "../store";
import { CaveatBanner } from "./CaveatBanner";

export function OutcomeCard() {
  const outcome = useStore((s) => s.outcome);
  const meta = useStore((s) => s.mission_meta);
  const reset = useStore((s) => s.reset);
  const startMission = useStore((s) => s.startMission);

  if (!outcome || !meta) return null;

  const win = isWinOutcome(outcome.outcome_key);
  const stamp = win ? "Mission Complete" : "Mission Failed";

  function restart() {
    reset();
    void startMission();
  }

  return (
    <div className="scrim heavy">
      <article className="paper-sheet outcome-card">
        <CaveatBanner tail="CASE CLOSED" />

        <div className="paper-doctype">
          <span>OUTCOME</span>
          <span className="paper-doctype-meta">Op. {meta.codename}</span>
        </div>
        <div className="paper-rule64" />

        <span className="stamp stamp-large">{stamp}</span>

        <h2 className="outcome-title">
          Op. {meta.codename} —{" "}
          <span className={`outcome-title-result ${win ? "" : "fail"}`}>
            {win ? "SUCCESS" : "FAILURE"}
          </span>
        </h2>

        <div className="outcome-summary">{outcome.summary}</div>

        <div className="outcome-stats">
          <Stat
            label="Days used"
            value={`${outcome.cycles_used} / ${outcome.cycles_max}`}
          />
          <Stat
            label="Agents lost"
            value={outcome.agents_lost.join(", ") || "none"}
          />
          <Stat
            label="Agents alive"
            value={outcome.agents_alive.join(", ") || "none"}
          />
          <Stat label="Outcome key" value={outcome.outcome_key} />
        </div>

        <div className="outcome-actions">
          <button type="button" className="paper-btn" onClick={restart}>
            Restart mission
          </button>
        </div>
      </article>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="outcome-stat-label">{label}</div>
      <div className="outcome-stat-value">{value}</div>
    </div>
  );
}
