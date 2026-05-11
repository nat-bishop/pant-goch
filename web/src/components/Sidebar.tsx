import { useMemo } from "react";
import { displayActor } from "../api";
import { useStore } from "../store";

interface Props {
  open: boolean;
  onToggle: () => void;
}

export function Sidebar({ open, onToggle }: Props) {
  const meta = useStore((s) => s.mission_meta);
  const state = useStore((s) => s.state);
  const inbox = useStore((s) => s.inbox);
  const openModal = useStore((s) => s.openModal);
  const reversedInbox = useMemo(() => [...inbox].reverse(), [inbox]);

  if (!meta) {
    return (
      <aside className={`desk-side ${open ? "" : "closed"}`}>
        <RailToggle open={open} onToggle={onToggle} />
      </aside>
    );
  }

  const cyclesRemaining =
    (state["mission.cycles_remaining"] as number | undefined) ?? meta.max_cycles;
  const used = Math.max(0, meta.max_cycles - cyclesRemaining);

  return (
    <aside className={`desk-side ${open ? "" : "closed"}`}>
      <RailToggle open={open} onToggle={onToggle} />

      {open ? (
        <>
          <section className="rail-section">
            <div className="rail-label">CYCLE</div>
            <div className="rail-cyc-row">
              <CycleTicks used={used} total={meta.max_cycles} />
              <span className="rail-cyc-num-inline">
                {used + 1}/{meta.max_cycles}
              </span>
            </div>
          </section>

          <section className="rail-section">
            <div className="rail-label">FIELD</div>
            {meta.agents.map((agent) => {
              const cover =
                (state[`agent[${agent.id}].cover`] as number | undefined) ?? 0;
              return (
                <AgentCard
                  key={agent.id}
                  id={agent.id}
                  file={agent.file_number ?? ""}
                  cover={cover}
                  coverMax={10}
                />
              );
            })}
          </section>

          <section className="rail-section">
            <div className="rail-label">INBOX</div>
            {reversedInbox.length === 0 ? (
              <div className="inbox-empty">— no dispatches —</div>
            ) : (
              <div className="inbox-list">
                {reversedInbox.map((d) => (
                  <button
                    key={`${d.cycle_index}-${d.acting}`}
                    type="button"
                    className="inbox-item"
                    onClick={() => openModal(d.cycle_index)}
                  >
                    <span className="inbox-cyc">C{d.cycle_index}</span>
                    <span className="inbox-from">{displayActor(d.acting)}</span>
                    <span className="inbox-snip">"{d.text.slice(0, 60)}…"</span>
                  </button>
                ))}
              </div>
            )}
          </section>

          <div className="rail-foot">handler · authorized</div>
        </>
      ) : (
        <>
          <section className="rail-section">
            <div className="rail-tag">CYC</div>
            <div className="rail-cyc-num">
              {used + 1}/{meta.max_cycles}
            </div>
          </section>
          <section className="rail-section">
            <div className="rail-tag">FLD</div>
            {meta.agents.map((agent) => {
              const cover =
                (state[`agent[${agent.id}].cover`] as number | undefined) ?? 0;
              return (
                <AgentChip key={agent.id} id={agent.id} cover={cover} coverMax={10} />
              );
            })}
          </section>
          <section className="rail-section">
            <div className="rail-tag">INB</div>
            {inbox.length > 0 ? (
              <button
                type="button"
                className="inbox-chip"
                onClick={onToggle}
                title="Open inbox"
                aria-label="Open inbox"
              >
                <span className="inbox-chip-icon">✉</span>
                <span className="inbox-chip-count">{inbox.length}</span>
              </button>
            ) : (
              <div className="inbox-chip inbox-chip-empty" aria-hidden>
                <span className="inbox-chip-icon">·</span>
              </div>
            )}
          </section>

          <div className="rail-foot" aria-hidden />
        </>
      )}
    </aside>
  );
}

function RailToggle({ open, onToggle }: Props) {
  return (
    <button
      type="button"
      className="rail-toggle"
      onClick={onToggle}
      title={open ? "Collapse rail" : "Expand rail"}
      aria-label={open ? "Collapse rail" : "Expand rail"}
    >
      {open ? "‹" : "›"}
    </button>
  );
}

function CycleTicks({ used, total }: { used: number; total: number }) {
  return (
    <span className="cyc-ticks">
      {Array.from({ length: total }).map((_, i) => (
        <span
          key={i}
          className={`cyc-tick ${i < used ? "used" : ""} ${i === used ? "now" : ""}`}
        />
      ))}
    </span>
  );
}

function AgentCard({
  id,
  file,
  cover,
  coverMax,
}: {
  id: string;
  file: string;
  cover: number;
  coverMax: number;
}) {
  const warn = cover <= 5;
  const pct = (cover / coverMax) * 100;
  return (
    <div className={`agent-card ${warn ? "warn" : ""}`}>
      <div className="agent-card-head">
        <span className="agent-card-id">{id}</span>
        <span className="agent-card-cover">
          {cover}/{coverMax}
        </span>
      </div>
      {file && <div className="agent-card-sub">file {file}</div>}
      <div className="agent-card-bar">
        <span style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function AgentChip({
  id,
  cover,
  coverMax,
}: {
  id: string;
  cover: number;
  coverMax: number;
}) {
  const warn = cover <= 5;
  const pct = (cover / coverMax) * 100;
  return (
    <div
      className={`agent-chip ${warn ? "warn" : ""}`}
      title={`${id} · cover ${cover}/${coverMax}`}
    >
      <span className="agent-chip-initial">{id[0]}</span>
      <span className="agent-chip-gauge">
        <span className="agent-chip-fill" style={{ height: `${pct}%` }} />
      </span>
      <span className="agent-chip-num">{cover}</span>
    </div>
  );
}
