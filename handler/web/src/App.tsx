import { memo, useEffect, useMemo, useState } from "react";
import { postPick, postWalk, startSession } from "./api";
import { Map, type MapLocation } from "./components/Map";
import { useStream } from "./hooks/useStream";
import { useStore, type SceneSnapshot } from "./store";

interface WorldMeta {
  player_starts_at: string;
  locations: {
    id: string;
    name: string;
    adjacent_to: string[];
    map_xy: [number, number] | null;
  }[];
  has_map: boolean;
}

interface LocationLookup {
  byId: Record<string, MapLocation>;
  ordered: MapLocation[];
  has_map: boolean;
}

function useWorld(): LocationLookup | null {
  const [meta, setMeta] = useState<WorldMeta | null>(null);
  useEffect(() => {
    fetch("/api/world")
      .then((r) => r.json())
      .then((m: WorldMeta) => setMeta(m))
      .catch((e) => console.warn("/world failed", e));
  }, []);
  return useMemo(() => {
    if (!meta) return null;
    const byId: LocationLookup["byId"] = {};
    for (const loc of meta.locations) {
      byId[loc.id] = loc;
    }
    return { byId, ordered: meta.locations, has_map: meta.has_map };
  }, [meta]);
}

export default function App() {
  const phase = useStore((s) => s.phase);
  const scenes = useStore((s) => s.scenes);
  const case_file = useStore((s) => s.case_file);
  const game_over_ending = useStore((s) => s.game_over_ending);
  const error = useStore((s) => s.error);
  const resetForRestart = useStore((s) => s.resetForRestart);

  const [streaming, setStreaming] = useState(false);
  useStream(streaming);

  const world = useWorld();
  const current = scenes[scenes.length - 1] ?? null;
  const currentLocation = current?.location_name ?? "—";

  const onStart = async () => {
    resetForRestart();
    await startSession();
    // Defer stream activation by one tick so the reset propagates
    // before the new events arrive.
    setTimeout(() => setStreaming(true), 0);
  };

  const onRestart = async () => {
    setStreaming(false);
    await onStart();
  };

  // ── render ─────────────────────────────────────────────────────────

  // Map is active when the player can actually walk: between scenes.
  // Active=false means scene play in progress; hotspots dim, tooltip
  // explains the player must use a "Leave for…" option to walk away.
  const mapActive = phase === "between_scenes";
  const adjacent =
    current && world ? world.byId[current.location_id]?.adjacent_to ?? [] : [];

  return (
    <div className="app">
      <div className="scene-pane">
        <Header location={phase === "idle" ? "" : currentLocation} />
        {phase !== "idle" && world && world.has_map && current && (
          <Map
            currentId={current.location_id}
            locations={world.ordered}
            adjacent={adjacent}
            active={mapActive}
          />
        )}
        <div className="scene-body">
          {phase === "idle" ? (
            <IdleStage onStart={onStart} error={error} />
          ) : (
            <>
              <SceneHistory scenes={scenes} />
              {phase === "awaiting_pick" && current && (
                <OptionsPanel
                  options={current.turns[current.turns.length - 1]?.options ?? []}
                />
              )}
              {phase === "between_scenes" && current && world && !world.has_map && (
                <WalkPanel currentId={current.location_id} world={world} />
              )}
              {phase === "between_scenes" && world && world.has_map && (
                <div className="walk-hint">
                  Click an adjacent location on the map to walk there.
                </div>
              )}
              {phase === "game_over" && (
                <GameOverPanel ending={game_over_ending ?? ""} onRestart={onRestart} />
              )}
              {phase === "stranded" && <StrandedPanel onRestart={onRestart} />}
            </>
          )}
        </div>
      </div>
      <CaseFilePane case_file={case_file} />
    </div>
  );
}

function IdleStage({ onStart, error }: { onStart: () => void; error: string | null }) {
  return (
    <div className="center-stage">
      <h1>HANDLER</h1>
      <p className="subtitle">
        Pant Goch — a Welsh Marches village where someone died on the wrong
        bend of the wrong road in 2019. You have six weeks to find out who
        was driving.
      </p>
      <button className="start-btn" onClick={onStart}>
        Begin investigation
      </button>
      {error && <p style={{ marginTop: "1rem", color: "var(--error)" }}>{error}</p>}
    </div>
  );
}

// ─── header ───────────────────────────────────────────────────────────────

function Header({ location }: { location: string }) {
  return (
    <div className="header">
      <span className="header-title">HANDLER · PANT GOCH</span>
      <span className="header-location">{location}</span>
    </div>
  );
}

// ─── streamed prose history ───────────────────────────────────────────────

const SceneCard = memo(function SceneCard({
  scene,
  isCurrent,
}: {
  scene: SceneSnapshot;
  isCurrent: boolean;
}) {
  return (
    <div>
      {scene.turns.map((turn, turnIdx) => {
        const isLatest = isCurrent && turnIdx === scene.turns.length - 1;
        return (
          <div
            key={turnIdx}
            className={
              "prose-turn " +
              (isLatest ? "prose-turn-current" : "prose-turn-prior")
            }
          >
            <StreamingText words={turn.words} streaming={turn.streaming && isLatest} />
            {turn.player_pick && (
              <div className="player-pick">› {turn.player_pick.text}</div>
            )}
          </div>
        );
      })}
    </div>
  );
});

function SceneHistory({ scenes }: { scenes: SceneSnapshot[] }) {
  // Prior scenes' SceneCards skip reconciliation entirely once their
  // turns array reference stops changing — `memo` short-circuits because
  // the `scene` prop is referentially stable for closed scenes. Only the
  // current scene's card re-renders per prose_word.
  return (
    <div className="prose">
      {scenes.map((scene, sceneIdx) => (
        <SceneCard
          key={scene.scene_id}
          scene={scene}
          isCurrent={sceneIdx === scenes.length - 1}
        />
      ))}
    </div>
  );
}

function StreamingText({ words, streaming }: { words: string[]; streaming: boolean }) {
  // Each word fades in as it arrives. The CSS animation runs once; React
  // doesn't re-trigger it on re-render because the keyed span identity
  // is stable.
  return (
    <span>
      {words.map((w, i) => (
        <span key={i} className="streamed-word">
          {w}
        </span>
      ))}
      {streaming && <span style={{ opacity: 0.4 }}>▍</span>}
    </span>
  );
}

// ─── options ──────────────────────────────────────────────────────────────

function OptionsPanel({ options }: { options: { text: string; leaves_for: string | null }[] }) {
  const [busy, setBusy] = useState(false);
  const attachPick = useStore((s) => s.attachPick);
  const onPick = async (idx: number) => {
    if (busy) return;
    setBusy(true);
    attachPick(options[idx]);
    try {
      await postPick(idx);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="options">
      {options.map((opt, i) => (
        <button
          key={i}
          className={opt.leaves_for ? "option-leave" : ""}
          onClick={() => onPick(i)}
          disabled={busy}
        >
          {opt.text}
          {opt.leaves_for && (
            <span className="option-leave-tag"> → walk</span>
          )}
        </button>
      ))}
    </div>
  );
}

// ─── walk targets (between scenes, fallback when no map authored) ────────

function WalkPanel({
  currentId,
  world,
}: {
  currentId: string;
  world: LocationLookup;
}) {
  const cur = world.byId[currentId];
  const [busy, setBusy] = useState(false);
  if (!cur) return null;
  const onWalk = async (to: string) => {
    if (busy) return;
    setBusy(true);
    try {
      await postWalk(to);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="walk-targets">
      <div className="walk-header">— Walk to —</div>
      {cur.adjacent_to.map((id) => (
        <button key={id} onClick={() => onWalk(id)} disabled={busy}>
          {world.byId[id]?.name ?? id}
        </button>
      ))}
    </div>
  );
}

// ─── game over / stranded ─────────────────────────────────────────────────

function GameOverPanel({ ending, onRestart }: { ending: string; onRestart: () => void }) {
  return (
    <div className="center-stage">
      <div className="ending-id">— {ending.replace(/_/g, " ").toUpperCase()} —</div>
      <button onClick={onRestart}>Start a new investigation</button>
    </div>
  );
}

function StrandedPanel({ onRestart }: { onRestart: () => void }) {
  return (
    <div className="center-stage">
      <div className="subtitle">
        The conversation has ended with nowhere to go from here.
      </div>
      <button onClick={onRestart}>Start a new investigation</button>
    </div>
  );
}

// ─── case-file sidebar ────────────────────────────────────────────────────

function CaseFilePane({
  case_file,
}: {
  case_file: ReturnType<typeof useStore.getState>["case_file"];
}) {
  return (
    <div className="casefile-pane">
      <div className="casefile-header">CASE FILE</div>
      <div className="casefile-body">
        {case_file.length === 0 && (
          <div className="casefile-empty">— no entries yet —</div>
        )}
        {case_file.map((entry, i) => {
          const isCurrent = i === case_file.length - 1 && entry.streaming;
          return (
            <div
              key={entry.scene_id}
              className={
                "casefile-entry" +
                (isCurrent ? " casefile-entry-current" : "")
              }
            >
              <div className="casefile-entry-anchor">— Scene {entry.scene_id} —</div>
              <StreamingText words={entry.words} streaming={entry.streaming} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
