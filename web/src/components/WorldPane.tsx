// WorldPane = the LEFT pane: backdrop photo + tabs + slate + overlays.
// Switches between three modes (location / map / notes). During a walk,
// MAP is forced active and tabs are disabled.

import { useEffect, useMemo } from "react";
import { useStore } from "../store";
import { ImageSlots } from "./ImageSlots";
import { MapView } from "./MapView";
import { NotesView } from "./NotesView";

export function WorldPane() {
  const phase = useStore((s) => s.phase);
  const scenes = useStore((s) => s.scenes);
  const walking = useStore((s) => s.walking);
  const worldMode = useStore((s) => s.worldMode);
  const setWorldMode = useStore((s) => s.setWorldMode);
  const currentChapterIdx = useStore((s) => s.currentChapterIdx);
  const npcLocations = useStore((s) => s.npcLocations);
  const world = useStore((s) => s.world);

  // The chapter currently centred in the right panel decides the backdrop
  // (so scrolling back swaps the photo).
  const visibleScene = scenes[currentChapterIdx] ?? scenes[scenes.length - 1] ?? null;
  const currentLocId = walking ? walking.toId : visibleScene?.location_id ?? world?.player_starts_at ?? null;

  const visitedIds = useMemo(() => {
    const set = new Set<string>();
    for (const s of scenes) set.add(s.location_id);
    return set;
  }, [scenes]);

  // Preload destination backdrop on a new walk so the crossfade hits cache.
  useEffect(() => {
    if (!walking) return;
    const img = new Image();
    img.src = `/api/world/backdrops/${walking.toId}.png`;
  }, [walking?.toId]);

  // During walks, force map.
  const effectiveMode = walking ? "map" : worldMode;
  const showMap = effectiveMode === "map";
  const showNotes = effectiveMode === "notes";
  const showLocation = effectiveMode === "location";

  // Backdrop URL: while idle (frontispiece), the authored cover.
  // During walks, the destination location's backdrop. Otherwise, the
  // currently-visible scene's backdrop.
  const backdropUrl = phase === "idle"
    ? "/api/world/cover.png"
    : currentLocId
      ? `/api/world/backdrops/${currentLocId}.png`
      : null;

  const npcsHere = useMemo(() => {
    if (!world || !currentLocId) return [];
    return world.npcs
      .filter((n) => (npcLocations[n.id] ?? n.default_location) === currentLocId)
      .map((n) => n.name);
  }, [world, npcLocations, currentLocId]);

  const locationName = walking
    ? world?.locations.find((l) => l.id === walking.toId)?.name ?? walking.toId
    : visibleScene?.location_name ?? null;

  return (
    <div className="world">
      <ImageSlots currentUrl={backdropUrl} />
      <div className="world-tint" />
      <div className="world-grain" />

      {phase !== "idle" && (
        <div className="world-tabs" role="tablist" aria-label="left pane mode">
          {(["location", "map", "notes"] as const).map((m) => (
            <button
              key={m}
              role="tab"
              aria-selected={effectiveMode === m}
              className={"world-tab" + (effectiveMode === m ? " active" : "")}
              onClick={() => setWorldMode(m)}
              disabled={walking != null}
            >
              {m}
            </button>
          ))}
        </div>
      )}

      <div className={"world-overlay world-overlay--map" + (showMap && phase !== "idle" ? " in" : "")}>
        <div className="world-overlay-eyebrow">
          {walking ? "on the road" : "the survey · pant goch & the marches"}
        </div>
        <MapView
          active={phase !== "idle" && phase !== "game_over" && !walking}
          currentLocId={walking ? walking.fromId : visibleScene?.location_id ?? null}
          visitedIds={visitedIds}
        />
        {walking && walking.phase === "arrive-hold" && (
          <div className="world-overlay-foot">
            approaching {world?.locations.find((l) => l.id === walking.toId)?.name ?? walking.toId}
          </div>
        )}
        {!walking && (
          <div className="world-overlay-foot foot-meta">
            {visitedIds.size} of {world?.locations.length ?? 0} visited
          </div>
        )}
      </div>

      <div className={"world-overlay world-overlay--notes" + (showNotes && phase !== "idle" ? " in" : "")}>
        <div className="world-overlay-eyebrow">case file · m. pritchard</div>
        <NotesView />
      </div>

      {showLocation && phase !== "idle" && locationName && (
        <div className="world-slate">
          <div className="world-slate-eyebrow">present location</div>
          <div className="world-slate-name">{locationName}</div>
          {npcsHere.length > 0 && (
            <div className="world-slate-npcs">
              — {npcsHere.join(", ").toLowerCase()} —
            </div>
          )}
        </div>
      )}
    </div>
  );
}
