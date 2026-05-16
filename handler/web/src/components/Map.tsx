import { useState } from "react";
import { postWalk } from "../api";

// Per location_id → fractional (x, y) ∈ [0, 1] over the map image.
export type MapCoords = Record<string, [number, number]>;

export interface MapLocation {
  id: string;
  name: string;
  adjacent_to: string[];
  map_xy: [number, number] | null;
}

export interface MapProps {
  /** Player's current location id. Marked with a permanent dot. */
  currentId: string;
  /** All locations + adjacencies + coords. Locations without map_xy
   * are not rendered as hotspots (defensive — production worlds with
   * a map should author coords for all locations). */
  locations: MapLocation[];
  /** Adjacent location ids (the only ones walkable from currentId). */
  adjacent: string[];
  /** When true, adjacent hotspots are clickable + glowing.
   * When false (scene in progress), all hotspots are dim and inert. */
  active: boolean;
}

export function Map({ currentId, locations, adjacent, active }: MapProps) {
  const [busy, setBusy] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);

  const onWalk = async (to: string) => {
    if (busy || !active) return;
    if (!adjacent.includes(to)) return; // defensive — non-adjacent click
    setBusy(true);
    try {
      await postWalk(to);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="map">
      <img
        className="map-image"
        src="/api/world/map.png"
        alt="Map of Pant Goch"
        draggable={false}
      />
      {locations.map((loc) => {
        if (!loc.map_xy) return null;
        const [x, y] = loc.map_xy;
        const isCurrent = loc.id === currentId;
        const isAdjacent = adjacent.includes(loc.id);
        const clickable = active && isAdjacent && !busy;
        const cls = [
          "map-hotspot",
          isCurrent && "map-hotspot-current",
          clickable && "map-hotspot-clickable",
          !clickable && !isCurrent && "map-hotspot-dim",
        ]
          .filter(Boolean)
          .join(" ");
        const tooltip = !active
          ? "Pick a 'Leave for…' option to walk away."
          : isAdjacent
            ? `Walk to ${loc.name}`
            : `${loc.name} (not adjacent)`;
        return (
          <button
            key={loc.id}
            type="button"
            className={cls}
            style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
            onClick={() => onWalk(loc.id)}
            onMouseEnter={() => setHovered(loc.id)}
            onMouseLeave={() => setHovered((h) => (h === loc.id ? null : h))}
            disabled={!clickable && !isCurrent}
            aria-label={tooltip}
            title={tooltip}
          >
            {hovered === loc.id && (
              <span className="map-hotspot-label">{loc.name}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
