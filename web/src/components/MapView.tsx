// MapView: backdrop is /api/world/map.png; overlay an SVG with hotspot
// pins (here / there / visited / reachable) and the walking tracer.
//
// Hotspot interaction: a reachable (adjacent) pin is clickable whenever
// the backend is awaiting input (`walkReady`) — including mid-scene, so
// the map is a way out of any turn → POST /walk. Reachable pins pulse a
// ring; hovering one previews the route from the current location.
//
// Tracer: a quadratic bezier from from.map_xy → to.map_xy with a control
// point offset perpendicular to the line by 8% of distance — gives all
// paths a subtle drawn-by-hand curve. Progress is driven by the
// `walking.phase` field on the store (depart → arrive-hold → complete),
// using CSS custom properties --tracer-progress and --tracer-len. The
// hover preview reuses the same geometry with a faint dashed stroke.

import { useEffect, useMemo, useState } from "react";
import { postWalk } from "../api";
import { useStore } from "../store";

const VB_W = 100;
const VB_H = 100 * (1536 / 2752); // ≈ 55.81, matches map.png aspect

interface XY { x: number; y: number; }

function toVb(map_xy: [number, number] | null): XY | null {
  if (!map_xy) return null;
  return { x: map_xy[0] * VB_W, y: map_xy[1] * VB_H };
}

/** Quadratic bezier from a → b with a control point offset perpendicular
 * to ab by 8% of distance. Path is returned as an SVG `d` string. */
function tracerPath(a: XY, b: XY): { d: string; len: number } {
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.hypot(dx, dy);
  // Perpendicular unit vector.
  const nx = dist === 0 ? 0 : -dy / dist;
  const ny = dist === 0 ? 0 : dx / dist;
  const offset = dist * 0.08;
  const cx = mx + nx * offset;
  const cy = my + ny * offset;
  const d = `M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`;
  // Approximate quadratic arc length: 0.5 * (chord + control_polyline)
  const chord = dist;
  const cp = Math.hypot(cx - a.x, cy - a.y) + Math.hypot(b.x - cx, b.y - cy);
  const len = 0.5 * (chord + cp);
  return { d, len };
}

export function MapView({
  active,
  currentLocId,
  visitedIds,
}: {
  active: boolean;
  currentLocId: string | null;
  visitedIds: Set<string>;
}) {
  const world = useStore((s) => s.world);
  const walking = useStore((s) => s.walking);
  const walkReady = useStore((s) => s.walkReady);
  const [walkBusy, setWalkBusy] = useState(false);
  // Pin the player is hovering, for the route preview. Only meaningful for
  // a reachable pin; cleared on mouse-out.
  const [hoverId, setHoverId] = useState<string | null>(null);

  // Tracer progress driven by walking.phase.
  const [tracerProgress, setTracerProgress] = useState(0);
  useEffect(() => {
    if (!walking) {
      setTracerProgress(0);
      return;
    }
    if (walking.phase === "depart") {
      // Animate to 75% over the CSS transition (900ms).
      requestAnimationFrame(() => setTracerProgress(0.75));
      // After arrival 75%, transition into arrive-hold automatically via
      // a timer so we get the breathing pulse.
      const t = setTimeout(() => {
        useStore.setState((s) =>
          s.walking && s.walking.phase === "depart"
            ? { walking: { ...s.walking, phase: "arrive-hold" } }
            : s,
        );
      }, 900);
      return () => clearTimeout(t);
    }
    if (walking.phase === "complete") {
      // Play out to 100% then clear walking + snap back to LOCATION.
      requestAnimationFrame(() => setTracerProgress(1));
      const t = setTimeout(() => {
        useStore.setState({ walking: null, worldMode: "location" });
      }, 1000);
      return () => clearTimeout(t);
    }
  }, [walking?.phase]);

  // Cap arrive-hold at 8s.
  useEffect(() => {
    if (!walking || walking.phase !== "arrive-hold") return;
    const cap = setTimeout(() => {
      useStore.setState((s) =>
        s.walking && s.walking.phase === "arrive-hold"
          ? { walking: null, worldMode: "location" }
          : s,
      );
    }, 8000);
    return () => clearTimeout(cap);
  }, [walking?.phase]);

  const locations = world?.locations ?? [];
  const adjacent: string[] = useMemo(() => {
    if (!currentLocId || !world) return [];
    return world.locations.find((l) => l.id === currentLocId)?.adjacent_to ?? [];
  }, [currentLocId, world]);

  const from = walking
    ? toVb(locations.find((l) => l.id === walking.fromId)?.map_xy ?? null)
    : null;
  const to = walking
    ? toVb(locations.find((l) => l.id === walking.toId)?.map_xy ?? null)
    : null;
  const tracer = from && to ? tracerPath(from, to) : null;

  const onPinClick = async (locId: string) => {
    if (!active || walkBusy || !walkReady) return;
    if (!adjacent.includes(locId)) return;
    setWalkBusy(true);
    try {
      await postWalk(locId);
    } catch (e) {
      console.warn("walk failed", e);
    } finally {
      setWalkBusy(false);
    }
  };

  // Route preview: a faint dashed tracer from the current location to the
  // hovered reachable pin. Reuses tracerPath() geometry; only while not
  // walking (the live tracer owns the overlay during a walk).
  const hereXy = currentLocId
    ? toVb(locations.find((l) => l.id === currentLocId)?.map_xy ?? null)
    : null;
  const hoverXy =
    hoverId && !walking
      ? toVb(locations.find((l) => l.id === hoverId)?.map_xy ?? null)
      : null;
  const preview = hereXy && hoverXy ? tracerPath(hereXy, hoverXy) : null;

  return (
    <div className="world-map-inner">
      <div className="world-map-frame">
        <img className="world-map-bg" src="/api/world/map.png" alt="map of Pant Goch" />
        <svg
          className="world-map-overlay"
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {preview && (
            <path
              d={preview.d}
              className="tracer tracer--preview"
              style={{ ["--tracer-len" as never]: preview.len.toFixed(2) }}
            />
          )}
          {tracer && (
            <path
              d={tracer.d}
              className={"tracer" + (walking?.phase === "arrive-hold" ? " held" : "")}
              style={{
                ["--tracer-len" as never]: tracer.len.toFixed(2),
                ["--tracer-progress" as never]: tracerProgress.toFixed(3),
              }}
            />
          )}
          {locations.map((l) => {
            const xy = toVb(l.map_xy);
            if (!xy) return null;
            const isHere = walking ? l.id === walking.fromId : l.id === currentLocId;
            const isThere = walking != null && l.id === walking.toId;
            const isVisited = visitedIds.has(l.id) && !isHere && !isThere;
            // Reachable NOW: an adjacent pin while the backend awaits input
            // (walkReady) and we're not mid-walk. This is the affordance the
            // old map lacked — gated on walkReady so the pulse only shows
            // when a click would actually register.
            const reachable =
              active && !walking && walkReady && adjacent.includes(l.id);
            const clickable = reachable && !walkBusy;
            const r = isHere || isThere ? 0.9 : isVisited ? 0.6 : 0.45;
            return (
              <g key={l.id} transform={`translate(${xy.x}, ${xy.y})`}>
                {reachable && (
                  // Pulsing ring — "you can walk here." Concentric, behind
                  // the pin; animation lives in CSS.
                  <circle r={r + 0.6} className="pin-ring" />
                )}
                <circle
                  r={r}
                  className={
                    "pin" +
                    (isHere ? " pin--here" : "") +
                    (isThere ? " pin--there" : "") +
                    (isVisited ? " pin--visited" : "") +
                    (reachable ? " pin--reachable" : "")
                  }
                />
                <circle
                  r={2.5}
                  className="pin-hit"
                  data-clickable={clickable ? "true" : "false"}
                  tabIndex={clickable ? 0 : -1}
                  role="button"
                  aria-label={
                    clickable ? `walk to ${l.name}` : `${l.name} (not adjacent)`
                  }
                  onClick={() => onPinClick(l.id)}
                  onMouseEnter={() => reachable && setHoverId(l.id)}
                  onMouseLeave={() => setHoverId((h) => (h === l.id ? null : h))}
                  onFocus={() => reachable && setHoverId(l.id)}
                  onBlur={() => setHoverId((h) => (h === l.id ? null : h))}
                  onKeyDown={(e) => {
                    if (clickable && (e.key === "Enter" || e.key === " ")) {
                      e.preventDefault();
                      onPinClick(l.id);
                    }
                  }}
                />
                <text
                  x="1.4"
                  y="0.7"
                  className={
                    "pin-label" +
                    (reachable ? " pin-label--reachable" : "") +
                    (!isHere && !isThere && !isVisited && !reachable
                      ? " pin-label--dim"
                      : "")
                  }
                >
                  {l.name}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
