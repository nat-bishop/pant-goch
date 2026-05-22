// CastBoard = the dossier's "People" section: the journalist's read on the
// villagers they have met. One card per met, *tracked* NPC — those with a
// tracked relationship; auxiliary villagers stay off the board. The state is
// diegetic: a band-driven tint and the reporter's own impression, never a
// number. Driven entirely by `npcStandings`, which the backend emits at scene
// entry and only when a band actually changes (a thaw, a breach).

import { useStore } from "../store";

/** Two-letter monogram for the "no photograph on file" slot. Initials of the
 * last two name words (so "Old Tom Bevan" → TB, "DC Helen Mortimer" → HM). */
function monogram(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  const [a, b] = words.slice(-2);
  return (a[0] + b[0]).toUpperCase();
}

export function CastBoard() {
  const world = useStore((s) => s.world);
  const npcLocations = useStore((s) => s.npcLocations);
  const npcStandings = useStore((s) => s.npcStandings);

  // Stable order = cast.yaml order, filtered to NPCs the player has met
  // (a standing has arrived for them). Nothing reshuffles as bands shift.
  const met = (world?.npcs ?? []).filter((n) => npcStandings[n.id]);

  if (met.length === 0) {
    return (
      <div className="notes-empty">
        No one has given you the time of day yet.
      </div>
    );
  }

  const locName = (locId: string | undefined): string | null =>
    world?.locations.find((l) => l.id === locId)?.name ?? null;

  return (
    <div className="cast-list">
      {met.map((npc) => {
        const standing = npcStandings[npc.id];
        const where = locName(npcLocations[npc.id] ?? npc.default_location);
        return (
          <div className="cast-card" key={npc.id} data-band={standing.band}>
            <div className="cast-slot" data-band={standing.band} aria-hidden="true">
              {monogram(npc.name)}
            </div>
            <div className="cast-body">
              <h3 className="cast-name">{npc.name}</h3>
              {where && (
                <div className="cast-meta">last seen · {where.toLowerCase()}</div>
              )}
              {standing.impression && (
                <p className="cast-impression">{standing.impression}</p>
              )}
              {standing.band === "closed" && (
                <div className="cast-closed">— closed —</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
