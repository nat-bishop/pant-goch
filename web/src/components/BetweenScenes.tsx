// Between-scenes signpost — rendered when a Leave closes the scene with
// no options/free-text, so the player isn't left on a dead screen. Owns
// its own walkReady/setWorldMode subscriptions so the memoized Chapter
// history doesn't re-render when the walk pick arms.

import { useStore } from "../store";

export function BetweenScenes({ show }: { show: boolean }) {
  const walkReady = useStore((s) => s.walkReady);
  const setWorldMode = useStore((s) => s.setWorldMode);

  if (!show) return null;
  if (!walkReady) {
    return (
      <div className="between-scenes between-scenes--writing">
        Writing up the case file…
      </div>
    );
  }
  return (
    <button
      type="button"
      className="between-scenes between-scenes--go"
      onClick={() => setWorldMode("map")}
    >
      The scene closes. Choose where to go next on the map →
    </button>
  );
}
