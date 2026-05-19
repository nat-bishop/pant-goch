// CatchUpLink: shown at the bottom of `.panel-scroll` when the player
// has scrolled up away from the live edge and new content has arrived
// since. Click → smooth-scroll to the live sentinel. Mirrors the
// Discord / ChatGPT "jump to latest" affordance but in the typographic
// register of this UI (mono small-caps, no big pill).

export function CatchUpLink({
  onCatchUp,
}: {
  onCatchUp: () => void;
}) {
  return (
    <button className="catch-up-link" onClick={onCatchUp} type="button">
      ↓ catch up
    </button>
  );
}
