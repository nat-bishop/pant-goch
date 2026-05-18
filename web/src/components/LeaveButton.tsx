// Persistent "leave the scene" link. Shared between ChoicesBlock and
// FreeTextInput so the styling and copy stay in one place.

export function LeaveButton({
  onLeave,
  busy,
}: {
  onLeave: () => void;
  busy: boolean;
}) {
  return (
    <button
      className="leave-link"
      onClick={onLeave}
      disabled={busy}
      type="button"
    >
      leave the scene
    </button>
  );
}
