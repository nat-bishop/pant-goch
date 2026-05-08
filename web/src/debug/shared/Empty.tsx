// Italic muted text for empty-state messages. Used wherever a panel has
// nothing to render (no ticks yet, no canon, no blocked beats, etc.) —
// shared so all four tabs render the empty state identically.

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="italic text-text-muted/70">{children}</div>;
}
