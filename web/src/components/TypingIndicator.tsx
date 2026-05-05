import { Avatar } from "./Avatar";

interface Props {
  seed: string;
  name: string;
}

export function TypingIndicator({ seed, name }: Props) {
  return (
    <div className="flex items-end gap-2" aria-live="polite" aria-label={`${name} is typing`}>
      <Avatar seed={seed} name={name} size={28} />
      <div
        className="flex items-center gap-1.5 rounded-2xl bg-bg-bubble-them px-4 py-3"
        style={{ borderBottomLeftRadius: "0.25rem" }}
      >
        <Dot delayMs={0} />
        <Dot delayMs={160} />
        <Dot delayMs={320} />
      </div>
    </div>
  );
}

function Dot({ delayMs }: { delayMs: number }) {
  return (
    <span
      className="block h-1.5 w-1.5 rounded-full bg-text-muted animate-dot-bounce"
      style={{ animationDelay: `${delayMs}ms` }}
    />
  );
}
