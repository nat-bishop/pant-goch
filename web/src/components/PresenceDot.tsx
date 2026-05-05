interface Props {
  size?: number;
}

export function PresenceDot({ size = 10 }: Props) {
  return (
    <span
      className="relative inline-block"
      style={{ width: size, height: size }}
      aria-label="Active"
    >
      <span
        className="absolute inset-0 rounded-full bg-presence-active"
        style={{ boxShadow: "0 0 0 2px var(--bg)" }}
      />
      <span
        className="absolute inset-0 rounded-full bg-presence-active animate-presence-pulse"
        style={{ transformOrigin: "center" }}
      />
    </span>
  );
}
