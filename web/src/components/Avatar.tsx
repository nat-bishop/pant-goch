import { gradientFromSeed, initialsFromName } from "../util/avatar";

interface Props {
  seed: string;
  name: string;
  size?: number;
  variant?: "circle" | "square";
}

export function Avatar({ seed, name, size = 32, variant = "circle" }: Props) {
  const { from, to } = gradientFromSeed(seed);
  const initials = initialsFromName(name);
  const fontSize = Math.max(11, Math.round(size * 0.42));
  const radius = variant === "square" ? size * 0.22 : size / 2;

  return (
    <div
      className="flex shrink-0 select-none items-center justify-center font-medium text-white shadow-sm"
      style={{
        width: size,
        height: size,
        borderRadius: radius,
        background: `linear-gradient(135deg, ${from}, ${to})`,
        fontSize,
        letterSpacing: "0.02em",
      }}
      aria-label={name}
    >
      {initials}
    </div>
  );
}

