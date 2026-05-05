// Deterministic gradient generator for in-world avatars.
// Same seed → same gradient, so Aria's identity is stable across reloads.

function hash(seed: string): number {
  let h = 5381;
  for (let i = 0; i < seed.length; i++) {
    h = (h * 33) ^ seed.charCodeAt(i);
  }
  return h >>> 0;
}

export interface AvatarGradient {
  from: string;
  to: string;
}

export function gradientFromSeed(seed: string): AvatarGradient {
  const h = hash(seed);
  const hue1 = h % 360;
  const hue2 = (hue1 + 40 + ((h >> 8) % 60)) % 360;
  return {
    from: `hsl(${hue1} 55% 52%)`,
    to: `hsl(${hue2} 60% 38%)`,
  };
}

export function initialsFromName(name: string): string {
  // Drop any "| affiliation" suffix so display names like
  // "Aria | Continuity Program" initialize as "A", not "AP".
  const main = name.split(/\s*\|\s*/)[0]?.trim() ?? "";
  const words = main.split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0]![0]!.toUpperCase();
  return (words[0]![0]! + words[words.length - 1]![0]!).toUpperCase();
}
