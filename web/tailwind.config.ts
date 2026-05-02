import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
        serif: ['"IBM Plex Serif"', "Georgia", "serif"],
      },
      colors: {
        paper: "#f7f4ec",
        "paper-hover": "#f0ece2",
        ink: "#1c1c1a",
        "ink-soft": "#3a3633",
        muted: "#6b665e",
        line: "#e2dcce",
        warm: "#a14b1f",
        cool: "#3a5a78",
      },
    },
  },
  plugins: [],
} satisfies Config;
