import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // Single dark institutional theme — no light/dark toggle.
  theme: {
    extend: {
      fontFamily: {
        // Default sans IS the typewriter. Mockup uses Courier Prime everywhere.
        sans: ['"Courier Prime"', '"Courier New"', "monospace"],
        mono: ['"Courier Prime"', '"Courier New"', "monospace"],
      },
      colors: {
        desk: "var(--desk)",
        "desk-2": "var(--desk-2)",
        "desk-rule": "var(--desk-rule)",
        "desk-rule-soft": "var(--desk-rule-soft)",
        paper: "var(--paper)",
        "paper-shade": "var(--paper-shade)",
        "paper-onion": "var(--paper-onion)",
        ink: "var(--ink)",
        "ink-soft": "var(--ink-soft)",
        "ink-faint": "var(--ink-faint)",
        rule: "var(--rule)",
        "rule-soft": "var(--rule-soft)",
        "chrome-bright": "var(--chrome-bright)",
        chrome: "var(--chrome)",
        "chrome-dim": "var(--chrome-dim)",
        "chrome-faint": "var(--chrome-faint)",
        stamp: "var(--stamp)",
        "stamp-soft": "var(--stamp-soft)",
      },
      keyframes: {
        "word-in": {
          from: { opacity: "0", transform: "translateY(2px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "paper-rise": {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "word-in": "word-in 80ms ease-out both",
        "paper-rise": "paper-rise 0.9s 0.05s cubic-bezier(0.2, 0.8, 0.2, 1) both",
      },
    },
  },
  plugins: [],
} satisfies Config;
