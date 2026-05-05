import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"Inter Variable"',
          '"Inter"',
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Arial",
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        bg: "var(--bg)",
        "bg-elevated": "var(--bg-elevated)",
        "bg-sidebar": "var(--bg-sidebar)",
        "bg-bubble-them": "var(--bg-bubble-them)",
        text: "var(--text)",
        "text-muted": "var(--text-muted)",
        "text-on-accent": "var(--text-on-accent)",
        border: "var(--border)",
        accent: "var(--accent)",
        "accent-hover": "var(--accent-hover)",
        "accent-glow": "var(--accent-glow)",
        "presence-active": "var(--presence-active)",
        unread: "var(--unread)",
        "meter-cool": "var(--meter-cool)",
        "meter-warn": "var(--meter-warn)",
        "meter-hot": "var(--meter-hot)",
      },
      transitionDuration: {
        fast: "150ms",
        base: "200ms",
        slow: "350ms",
      },
      keyframes: {
        "bubble-in": {
          "0%": { opacity: "0", transform: "translateY(8px) scale(0.97)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        "dot-bounce": {
          "0%, 60%, 100%": { transform: "translateY(0)", opacity: "0.55" },
          "30%": { transform: "translateY(-4px)", opacity: "1" },
        },
        "presence-pulse": {
          "0%": { transform: "scale(1)", opacity: "0.6" },
          "100%": { transform: "scale(1.9)", opacity: "0" },
        },
        "pill-slide": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "send-slide": {
          "0%": { opacity: "0", transform: "translateX(8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        "meter-glow": {
          "0%, 100%": { boxShadow: "0 0 0 0 transparent" },
          "30%": { boxShadow: "0 0 12px 2px var(--meter-glow-color, var(--accent))" },
        },
      },
      animation: {
        "bubble-in": "bubble-in 220ms cubic-bezier(0.2, 0.8, 0.2, 1) both",
        "dot-bounce": "dot-bounce 1000ms ease-in-out infinite",
        "presence-pulse": "presence-pulse 2500ms ease-out infinite",
        "pill-slide": "pill-slide 200ms cubic-bezier(0.2, 0.8, 0.2, 1) both",
        "send-slide": "send-slide 200ms cubic-bezier(0.2, 0.8, 0.2, 1) both",
        "meter-glow": "meter-glow 600ms ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
