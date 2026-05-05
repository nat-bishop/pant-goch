import { useEffect, useRef } from "react";

const LINE_HEIGHT_PX = 22;
const MAX_LINES = 6;

export function useAutoGrow(value: string) {
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    const max = LINE_HEIGHT_PX * MAX_LINES + 16;
    el.style.height = `${Math.min(el.scrollHeight, max)}px`;
  }, [value]);

  return ref;
}
