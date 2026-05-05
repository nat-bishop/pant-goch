import { useCallback, useEffect, useRef, useState } from "react";

const STICKY_THRESHOLD_PX = 80;

export function useScrollAnchor<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const update = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setIsAtBottom(distance <= STICKY_THRESHOLD_PX);
  }, []);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    const el = ref.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
    setIsAtBottom(true);
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    update();
    el.addEventListener("scroll", update, { passive: true });
    return () => el.removeEventListener("scroll", update);
  }, [update]);

  return { ref, isAtBottom, scrollToBottom, recompute: update };
}
