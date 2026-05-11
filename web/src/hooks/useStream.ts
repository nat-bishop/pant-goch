import { useEffect } from "react";
import { readSseStream, streamUrl } from "../api";
import { useStore } from "../store";

/** Open the SSE stream as soon as a session id is available, dispatch every
 * incoming event into the store. Returns nothing — purely a side effect.
 *
 * Backend already paces word emissions ≥80ms so the renderer can naively
 * append events as they arrive (HANDLER_SPEC_V2 §11). The hook stays simple.
 */
export function useStream(): void {
  const session_id = useStore((s) => s.session_id);
  const handleEvent = useStore((s) => s.handleEvent);

  useEffect(() => {
    if (!session_id) return;
    const controller = new AbortController();
    let cancelled = false;

    (async () => {
      try {
        const response = await fetch(streamUrl(session_id), {
          signal: controller.signal,
        });
        if (!response.ok) {
          console.warn("SSE stream returned", response.status);
          return;
        }
        for await (const ev of readSseStream(response)) {
          if (cancelled) break;
          handleEvent(ev);
        }
      } catch (e) {
        if (controller.signal.aborted) return;
        console.warn("SSE stream errored", e);
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [session_id, handleEvent]);
}
