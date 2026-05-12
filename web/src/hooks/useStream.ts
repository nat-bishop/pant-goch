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

    // Defer the fetch past React's commit phase so StrictMode's dev-only
    // discard cleanup (which fires synchronously after the first commit)
    // cancels us before we open the stream. Two concurrent consumers on
    // the same session race against the server's single-consumer SSE
    // queue — each event goes to exactly one consumer — which interleaves
    // stale-read writes into the shared Zustand store and corrupts cycle
    // state. queueMicrotask isn't enough here: React's commit yields to
    // microtasks between setup1 and cleanup1 in dev. setTimeout(0) clears
    // the entire commit boundary so cancelled is reliably true by the
    // time the discarded effect's deferred body runs.
    const deferred = setTimeout(() => {
      if (cancelled) return;
      void (async () => {
        try {
          const response = await fetch(streamUrl(session_id), {
            signal: controller.signal,
          });
          if (!response.ok) {
            console.warn("SSE stream returned", response.status);
            return;
          }
          const sink = window as unknown as { __SSE_LOG__?: unknown[] };
          if (import.meta.env.DEV) sink.__SSE_LOG__ = [];
          for await (const ev of readSseStream(response)) {
            if (cancelled) break;
            if (import.meta.env.DEV) sink.__SSE_LOG__?.push(ev);
            handleEvent(ev);
          }
        } catch (e) {
          if (controller.signal.aborted) return;
          console.warn("SSE stream errored", e);
        }
      })();
    }, 0);

    return () => {
      cancelled = true;
      clearTimeout(deferred);
      controller.abort();
    };
  }, [session_id, handleEvent]);
}
