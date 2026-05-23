import { useEffect } from "react";
import { readSseStream, streamUrl } from "../api";
import { useStore } from "../store";

/** Open the SSE stream when the user signals "start" by transitioning out
 * of phase=idle. Dispatches every incoming event into the store. Returns
 * nothing — purely a side effect.
 *
 * Backend already paces word emissions >=150 ms so the renderer can naively
 * append events as they arrive. The hook stays simple.
 *
 * Lifted from v1 web/src/hooks/useStream.ts:11-61. The 14-line comment
 * below explaining the `setTimeout(0)` deferred-fetch trick is the
 * load-bearing part — without it, React StrictMode dev-mode
 * double-mount races two SSE consumers on the server's single-consumer
 * queue, interleaving stale-read writes into the shared Zustand store.
 */
export function useStream(active: boolean): void {
  const handleEvent = useStore((s) => s.handleEvent);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let cancelled = false;

    // A full page unload (refresh / navigate away) tears down the in-flight
    // fetch from under us, surfacing a network-error TypeError in the loop
    // below. React's cleanup aborts the controller, but not before the
    // browser kills the connection — so without this the catch can't tell an
    // intentional unload from a real mid-session drop and logs a spurious
    // warning. Aborting on pagehide sets `signal.aborted` first, so the
    // existing guard swallows it; a genuine network failure still warns.
    const onPageHide = () => controller.abort();
    window.addEventListener("pagehide", onPageHide);

    // Defer the fetch past React's commit phase so StrictMode's dev-only
    // discard cleanup (which fires synchronously after the first commit)
    // cancels us before we open the stream. Two concurrent consumers on
    // the same session race against the server's single-consumer SSE
    // queue — each event goes to exactly one consumer — which interleaves
    // stale-read writes into the shared Zustand store and corrupts day
    // state. queueMicrotask isn't enough here: React's commit yields to
    // microtasks between setup1 and cleanup1 in dev. setTimeout(0) clears
    // the entire commit boundary so cancelled is reliably true by the
    // time the discarded effect's deferred body runs.
    const deferred = setTimeout(() => {
      if (cancelled) return;
      void (async () => {
        try {
          const response = await fetch(streamUrl(), {
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
      window.removeEventListener("pagehide", onPageHide);
      controller.abort();
    };
  }, [active, handleEvent]);
}
