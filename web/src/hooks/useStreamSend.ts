// Drive an SSE stream to completion, surface error messages, then refresh
// the inbox and reopen the current thread so any newly emitted emails show
// up. Used by EmailComposer for /send.

import { useState } from "react";
import { SseEvent } from "../api";
import { useStore } from "../store";

export function useStreamSend() {
  const { refreshInbox, view, openThread, appendEmailOptimistic } = useStore();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(stream: AsyncGenerator<SseEvent, void, void>) {
    setBusy(true);
    setError(null);
    try {
      for await (const ev of stream) {
        if (ev.type === "send_recorded") appendEmailOptimistic(ev.email);
        else if (ev.type === "error") setError(ev.message);
        else if (ev.type === "done") break;
      }
      await refreshInbox();
      if (view.kind === "thread") {
        await openThread(view.threadId, view.folderReturnTo);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return { busy, error, run };
}
