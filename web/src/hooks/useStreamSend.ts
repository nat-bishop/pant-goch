// Shared post-send pipeline for EmailComposer and DraftEditor: drive an SSE
// stream to completion, surface error messages, then refresh the inbox and
// reopen the current thread so the new emails show up.

import { useState } from "react";
import { SseEvent } from "../api";
import { useStore } from "../store";

export function useStreamSend() {
  const { refreshInbox, view, openThread } = useStore();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(stream: AsyncGenerator<SseEvent, void, void>) {
    setBusy(true);
    setError(null);
    try {
      for await (const ev of stream) {
        if (ev.type === "error") setError(ev.message);
        if (ev.type === "done") break;
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
