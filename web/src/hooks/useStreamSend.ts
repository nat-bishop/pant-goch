// Drive an SSE stream to completion. Player's send appears instantly
// (send_recorded). Aria's reply is buffered and revealed after a paced
// "typing" delay so the UI feels like she's composing rather than spitting
// out an LLM response. /open and notice-thread emails skip the delay (no
// indicator the user would see anyway).

import { useState } from "react";
import { Email, SseEvent } from "../api";
import { useStore } from "../store";
import { sleep, typingDelayMs } from "../util/pacing";

export function useStreamSend() {
  const {
    refreshInbox,
    view,
    appendEmailOptimistic,
    setPendingTypingFor,
    openChat,
    recordTickDecision,
  } = useStore();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(stream: AsyncGenerator<SseEvent, void, void>) {
    setBusy(true);
    setError(null);
    const queue: Email[] = [];
    const currentChat = view.chatId;
    let typingShown = false;
    if (currentChat === null) {
      // No chat selected — should be impossible while a composer is active
      // (App gates on view.chatId before rendering MessageComposer). Bail.
      setBusy(false);
      return;
    }

    try {
      for await (const ev of stream) {
        if (ev.type === "send_recorded") {
          appendEmailOptimistic(ev.email);
        } else if (ev.type === "new_email") {
          queue.push(ev.email);
          if (!typingShown && ev.email.thread_id === currentChat) {
            setPendingTypingFor(currentChat);
            typingShown = true;
          }
        } else if (ev.type === "tick_decision") {
          // Always record (cheap), the DebugPanel decides whether to render.
          // Production builds tree-shake the panel and ignore this state.
          recordTickDecision(ev.decision);
        } else if (ev.type === "error") {
          setError(ev.message);
        } else if (ev.type === "done") {
          break;
        }
      }

      for (const email of queue) {
        if (email.thread_id === currentChat) {
          setPendingTypingFor(currentChat);
          await sleep(typingDelayMs(email.body ?? ""));
        }
        appendEmailOptimistic(email);
      }
      setPendingTypingFor(null);

      // Refresh GameState (calibration etc.) and canonicalize the open
      // chat — marks newly-arrived NPC emails as read on the backend.
      await Promise.all([refreshInbox(), openChat(currentChat)]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPendingTypingFor(null);
    } finally {
      setBusy(false);
    }
  }

  return { busy, error, run };
}
