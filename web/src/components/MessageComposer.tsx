import { useEffect, useState } from "react";
import { streamSend } from "../api";
import { useAutoGrow } from "../hooks/useAutoGrow";
import { useStreamSend } from "../hooks/useStreamSend";
import { useStore } from "../store";

interface Props {
  inReplyTo: string | null;
  recipient: string;
  recipientName: string;
}

export function MessageComposer({
  inReplyTo,
  recipient,
  recipientName,
}: Props) {
  const [body, setBody] = useState("");
  const { busy, error, run } = useStreamSend();
  const pendingTypingFor = useStore((s) => s.pendingTypingFor);
  const currentChatId = useStore((s) => s.view.chatId);
  const ref = useAutoGrow(body);
  const [focused, setFocused] = useState(false);

  const trimmed = body.trim();
  const blocked = busy || pendingTypingFor === currentChatId;
  const canSend = !blocked && trimmed.length > 0;

  useEffect(() => {
    if (!blocked) ref.current?.focus();
  }, [blocked, ref]);

  const onSend = async () => {
    if (!canSend) return;
    setBody("");
    await run(streamSend({ in_reply_to: inReplyTo, recipient, body: trimmed }));
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="border-t border-border bg-bg px-4 py-3">
      <div className="mx-auto max-w-3xl">
        <div
          className="flex items-end gap-2 rounded-xl border bg-bg-elevated px-3 py-2 transition-shadow duration-fast"
          style={{
            borderColor: focused ? "var(--accent)" : "var(--border)",
            boxShadow: focused ? "0 0 0 3px var(--accent-glow)" : "none",
          }}
        >
          <textarea
            ref={ref}
            value={body}
            onChange={(e) => setBody(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={onKeyDown}
            placeholder={`Reply to ${recipientName}…`}
            rows={1}
            disabled={blocked}
            aria-label={`Reply to ${recipientName}`}
            className="flex-1 resize-none bg-transparent text-[14px] leading-[22px] text-text placeholder:text-text-muted focus:outline-none disabled:opacity-50"
            style={{ maxHeight: 22 * 6 + 16 }}
          />
          {trimmed.length > 0 && (
            <button
              onClick={onSend}
              disabled={!canSend}
              aria-label="Send"
              title="Send (Enter)"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent text-text-on-accent transition-all duration-fast hover:bg-accent-hover disabled:opacity-50 animate-send-slide"
            >
              <SendIcon />
            </button>
          )}
        </div>
        {error && (
          <div className="mt-2 text-xs text-unread" role="alert">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

function SendIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}
