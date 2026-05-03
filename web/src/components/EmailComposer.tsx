import { useState } from "react";
import { streamSend } from "../api";
import { useStreamSend } from "../hooks/useStreamSend";
import { stripAddress } from "../util/email";

interface Props {
  inReplyTo: string | null;
  recipient: string;
}

export function EmailComposer({ inReplyTo, recipient }: Props) {
  const [body, setBody] = useState("");
  const { busy, error, run } = useStreamSend();

  const trimmed = body.trim();
  const canSend = !busy && trimmed.length > 0;

  const onSend = async () => {
    if (!canSend) return;
    await run(streamSend({ in_reply_to: inReplyTo, recipient, body: trimmed }));
    setBody("");
  };

  const recipientName = stripAddress(recipient);

  return (
    <div className="rounded border border-line bg-paper p-5 shadow-sm">
      <div className="mb-3 flex items-center gap-2 border-b border-line pb-2 text-xs">
        <span className="text-muted">to:</span>
        <span className="rounded-full bg-paper-hover px-2 py-0.5 text-ink">
          {recipientName}
        </span>
      </div>
      <textarea
        value={body}
        onChange={(e) => setBody(e.target.value)}
        placeholder="Type your reply…"
        rows={6}
        className="w-full resize-y rounded border border-line bg-paper-hover/40 px-3 py-2 text-base leading-relaxed placeholder-muted focus:border-ink focus:outline-none"
        disabled={busy}
      />
      <div className="mt-3 flex items-center justify-end">
        <button
          onClick={onSend}
          disabled={!canSend}
          className="rounded border border-warm bg-warm px-5 py-1.5 text-xs font-bold tracking-wide text-paper hover:bg-warm/90 disabled:opacity-50"
        >
          {busy ? "sending…" : "send"}
        </button>
      </div>
      {error && <div className="mt-2 text-xs text-warm">{error}</div>}
    </div>
  );
}
