import { useState } from "react";
import { DraftAction, Email, streamDraftDecision } from "../api";
import { useStreamSend } from "../hooks/useStreamSend";
import { stripAddress } from "../util/email";

interface Props {
  draft: Email;
}

/**
 * Tess's unsent draft to Maya. Three buttons, styled equally — holding must
 * read as a real, respectful option, not a cancel affordance. The draft text
 * is anchored prose (handed down from scenario.yaml) and is shown verbatim;
 * the editor only reveals when the player chooses to edit.
 */
export function DraftEditor({ draft }: Props) {
  const [editing, setEditing] = useState(false);
  const [edited, setEdited] = useState(draft.body || "");
  const { busy, error, run } = useStreamSend();

  const decide = async (action: DraftAction) => {
    if (busy) return;
    const body = action === "edit_and_send" ? edited : null;
    await run(streamDraftDecision(draft.id, action, body));
  };

  const recipient = stripAddress(draft.to);

  return (
    <article className="mb-6 rounded border border-warm/40 bg-warm/5 p-6 shadow-sm">
      <header className="mb-4 border-b border-warm/30 pb-3 text-xs">
        <div className="font-bold uppercase tracking-[0.18em] text-warm">
          Unsent draft
        </div>
        <div className="mt-2 flex items-center gap-2">
          <span className="text-muted">from:</span>
          <span className="rounded-full bg-paper-hover px-2 py-0.5 text-ink">
            Tess
          </span>
          <span className="text-muted">to:</span>
          <span className="rounded-full bg-paper-hover px-2 py-0.5 text-ink">
            {recipient}
          </span>
        </div>
      </header>

      {!editing ? (
        <div className="prose-email text-ink">{draft.body}</div>
      ) : (
        <textarea
          value={edited}
          onChange={(e) => setEdited(e.target.value)}
          rows={Math.max(8, edited.split("\n").length + 1)}
          className="w-full resize-y rounded border border-line bg-paper px-3 py-2 font-serif text-base leading-relaxed focus:border-ink focus:outline-none"
          disabled={busy}
        />
      )}

      <div className="mt-6 grid grid-cols-3 gap-3 text-xs">
        {!editing ? (
          <>
            <DecisionCard
              label="send unchanged"
              hint="ship Tess's draft as written"
              onClick={() => decide("send_unchanged")}
              disabled={busy}
            />
            <DecisionCard
              label="edit and send"
              hint="make changes before sending"
              onClick={() => setEditing(true)}
              disabled={busy}
            />
            <DecisionCard
              label="save draft and close"
              hint="leave it for Tess to handle later"
              onClick={() => decide("hold")}
              disabled={busy}
            />
          </>
        ) : (
          <>
            <DecisionCard
              label={busy ? "sending…" : "send the edit"}
              hint="send the version you wrote"
              onClick={() => decide("edit_and_send")}
              disabled={busy}
            />
            <DecisionCard
              label="cancel edit"
              hint="go back to the three options"
              onClick={() => {
                setEditing(false);
                setEdited(draft.body || "");
              }}
              disabled={busy}
            />
            <span />
          </>
        )}
      </div>
      {error && <div className="mt-3 text-xs text-warm">{error}</div>}
    </article>
  );
}

function DecisionCard({
  label,
  hint,
  onClick,
  disabled,
}: {
  label: string;
  hint: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex h-full flex-col items-start gap-1 rounded border border-line bg-paper px-3 py-3 text-left text-xs hover:border-ink disabled:opacity-50"
    >
      <span className="font-bold tracking-wide text-ink">{label}</span>
      <span className="text-muted">{hint}</span>
    </button>
  );
}
