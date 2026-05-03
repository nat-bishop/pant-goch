import { useStore } from "../store";
import { EmailComposer } from "./EmailComposer";
import { Email } from "../api";
import { stripAddress } from "../util/email";

export function ThreadView() {
  const { threadEmails, view, back, archiveEmail } = useStore();

  if (view.kind !== "thread" || !threadEmails) return null;

  const composeTarget = pickComposeTarget(threadEmails);

  const subject = threadEmails[0]?.subject ?? view.threadId;
  const folderTag = view.folderReturnTo.toLowerCase();

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line bg-paper px-6 py-3 text-sm">
        <div className="flex items-center gap-3 truncate">
          <button
            onClick={back}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted hover:bg-paper-hover hover:text-ink"
            aria-label="back"
            title="back"
          >
            ←
          </button>
          <span className="truncate font-bold text-ink">{subject}</span>
        </div>
        <span className="ml-4 shrink-0 text-[11px] uppercase tracking-wider text-muted">
          {folderTag}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="mx-auto max-w-3xl">
          {threadEmails.map((email) => (
            <EmailCard
              key={email.id}
              email={email}
              onArchive={() => archiveEmail(email.id)}
            />
          ))}

          {composeTarget && (
            <EmailComposer
              inReplyTo={composeTarget.beat_id ?? null}
              recipient={composeTarget.from}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function EmailCard({
  email,
  onArchive,
}: {
  email: Email;
  onArchive: () => Promise<void>;
}) {
  const sentByPlayer = email.from === "you";
  const senderName = sentByPlayer ? "you" : stripAddress(email.from);
  const recipientName = stripAddress(email.to);

  return (
    <article className="mb-6 rounded border border-line bg-paper p-6 shadow-sm">
      <header className="mb-4 flex items-start justify-between gap-4 border-b border-line pb-3">
        <div className="min-w-0 text-xs">
          <div className="truncate text-sm font-bold text-ink">{senderName}</div>
          <div className="mt-0.5 truncate text-muted">
            to {sentByPlayer ? recipientName : "you"}
          </div>
        </div>
        <span className="shrink-0 text-xs text-muted">{email.sent_at}</span>
      </header>

      <div className="prose-email text-ink">{email.body}</div>

      {email.folder === "Inbox" && !sentByPlayer && (
        <div className="mt-5 flex gap-3 border-t border-line pt-3 text-xs">
          <button
            onClick={onArchive}
            className="text-muted hover:text-ink"
          >
            archive
          </button>
        </div>
      )}
    </article>
  );
}

function pickComposeTarget(emails: Email[]): Email | null {
  for (let i = emails.length - 1; i >= 0; i--) {
    const e = emails[i];
    if (e.folder === "Inbox" && e.from !== "you" && e.beat_id) {
      return e;
    }
  }
  return null;
}
