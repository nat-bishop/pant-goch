import { useStore } from "../store";
import { Email } from "../api";
import { stripAddress } from "../util/email";

interface Props {
  folder: string;
}

export function InboxList({ folder }: Props) {
  const { folders, openThread, openEmail } = useStore();
  const emails = folders[folder] ?? [];

  if (emails.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-muted">
        nothing in {folder.toLowerCase()}
      </div>
    );
  }

  const onClick = async (email: Email) => {
    if (!email.thread_id) return;
    if (folder === "Inbox" && !email.is_read) {
      await openEmail(email.id);
    }
    await openThread(email.thread_id, folder);
  };

  return (
    <ul className="h-full overflow-y-auto px-2 py-1">
      {emails.map((e) => {
        const unread = !e.is_read && folder === "Inbox";
        const isDraft = folder === "Drafts";
        const sender = displaySender(e, folder);
        return (
          <li
            key={e.id}
            onClick={() => onClick(e)}
            className={`group flex cursor-pointer items-center gap-3 rounded-sm border-b border-line/60 px-4 py-2.5 hover:bg-paper-hover ${
              isDraft ? "bg-warm/5" : ""
            }`}
          >
            <span
              className={`h-1.5 w-1.5 shrink-0 rounded-full ${unread ? "bg-warm" : "bg-transparent"}`}
              aria-label={unread ? "unread" : undefined}
            />

            <span
              className={`w-44 shrink-0 truncate text-sm ${
                unread
                  ? "font-bold text-ink"
                  : "text-ink-soft"
              }`}
            >
              {sender}
            </span>

            <span className="min-w-0 flex-1 truncate text-sm">
              <span className={unread ? "font-bold text-ink" : "text-ink-soft"}>
                {e.subject}
              </span>
              {e.preview && (
                <span className="text-muted"> — {e.preview}</span>
              )}
            </span>

            <span className="w-28 shrink-0 text-right text-xs text-muted">
              {e.sent_at}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function displaySender(email: Email, folder: string): string {
  const raw = folder === "Drafts" || folder === "Sent" ? email.to : email.from;
  return stripAddress(raw);
}
