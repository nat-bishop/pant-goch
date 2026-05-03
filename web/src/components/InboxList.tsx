import { useStore } from "../store";
import { Email } from "../api";
import { stripAddress } from "../util/email";

interface Props {
  folder: string;
}

interface ThreadRow {
  head: Email;
  members: Email[];
  anyUnread: boolean;
}

export function InboxList({ folder }: Props) {
  const { folders, openThread } = useStore();
  const emails = folders[folder] ?? [];

  if (emails.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-muted">
        nothing in {folder.toLowerCase()}
      </div>
    );
  }

  const rows = groupByThread(emails);

  const onClick = async (head: Email) => {
    if (!head.thread_id) return;
    await openThread(head.thread_id, folder);
  };

  return (
    <ul className="h-full overflow-y-auto px-2 py-1">
      {rows.map((row) => {
        const head = row.head;
        const unread = row.anyUnread && folder === "Inbox";
        const sender = displaySender(head, folder);
        return (
          <li
            key={head.thread_id ?? head.id}
            onClick={() => onClick(head)}
            className="group flex cursor-pointer items-center gap-3 rounded-sm border-b border-line/60 px-4 py-2.5 hover:bg-paper-hover"
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
                {head.subject}
              </span>
              {head.preview && (
                <span className="text-muted"> — {head.preview}</span>
              )}
            </span>

            <span className="w-28 shrink-0 text-right text-xs text-muted">
              {head.sent_at}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function groupByThread(emails: Email[]): ThreadRow[] {
  const rows: ThreadRow[] = [];
  const byThread = new Map<string, ThreadRow>();
  for (const e of emails) {
    const key = e.thread_id ?? `__solo__:${e.id}`;
    let row = byThread.get(key);
    if (!row) {
      row = { head: e, members: [], anyUnread: false };
      byThread.set(key, row);
      rows.push(row);
    }
    row.members.push(e);
    if (!e.is_read) row.anyUnread = true;
  }
  return rows;
}

function displaySender(email: Email, folder: string): string {
  const raw = folder === "Sent" ? email.to : email.from;
  return stripAddress(raw);
}
