import { useStore } from "../store";

const FOLDERS = ["Inbox", "Drafts", "Sent", "Archive"] as const;

export function FolderTabs() {
  const { view, state, goFolder } = useStore();
  const activeFolder = view.kind === "folder" ? view.folder : null;

  return (
    <nav className="flex w-64 shrink-0 flex-col border-r border-line bg-paper py-4">
      <ul className="px-2">
        {FOLDERS.map((f) => {
          const unread = state?.unread_counts?.[f] ?? 0;
          const active = activeFolder === f;
          const isDraftsBadge = f === "Drafts" && unread > 0;
          return (
            <li key={f} className="mb-0.5">
              <button
                onClick={() => goFolder(f)}
                className={`group flex w-full items-center justify-between rounded-r-full py-1.5 pl-6 pr-4 text-left text-sm transition-colors ${
                  active
                    ? "bg-paper-hover font-bold text-ink"
                    : "text-muted hover:bg-paper-hover/60 hover:text-ink-soft"
                }`}
              >
                <span className="flex items-center gap-3">
                  {active && <span className="-ml-3 h-4 w-0.5 rounded bg-ink" />}
                  <span>{f}</span>
                </span>
                {unread > 0 && (
                  <span
                    className={`rounded-full px-1.5 py-0.5 text-[10px] font-bold tracking-wide ${
                      isDraftsBadge
                        ? "bg-warm text-paper"
                        : "bg-ink text-paper"
                    }`}
                  >
                    {unread}
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="mx-6 mt-6 border-t border-line pt-4 text-[11px] text-muted">
        <div className="font-bold tracking-wide text-ink-soft">Tess</div>
        <div>on a 72-hour hold</div>
      </div>
    </nav>
  );
}
