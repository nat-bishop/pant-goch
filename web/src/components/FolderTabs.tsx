import { useStore } from "../store";

const FOLDERS = ["Inbox", "Sent", "Archive"] as const;

export function FolderTabs() {
  const { view, state, goFolder } = useStore();
  const activeFolder = view.kind === "folder" ? view.folder : null;

  return (
    <nav className="flex w-64 shrink-0 flex-col border-r border-line bg-paper py-4">
      <ul className="px-2">
        {FOLDERS.map((f) => {
          const unread = state?.unread_counts?.[f] ?? 0;
          const active = activeFolder === f;
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
                  <span className="rounded-full bg-ink px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-paper">
                    {unread}
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
