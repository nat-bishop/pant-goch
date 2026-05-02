import { useStore } from "../store";
import { stripAddress } from "../util/email";

export function SearchResults() {
  const { view, searchHits, back } = useStore();
  if (view.kind !== "search") return null;

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
          <span className="truncate text-ink">
            search results for{" "}
            <span className="font-bold">"{view.query}"</span>
          </span>
        </div>
        <span className="ml-4 shrink-0 text-[11px] uppercase tracking-wider text-muted">
          {searchHits?.length ?? 0} hit{searchHits?.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-1">
        {!searchHits || searchHits.length === 0 ? (
          <div className="px-6 py-8 text-sm italic text-muted">
            no matches.
          </div>
        ) : (
          <ul>
            {searchHits.map((h) => (
              <li
                key={h.id}
                className="flex items-center gap-3 rounded-sm border-b border-line/60 px-4 py-2.5"
              >
                <span className="w-2 shrink-0" />
                <span className="w-44 shrink-0 truncate text-sm text-ink-soft">
                  {stripAddress(h.from)}
                </span>
                <span className="min-w-0 flex-1 truncate text-sm">
                  <span className="text-ink-soft">{h.subject}</span>
                  {h.snippet && (
                    <span className="text-muted"> — {h.snippet}</span>
                  )}
                </span>
                <span className="flex w-44 shrink-0 items-center justify-end gap-2 text-xs text-muted">
                  <span className="rounded-full border border-line px-2 py-0.5 text-[10px] uppercase tracking-wider">
                    {h.is_archive ? "archive" : h.folder.toLowerCase()}
                  </span>
                  <span className="w-20 text-right">{h.sent_at}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
