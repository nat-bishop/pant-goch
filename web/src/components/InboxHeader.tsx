import { useStore } from "../store";

export function InboxHeader() {
  const { scenario, state } = useStore();

  return (
    <header className="flex items-center gap-6 border-b border-line bg-paper px-6 py-3">
      <div className="shrink-0 text-base font-bold tracking-tight">
        {scenario?.title}
      </div>

      <div className="flex-1" />

      <div className="flex shrink-0 items-center gap-3">
        {state?.ended && (
          <span className="rounded-full border border-warm/40 bg-warm/5 px-3 py-1 text-[11px] tracking-wide text-warm">
            transition complete
          </span>
        )}
      </div>
    </header>
  );
}
