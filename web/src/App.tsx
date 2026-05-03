import { useEffect } from "react";
import { useStore } from "./store";
import { InboxHeader } from "./components/InboxHeader";
import { FolderTabs } from "./components/FolderTabs";
import { InboxList } from "./components/InboxList";
import { ThreadView } from "./components/ThreadView";

export default function App() {
  const { boot, bootError, scenario, state, view } = useStore();

  useEffect(() => {
    boot();
  }, [boot]);

  if (bootError) {
    return (
      <div className="flex h-full items-center justify-center text-muted">
        sidecar unreachable: {bootError}
      </div>
    );
  }

  if (!scenario || !state) {
    return <div className="flex h-full items-center justify-center text-muted">…</div>;
  }

  return (
    <div className="flex h-full flex-col">
      <InboxHeader />
      <div className="flex flex-1 overflow-hidden">
        <FolderTabs />
        <main className="flex-1 overflow-hidden">
          {view.kind === "folder" && <InboxList folder={view.folder} />}
          {view.kind === "thread" && <ThreadView />}
        </main>
      </div>
    </div>
  );
}
