import { ReactNode } from "react";
import { AppHeader } from "./AppHeader";
import { ChatSidebar } from "./ChatSidebar";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full flex-col bg-bg-sidebar text-text">
      <AppHeader />
      <div className="flex flex-1 overflow-hidden">
        <ChatSidebar />
        <div className="flex flex-1 overflow-hidden pb-2 pr-2">
          <main className="flex flex-1 flex-col overflow-hidden rounded-xl border border-border/50 bg-bg shadow-[0_2px_8px_rgba(0,0,0,0.06),_0_8px_24px_rgba(0,0,0,0.08)] dark:border-white/5 dark:shadow-[0_2px_8px_rgba(0,0,0,0.4),_0_12px_32px_rgba(0,0,0,0.55)]">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}
