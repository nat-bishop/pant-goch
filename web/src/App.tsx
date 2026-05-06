import { useEffect } from "react";
import { Email, NpcInfo } from "./api";
import { AppHeader } from "./components/AppHeader";
import { ChatHeader } from "./components/ChatHeader";
import { ChatSidebar } from "./components/ChatSidebar";
import { DebugPanel } from "./components/DebugPanel";
import { MessageComposer } from "./components/MessageComposer";
import { MessageList } from "./components/MessageList";
import { useTheme } from "./hooks/useTheme";
import { useStore } from "./store";
import { stripAddress } from "./util/email";

export default function App() {
  const boot = useStore((s) => s.boot);
  const bootError = useStore((s) => s.bootError);
  const scenario = useStore((s) => s.scenario);
  const npcs = useStore((s) => s.npcs);
  const state = useStore((s) => s.state);
  const view = useStore((s) => s.view);
  const chatMessages = useStore((s) => s.chatMessages);

  useTheme();

  useEffect(() => {
    boot();
  }, [boot]);

  if (bootError) return <Centered>sidecar unreachable: {bootError}</Centered>;
  if (!scenario || !state || !view.chatId) return <Centered>…</Centered>;

  const selectedNpc = npcs.find((n) => n.thread_id === view.chatId);
  if (!selectedNpc) return <Centered>…</Centered>;

  const emails = chatMessages[view.chatId] ?? [];
  return (
    <div className="flex h-full flex-col bg-bg-sidebar text-text">
      <AppHeader />
      <div className="flex flex-1 overflow-hidden">
        <ChatSidebar />
        <div className="flex flex-1 overflow-hidden pb-2 pr-2">
          <main className="flex flex-1 flex-col overflow-hidden rounded-xl border border-border/50 bg-bg shadow-[0_2px_8px_rgba(0,0,0,0.06),_0_8px_24px_rgba(0,0,0,0.08)] dark:border-white/5 dark:shadow-[0_2px_8px_rgba(0,0,0,0.4),_0_12px_32px_rgba(0,0,0,0.55)]">
            <NpcChat npc={selectedNpc} emails={emails} />
          </main>
        </div>
      </div>
      {/* DEV-only overlay surfacing per-tick decisions. Vite's
          import.meta.env.DEV is a static literal in production builds, so
          the entire DebugPanel module tree-shakes out of the prod bundle. */}
      {import.meta.env.DEV && <DebugPanel />}
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center bg-bg text-sm text-text-muted">
      {children}
    </div>
  );
}

function NpcChat({ npc, emails }: { npc: NpcInfo; emails: Email[] }) {
  const target = pickComposeTarget(emails);
  const subtitle = npc.replies_enabled ? "Counterpart · Active" : "Notice";
  return (
    <>
      <ChatHeader
        seed={npc.email_address}
        name={npc.display_name}
        subtitle={subtitle}
      />
      <MessageList chatId={npc.thread_id} emails={emails} npc={npc} />
      {npc.replies_enabled && target && (
        <MessageComposer
          inReplyTo={target.id}
          recipient={target.from}
          recipientNpcId={npc.id}
          recipientName={stripAddress(target.from)}
        />
      )}
    </>
  );
}

function pickComposeTarget(emails: Email[]): Email | null {
  for (let i = emails.length - 1; i >= 0; i--) {
    const e = emails[i]!;
    if (e.from !== "you") return e;
  }
  return null;
}
