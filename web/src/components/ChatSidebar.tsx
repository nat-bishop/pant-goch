import { useState } from "react";
import { useStore } from "../store";
import { Avatar } from "./Avatar";
import { ChatRow } from "./ChatRow";
import { PresenceDot } from "./PresenceDot";

export function ChatSidebar() {
  const view = useStore((s) => s.view);
  const npcs = useStore((s) => s.npcs);
  const openChat = useStore((s) => s.openChat);
  const chatMessages = useStore((s) => s.chatMessages);
  const unreadByThread = useStore(
    (s) => s.state?.unread_counts_by_thread ?? {},
  );
  const [directOpen, setDirectOpen] = useState(true);

  // Hide a row when its thread has no emails AND no unread count yet — this
  // is how the second NPC stays invisible until their first beat fires.
  const visibleNpcs = npcs.filter((n) => {
    const hasEmails = (chatMessages[n.thread_id]?.length ?? 0) > 0;
    const hasUnread = (unreadByThread[n.thread_id] ?? 0) > 0;
    return hasEmails || hasUnread;
  });

  return (
    <aside className="flex w-64 shrink-0 flex-col bg-bg-sidebar pl-2">
      <div className="px-4 py-4">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-text-muted">
          Chat
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 pb-4">
        <Section
          label="Direct"
          open={directOpen}
          onToggle={() => setDirectOpen((o) => !o)}
        >
          {visibleNpcs.map((n) => (
            <ChatRow
              key={n.thread_id}
              active={view.chatId === n.thread_id}
              unread={unreadByThread[n.thread_id] ?? 0}
              onClick={() => openChat(n.thread_id)}
              icon={<Avatar seed={n.email_address} name={n.display_name} size={32} />}
              trailing={
                n.replies_enabled ? (
                  <span className="absolute -bottom-0.5 -right-0.5">
                    <PresenceDot size={10} />
                  </span>
                ) : undefined
              }
              name={n.display_name}
              subtitle={n.replies_enabled ? "Counterpart" : "Notice"}
            />
          ))}
        </Section>
      </nav>
    </aside>
  );
}

function Section({
  label,
  open,
  onToggle,
  children,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-3">
      <button
        onClick={onToggle}
        className="mb-1 flex w-full items-center gap-1 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted hover:text-text"
      >
        <Chevron open={open} />
        {label}
      </button>
      {open && <div className="flex flex-col gap-0.5">{children}</div>}
    </div>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        transform: `rotate(${open ? 0 : -90}deg)`,
        transition: "transform 200ms",
      }}
      aria-hidden
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}
