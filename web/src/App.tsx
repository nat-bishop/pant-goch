import { useEffect } from "react";
import { Email } from "./api";
import { AppShell } from "./components/AppShell";
import { ChatHeader } from "./components/ChatHeader";
import { MessageComposer } from "./components/MessageComposer";
import { MessageList } from "./components/MessageList";
import { useTheme } from "./hooks/useTheme";
import { useStore } from "./store";
import { stripAddress } from "./util/email";
import { ARIA_DISPLAY_NAME, ARIA_SEED } from "./util/npc";

export default function App() {
  const boot = useStore((s) => s.boot);
  const bootError = useStore((s) => s.bootError);
  const scenario = useStore((s) => s.scenario);
  const state = useStore((s) => s.state);
  const chatMessages = useStore((s) => s.chatMessages);

  useTheme();

  useEffect(() => {
    boot();
  }, [boot]);

  if (bootError) {
    return (
      <div className="flex h-full items-center justify-center bg-bg text-sm text-text-muted">
        sidecar unreachable: {bootError}
      </div>
    );
  }

  if (!scenario || !state) {
    return (
      <div className="flex h-full items-center justify-center bg-bg text-sm text-text-muted">
        …
      </div>
    );
  }

  const emails = chatMessages.aria ?? [];

  return (
    <AppShell>
      <AriaChat emails={emails} />
    </AppShell>
  );
}

function AriaChat({ emails }: { emails: Email[] }) {
  const target = pickComposeTarget(emails);
  return (
    <>
      <ChatHeader
        seed={ARIA_SEED}
        name={ARIA_DISPLAY_NAME}
        subtitle="Counterpart · Active"
      />
      <MessageList chatId="aria" emails={emails} />
      {target && (
        <MessageComposer
          inReplyTo={target.id}
          recipient={target.from}
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
