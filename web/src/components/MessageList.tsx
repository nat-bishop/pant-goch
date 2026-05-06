import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { Email, NpcInfo } from "../api";
import { useScrollAnchor } from "../hooks/useScrollAnchor";
import { useStore } from "../store";
import { MessageGroup } from "./MessageGroup";
import { NewMessagePill } from "./NewMessagePill";
import { TypingIndicator } from "./TypingIndicator";

interface Props {
  chatId: string;
  emails: Email[];
  npc: NpcInfo;
}

export function MessageList({ chatId, emails, npc }: Props) {
  const pendingTypingFor = useStore((s) => s.pendingTypingFor);
  const showTyping = pendingTypingFor === chatId;
  const { ref, isAtBottom, scrollToBottom, recompute } =
    useScrollAnchor<HTMLDivElement>();
  const lastCount = useRef(0);

  const groups = useMemo(() => groupBySender(emails), [emails]);

  // First mount: jump to bottom instantly. Subsequent updates: smooth-scroll
  // only if the user was already pinned to the bottom.
  useLayoutEffect(() => {
    if (lastCount.current === 0 && emails.length > 0) {
      scrollToBottom("auto");
      lastCount.current = emails.length;
      return;
    }
    if (emails.length > lastCount.current && isAtBottom) {
      scrollToBottom("smooth");
    }
    lastCount.current = emails.length;
    recompute();
  }, [emails.length, isAtBottom, scrollToBottom, recompute]);

  useEffect(() => {
    if (showTyping && isAtBottom) scrollToBottom("smooth");
  }, [showTyping, isAtBottom, scrollToBottom]);

  // Reset on chat switch
  useEffect(() => {
    lastCount.current = 0;
  }, [chatId]);

  return (
    <div className="relative flex-1 overflow-hidden">
      <div
        ref={ref}
        aria-live="polite"
        className="h-full overflow-y-auto px-6 py-5"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-3">
          {groups.map((g, i) => (
            <MessageGroup key={`${g[0]!.id}-${i}`} emails={g} />
          ))}
          {showTyping && (
            <TypingIndicator seed={npc.email_address} name={npc.display_name} />
          )}
        </div>
      </div>
      {!isAtBottom && <NewMessagePill onClick={() => scrollToBottom("smooth")} />}
    </div>
  );
}

function groupBySender(emails: Email[]): Email[][] {
  const groups: Email[][] = [];
  for (const e of emails) {
    const last = groups[groups.length - 1];
    if (last && last[0]!.from === e.from) {
      last.push(e);
    } else {
      groups.push([e]);
    }
  }
  return groups;
}
