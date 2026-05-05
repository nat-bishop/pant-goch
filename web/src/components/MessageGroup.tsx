import clsx from "clsx";
import { Email } from "../api";
import { stripAddress } from "../util/email";
import { Avatar } from "./Avatar";
import { BubblePosition, MessageBubble } from "./MessageBubble";

interface Props {
  emails: Email[];
}

export function MessageGroup({ emails }: Props) {
  if (emails.length === 0) return null;
  const own = emails[0]!.from === "you";
  const senderName = stripAddress(emails[0]!.from);
  const senderSeed = emails[0]!.from;
  const lastSentAt = emails[emails.length - 1]!.sent_at;

  return (
    <div
      className={clsx(
        "flex w-full gap-2",
        own ? "justify-end" : "items-end justify-start",
      )}
    >
      {!own && (
        <div className="w-7 shrink-0">
          <Avatar seed={senderSeed} name={senderName} size={28} />
        </div>
      )}
      <div
        className={clsx(
          "flex min-w-0 flex-1 flex-col gap-[2px]",
          own ? "items-end" : "items-start",
        )}
      >
        {emails.map((email, i) => {
          const position = positionFor(i, emails.length);
          const eyebrow = email.beat_id ? email.subject : undefined;
          return (
            <MessageBubble
              key={email.id}
              body={email.body ?? email.preview}
              variant={own ? "me" : "them"}
              position={position}
              eyebrow={eyebrow}
            />
          );
        })}
        <div
          className={clsx(
            "mt-1 px-1 text-[11px] text-text-muted",
            own ? "self-end" : "self-start",
          )}
        >
          {lastSentAt}
          {own && <span className="ml-1.5 text-text-muted/70">✓</span>}
        </div>
      </div>
    </div>
  );
}

function positionFor(i: number, n: number): BubblePosition {
  if (n === 1) return "single";
  if (i === 0) return "first";
  if (i === n - 1) return "last";
  return "middle";
}
