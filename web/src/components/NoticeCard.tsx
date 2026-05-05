import { Email } from "../api";
import { stripAddress } from "../util/email";

interface Props {
  email: Email;
}

export function NoticeCard({ email }: Props) {
  const sender = stripAddress(email.from);

  return (
    <article className="overflow-hidden rounded-lg border border-border bg-bg-elevated shadow-sm animate-bubble-in">
      <header className="flex items-baseline justify-between gap-3 border-b border-border bg-bg-bubble-them px-5 py-2.5">
        <div className="min-w-0">
          <div className="truncate text-[12px] font-medium text-text">
            {sender}
          </div>
          <div className="truncate text-[11px] uppercase tracking-wider text-text-muted">
            {email.subject}
          </div>
        </div>
        <span className="shrink-0 text-[11px] text-text-muted">
          {email.sent_at}
        </span>
      </header>
      <div className="whitespace-pre-line px-5 py-4 font-mono text-[12.5px] leading-[1.6] text-text">
        {email.body}
      </div>
    </article>
  );
}
