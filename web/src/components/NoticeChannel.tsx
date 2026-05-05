import { Email } from "../api";
import { ChannelIcon } from "./Avatar";
import { NoticeCard } from "./NoticeCard";

interface Props {
  emails: Email[];
}

export function NoticeChannel({ emails }: Props) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-border bg-bg px-6 py-3">
        <ChannelIcon size={36} />
        <div className="min-w-0">
          <div className="text-[14px] font-semibold text-text">notices</div>
          <div className="text-[11px] text-text-muted">
            Posts from Continuity Program Operations
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-4">
          {emails.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border px-6 py-12 text-center text-[13px] text-text-muted">
              No notices yet. The Program has nothing to report.
            </div>
          ) : (
            emails.map((email) => <NoticeCard key={email.id} email={email} />)
          )}
        </div>
      </div>
    </div>
  );
}
