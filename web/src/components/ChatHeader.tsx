import { Avatar } from "./Avatar";
import { PresenceDot } from "./PresenceDot";

interface Props {
  seed: string;
  name: string;
  subtitle: string;
}

export function ChatHeader({ seed, name, subtitle }: Props) {
  return (
    <div className="flex items-center gap-3 border-b border-border bg-bg px-6 py-3">
      <div className="relative">
        <Avatar seed={seed} name={name} size={36} />
        <span className="absolute -bottom-0.5 -right-0.5">
          <PresenceDot size={11} />
        </span>
      </div>
      <div className="min-w-0">
        <div className="truncate text-[14px] font-semibold text-text">
          {name}
        </div>
        <div className="truncate text-[11px] text-text-muted">{subtitle}</div>
      </div>
    </div>
  );
}
