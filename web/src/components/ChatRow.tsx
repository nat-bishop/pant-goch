import clsx from "clsx";
import { ReactNode } from "react";

interface Props {
  active: boolean;
  unread: number;
  onClick: () => void;
  icon: ReactNode;
  name: string;
  subtitle?: string;
  trailing?: ReactNode;
}

export function ChatRow({
  active,
  unread,
  onClick,
  icon,
  name,
  subtitle,
  trailing,
}: Props) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "group relative flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left transition-colors duration-fast",
        active
          ? "bg-bg-elevated"
          : "hover:bg-bg-bubble-them",
      )}
    >
      <span
        className={clsx(
          "absolute left-0 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-r-full bg-accent transition-opacity duration-base",
          active ? "opacity-100" : "opacity-0",
        )}
        aria-hidden
      />
      <div className="relative">
        {icon}
        {trailing}
      </div>
      <div className="min-w-0 flex-1">
        <div
          className={clsx(
            "truncate text-[13px]",
            unread > 0 ? "font-semibold text-text" : "font-medium text-text",
          )}
        >
          {name}
        </div>
        {subtitle && (
          <div className="truncate text-[11px] text-text-muted">{subtitle}</div>
        )}
      </div>
      {unread > 0 && (
        <span className="ml-1 flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-unread px-1.5 text-[10px] font-semibold leading-none text-white">
          {unread}
        </span>
      )}
    </button>
  );
}
