interface Props {
  onClick: () => void;
}

export function NewMessagePill({ onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full bg-accent px-4 py-1.5 text-xs font-medium text-text-on-accent shadow-lg shadow-accent-glow animate-pill-slide hover:bg-accent-hover"
    >
      <ArrowDown />
      New message
    </button>
  );
}

function ArrowDown() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 5v14M19 12l-7 7-7-7" />
    </svg>
  );
}
