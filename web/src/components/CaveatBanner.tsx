interface Props {
  /** Trailing handling caveat — defaults to "UK EYES ONLY". */
  tail?: string;
}

export function CaveatBanner({ tail = "UK EYES ONLY" }: Props) {
  return (
    <div className="paper-caveat">
      <span className="paper-caveat-rule" />
      <span className="paper-caveat-text">
        TOP SECRET <span className="paper-caveat-sep">//</span> {tail}
      </span>
      <span className="paper-caveat-rule" />
    </div>
  );
}
