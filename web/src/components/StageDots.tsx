interface Props {
  /** 0-based current stage. */
  current: number;
  /** Number of stages (>= 1). */
  total: number;
}

/** Stage progress dots — filled for completed stages, hollow for remaining,
 * connected by short rules between (mockup .stages style). */
export function StageDots({ current, total }: Props) {
  const dots: JSX.Element[] = [];
  for (let i = 0; i < total; i++) {
    dots.push(
      <span
        key={`d-${i}`}
        className={`rounded-full border-[1.5px] border-chrome-bright ${
          i < current ? "bg-chrome-bright" : ""
        }`}
        style={{ width: 10, height: 10 }}
      />,
    );
    if (i < total - 1) {
      dots.push(
        <span
          key={`l-${i}`}
          className="bg-chrome-faint"
          style={{ width: 18, height: 1, margin: "0 4px" }}
        />,
      );
    }
  }
  return <span className="inline-flex items-center gap-0">{dots}</span>;
}
