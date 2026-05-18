// The cover page. Title, subtitle, byline, "begin the investigation".
// While starting=true (warm-up), the button is disabled and three
// breathing dots appear after the label.

export function Frontispiece({
  onStart,
  starting,
  error,
}: {
  onStart: () => void;
  starting: boolean;
  error: string | null;
}) {
  return (
    <header className="frontis">
      <div />
      <div>
        <div className="frontis-mark">a folk-noir investigation</div>
        <h1 className="frontis-title">Pant Goch.</h1>
        <div className="frontis-rule" />
        <p className="frontis-sub">
          Six weeks. A village in the Welsh Marches that has held its breath
          since 2019. A letter from a man who places flowers, monthly, at the
          bend where his daughter went off the road. The story is whether
          there is a story.
        </p>
        <p className="frontis-byline">field notebook · m. pritchard · 2026</p>
        {error && <div className="frontis-error">{error}</div>}
      </div>
      <div className="frontis-scroll">
        <button onClick={onStart} disabled={starting} aria-busy={starting}>
          begin the investigation
          {starting && (
            <span className="warming-dots" aria-label="model warming up">
              <span>·</span>
              <span>·</span>
              <span>·</span>
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
