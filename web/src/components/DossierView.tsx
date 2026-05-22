// DossierView = the body of the left-pane "dossier" overlay: the reporter's
// working file. Two sections behind a segmented sub-nav — "The Record" (the
// prose case file, NotesView) and "The People" (CastBoard, the journalist's
// read on the cast). Reference surfaces grouped under one notebook, rather
// than competing for a top-level tab.

import { useState } from "react";
import { CastBoard } from "./CastBoard";
import { NotesView } from "./NotesView";

type Section = "record" | "people";

export function DossierView() {
  const [section, setSection] = useState<Section>("record");

  return (
    <div className="dossier">
      <div className="dossier-subnav" role="tablist" aria-label="dossier section">
        {(["record", "people"] as const).map((s) => (
          <button
            key={s}
            type="button"
            role="tab"
            aria-selected={section === s}
            className={"dossier-subtab" + (section === s ? " active" : "")}
            onClick={() => setSection(s)}
          >
            {s === "record" ? "the record" : "the people"}
          </button>
        ))}
      </div>
      <div className="dossier-body">
        {section === "record" ? <NotesView /> : <CastBoard />}
      </div>
    </div>
  );
}
