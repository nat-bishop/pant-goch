// NotesView: case-file entries laid out as note cards in the left
// overlay. Excerpt is the first sentence of the entry's prose.
//
// Clicking a note switches the world mode back to LOCATION and scrolls
// the right panel to that chapter.

import { useStore } from "../store";

function firstSentence(words: string[]): string {
  const text = words.join("").trim();
  if (!text) return "";
  const m = text.match(/^.+?[.!?](?:\s|$)/);
  return (m ? m[0] : text.slice(0, 180)).trim();
}

export function NotesView() {
  const case_file = useStore((s) => s.case_file);
  const scenes = useStore((s) => s.scenes);
  const setWorldMode = useStore((s) => s.setWorldMode);

  if (case_file.length === 0) {
    return (
      <div className="notes-empty">
        The case file is empty. The investigation has not begun.
      </div>
    );
  }

  const onJump = (sceneId: number) => {
    setWorldMode("location");
    setTimeout(() => {
      const el = document.getElementById(`ch-${sceneId}`);
      el?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  };

  return (
    <div className="notes-list">
      {case_file.map((entry, i) => {
        const scene = scenes.find((s) => s.scene_id === entry.scene_id);
        return (
          <button
            type="button"
            className="note"
            key={entry.scene_id}
            onClick={() => onJump(entry.scene_id)}
          >
            <div className="note-meta">
              <span>Entry {String(i + 1).padStart(2, "0")}</span>
              <span>Scene {entry.scene_id}</span>
            </div>
            <h3 className="note-title">{scene?.location_name ?? "—"}</h3>
            <p className="note-excerpt">{firstSentence(entry.words)}</p>
            <div className="note-link">↳ scroll to entry</div>
          </button>
        );
      })}
    </div>
  );
}
