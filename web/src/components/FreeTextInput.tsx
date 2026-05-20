// Free-text input — rendered when the choice author emits a FreeText
// variant. The player types their own line; submit hits POST /free_text.
// Mirrors ChoicesBlock's place in the chapter — replaces it for the
// awaiting_free_text phase only.

import { useEffect, useRef, useState } from "react";

export function FreeTextInput({
  prompt,
  suggested,
  onSubmit,
  busy,
}: {
  prompt: string;
  suggested: string | null;
  onSubmit: (text: string) => void;
  busy: boolean;
}) {
  // The suggested line is pre-loaded as an editable backup: the player can
  // submit it as-is, reword it, or clear it. It renders muted while still
  // verbatim, signalling "a suggestion, not yet your words."
  const [text, setText] = useState(suggested ?? "");
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.focus();
    // Select the suggestion so a single keystroke replaces it wholesale,
    // while Enter still submits it verbatim.
    if (suggested) el.select();
  }, [suggested]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    onSubmit(trimmed);
  };

  const showMuted = !!suggested && text === suggested;

  return (
    <form className="freetext-wrap" onSubmit={submit}>
      <div className="choices-eyebrow">— {prompt} —</div>
      <textarea
        ref={inputRef}
        className={`freetext-input${showMuted ? " suggested" : ""}`}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit(e as unknown as React.FormEvent);
          }
        }}
        disabled={busy}
        rows={2}
        placeholder="(Enter to submit; Shift+Enter for newline.)"
      />
    </form>
  );
}
