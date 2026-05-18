// Free-text input — rendered when the choice author emits a FreeText
// variant. The player types their own line; submit hits POST /free_text.
// Mirrors ChoicesBlock's place in the chapter — replaces it for the
// awaiting_free_text phase only.

import { useEffect, useRef, useState } from "react";
import { LeaveButton } from "./LeaveButton";

export function FreeTextInput({
  prompt,
  onSubmit,
  onLeave,
  busy,
}: {
  prompt: string;
  onSubmit: (text: string) => void;
  onLeave: () => void;
  busy: boolean;
}) {
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    onSubmit(trimmed);
  };

  return (
    <form className="freetext-wrap" onSubmit={submit}>
      <div className="choices-eyebrow">— {prompt} —</div>
      <textarea
        ref={inputRef}
        className="freetext-input"
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
      <LeaveButton onLeave={onLeave} busy={busy} />
    </form>
  );
}
