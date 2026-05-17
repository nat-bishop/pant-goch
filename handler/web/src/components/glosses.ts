// Gloss matcher: builds an alias regex from /world.glosses and resolves
// matches in arriving prose. The frontend renders the prose word-by-word
// as events arrive; this utility turns a paragraph of joined text into a
// list of `{kind: 'text' | 'gloss', ...}` segments for the renderer.
//
// Two responsibilities:
//   1. `buildGlossIndex(glosses)` → memoisable index keyed by lowercased
//      phrase (canonical name + aliases). Returned alongside a single
//      regex with alternation sorted longest-first (so multi-word phrases
//      win over their substring siblings).
//   2. `segmentText(text, index)` → segments. Each match becomes one
//      gloss segment; everything between matches stays as plain text.

import type { GlossDef } from "../api";

export interface GlossIndex {
  /** Canonical glosses by id, for popup lookup. */
  byId: Record<string, GlossDef>;
  /** Lowercased phrase → gloss id. */
  phraseToId: Record<string, string>;
  /** Single regex combining all phrases (case-insensitive, word-bounded). */
  regex: RegExp | null;
}

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "gloss"; text: string; glossId: string };

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function buildGlossIndex(glosses: GlossDef[]): GlossIndex {
  const byId: Record<string, GlossDef> = {};
  const phraseToId: Record<string, string> = {};
  const phrases: string[] = [];
  for (const g of glosses) {
    byId[g.id] = g;
    const variants = [g.name, ...g.aliases];
    for (const v of variants) {
      const lower = v.toLowerCase();
      if (phraseToId[lower]) continue; // first id wins on collision
      phraseToId[lower] = g.id;
      phrases.push(v);
    }
  }
  if (phrases.length === 0) return { byId, phraseToId, regex: null };
  // Longest-first so "Old Tom Bevan" beats "Tom Bevan" beats "Tom".
  phrases.sort((a, b) => b.length - a.length);
  // Word boundaries on both sides — avoid matching "Iwan" inside "Iwanwhatever".
  const pattern = `\\b(?:${phrases.map(escapeRegExp).join("|")})\\b`;
  return { byId, phraseToId, regex: new RegExp(pattern, "gi") };
}

/** Split `text` into an ordered list of segments. Matches are looked up
 * case-insensitively but the matched substring is preserved verbatim so
 * "iwan" in mid-sentence renders as "iwan" (the popup heading still
 * uses the canonical `name`). */
export function segmentText(text: string, index: GlossIndex): Segment[] {
  if (!index.regex || !text) return text ? [{ kind: "text", text }] : [];
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(index.regex)) {
    const matched = m[0];
    if (!matched) continue;
    const start = m.index ?? 0;
    if (start > last) out.push({ kind: "text", text: text.slice(last, start) });
    const id = index.phraseToId[matched.toLowerCase()];
    out.push(
      id ? { kind: "gloss", text: matched, glossId: id } : { kind: "text", text: matched },
    );
    last = start + matched.length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

/** Split a flat word stream (each word may carry trailing whitespace
 * including `\n\n`) into paragraph buckets. A paragraph ends after a
 * word whose trailing whitespace contains a blank line (`\n\n` or more).
 * Returns the words grouped per paragraph in order. */
export function paragraphsFromWords(words: string[]): string[][] {
  if (words.length === 0) return [];
  const out: string[][] = [[]];
  for (const w of words) {
    out[out.length - 1].push(w);
    if (/\n\s*\n/.test(w)) out.push([]);
  }
  // Drop trailing empty bucket if the last word ended on a blank line.
  if (out[out.length - 1].length === 0) out.pop();
  return out;
}
