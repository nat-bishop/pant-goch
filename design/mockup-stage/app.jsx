/* global React, ReactDOM */

const { useState, useEffect, useRef, useMemo, useCallback, Fragment } = React;

const MAP_VB = { w: 100, h: 78 };

// ──────────────────────────────────────────────────────────
// Gloss popup context — one popup at root, set by hover
// handlers on any .gloss-trigger span.
// ──────────────────────────────────────────────────────────

const GlossContext = React.createContext({
  show: () => {},
  hide: () => {},
});

function useGloss() { return React.useContext(GlossContext); }

// ──────────────────────────────────────────────────────────
// Tokenizer — paragraphs use [phrase|gloss_id] markup inline.
// Returns a list of stream-units: text-word | space | gloss.
// Each unit is one beat of the streaming animation.
// ──────────────────────────────────────────────────────────

function tokenize(text) {
  // returns [{ type: 'text'|'gloss', text, glossId? }]
  const tokens = [];
  const re = /\[([^|\]]+)\|([^\]]+)\]/g;
  let lastIdx = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > lastIdx) {
      tokens.push({ type: 'text', text: text.slice(lastIdx, m.index) });
    }
    tokens.push({ type: 'gloss', text: m[1], glossId: m[2] });
    lastIdx = m.index + m[0].length;
  }
  if (lastIdx < text.length) {
    tokens.push({ type: 'text', text: text.slice(lastIdx) });
  }
  return tokens;
}

// Expand parsed tokens into stream-units (words, spaces, glosses).
function expandUnits(parsed) {
  const units = [];
  for (const tok of parsed) {
    if (tok.type === 'gloss') {
      units.push({ kind: 'gloss', text: tok.text, glossId: tok.glossId });
    } else {
      // split text into words + spaces
      const parts = tok.text.split(/(\s+)/).filter(Boolean);
      for (const p of parts) {
        units.push({ kind: /^\s+$/.test(p) ? 'space' : 'word', text: p });
      }
    }
  }
  return units;
}

// ──────────────────────────────────────────────────────────
// Streaming hook — works in stream-units (not raw words)
// ──────────────────────────────────────────────────────────

function useChapterStream(paragraphs, speedMs, active, onComplete) {
  const [paraIdx, setParaIdx] = useState(0);
  const [unitIdx, setUnitIdx] = useState(0);
  const [done, setDone] = useState(false);
  const timerRef = useRef(null);

  const allUnits = useMemo(
    () => paragraphs.map((p) => expandUnits(tokenize(p.text))),
    [paragraphs]
  );

  useEffect(() => {
    if (!active || done) return;
    if (paraIdx >= paragraphs.length) {
      setDone(true);
      onComplete?.();
      return;
    }
    const units = allUnits[paraIdx];
    if (unitIdx >= units.length) {
      timerRef.current = setTimeout(() => {
        setParaIdx((i) => i + 1); setUnitIdx(0);
      }, 320);
      return () => clearTimeout(timerRef.current);
    }
    const u = units[unitIdx];
    let delay = speedMs;
    if (u.kind === 'word' || u.kind === 'gloss') {
      const txt = u.text;
      if (/[—–-]$/.test(txt)) delay = speedMs * 2.4;
      if (/[,;:]$/.test(txt)) delay = speedMs * 2.0;
      if (/[.?!]$/.test(txt)) delay = speedMs * 3.4;
    } else {
      delay = Math.max(20, speedMs * 0.3); // spaces advance fast
    }
    timerRef.current = setTimeout(() => setUnitIdx((i) => i + 1), delay);
    return () => clearTimeout(timerRef.current);
  }, [active, paragraphs, paraIdx, unitIdx, done, speedMs, onComplete, allUnits]);

  return { paraIdx, unitIdx, done, allUnits };
}

// ──────────────────────────────────────────────────────────
// Gloss trigger span — wraps a phrase, opens popup on hover
// ──────────────────────────────────────────────────────────

function GlossTrigger({ text, glossId }) {
  const gloss = useGloss();
  const ref = useRef(null);

  const open = useCallback(() => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    gloss.show(glossId, rect);
  }, [glossId, gloss]);

  const close = useCallback(() => {
    gloss.hide(glossId);
  }, [glossId, gloss]);

  return (
    <span
      ref={ref}
      className="gloss-trigger"
      data-gloss-id={glossId}
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={close}
      tabIndex={0}
    >
      {text}
    </span>
  );
}

// The floating popup itself — rendered once at root.
function GlossPopup({ entry }) {
  const [stayOpen, setStayOpen] = useState(false);
  if (!entry) return null;
  const g = window.GLOSSES[entry.glossId];
  if (!g) return null;
  const { rect } = entry;
  // Default: position popup to the LEFT of the trigger so it
  // floats over the photograph (left pane), not the prose panel.
  const popupW = 280;
  const popupH = 130;
  const gap = 14;
  // Place above-or-below depending on space
  const above = rect.top > popupH + gap + 24;
  const top = above ? rect.top - popupH - gap : rect.bottom + gap;
  // Place left-aligned with the trigger, but flushed left into
  // the photograph by ~30px so the arrow points back to the
  // trigger from the right of the popup.
  let left = Math.max(20, rect.left - popupW * 0.6);
  // clamp inside viewport
  left = Math.min(left, window.innerWidth - popupW - 20);

  return (
    <div
      className={"gloss-popup" + (above ? " above" : " below")}
      style={{
        left: `${left}px`,
        top: `${top}px`,
        width: `${popupW}px`,
      }}
    >
      <div className="gloss-popup-name">{g.name}</div>
      <div className="gloss-popup-body">{g.body}</div>
      {g.cross_ref && (
        <div className="gloss-popup-cross">↳ {g.cross_ref}</div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────────────────
// World pane — image cross-fade + map overlay during walks +
// switchable mode (location / map / notes)
// ──────────────────────────────────────────────────────────

function ImageSlots({ currentUrl }) {
  const [slotA, setSlotA] = useState(currentUrl);
  const [slotB, setSlotB] = useState(null);
  const [activeSlot, setActiveSlot] = useState("a");

  useEffect(() => {
    if (!currentUrl) return;
    if (activeSlot === "a" && slotA !== currentUrl) {
      setSlotB(currentUrl);
      const t = setTimeout(() => {
        setActiveSlot("b");
        setSlotA(null);
      }, 50);
      return () => clearTimeout(t);
    }
    if (activeSlot === "b" && slotB !== currentUrl) {
      setSlotA(currentUrl);
      const t = setTimeout(() => {
        setActiveSlot("a");
        setSlotB(null);
      }, 50);
      return () => clearTimeout(t);
    }
  }, [currentUrl, activeSlot, slotA, slotB]);

  return (
    <Fragment>
      {slotA && (
        <div
          className="world-img"
          style={{
            backgroundImage: `url("${slotA}")`,
            opacity: activeSlot === "a" ? 1 : 0,
          }}
        />
      )}
      {slotB && (
        <div
          className="world-img"
          style={{
            backgroundImage: `url("${slotB}")`,
            opacity: activeSlot === "b" ? 1 : 0,
          }}
        />
      )}
    </Fragment>
  );
}

function MapView({ currentLocId, visitedIds, walking, fromId, toId }) {
  const locs = window.WORLD.locations;
  const from = locs.find((l) => l.id === fromId);
  const to = locs.find((l) => l.id === toId);
  const tracerD = useMemo(() => {
    if (!from || !to) return null;
    const key = [fromId, toId].sort().join("|");
    return window.ROAD_PATHS[key] || `M ${from.xy[0]} ${from.xy[1]} L ${to.xy[0]} ${to.xy[1]}`;
  }, [fromId, toId]);

  const contours = [
    { cx: 44, cy: 10, rx: 22, ry: 8 },
    { cx: 46, cy: 11, rx: 32, ry: 11 },
    { cx: 18, cy: 64, rx: 14, ry: 5 },
  ];

  return (
    <div className="world-map-inner">
      <svg
        className="world-map-svg"
        viewBox={`0 0 ${MAP_VB.w} ${MAP_VB.h}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {contours.map((c, i) => (
          <ellipse key={i} cx={c.cx} cy={c.cy} rx={c.rx} ry={c.ry} className="contour" />
        ))}
        <path
          className="river"
          d="M 100 64 C 88 67, 78 62, 66 65 S 44 72, 32 68 S 12 64, 0 68"
        />
        {Object.entries(window.ROAD_PATHS).map(([k, d]) => (
          <path key={k} d={d} className="road" />
        ))}
        {walking && tracerD && (
          <path key={fromId + "-" + toId} d={tracerD} className="tracer" />
        )}
        {/* all pins — static, location-aware */}
        {locs.map((l) => {
          const isHere = walking ? l.id === fromId : l.id === currentLocId;
          const isThere = walking && l.id === toId;
          const isVisited = visitedIds.has(l.id) && !isHere && !isThere;
          return (
            <g key={l.id} transform={`translate(${l.xy[0]}, ${l.xy[1]})`}>
              <circle
                r={isHere || isThere ? 0.9 : isVisited ? 0.6 : 0.45}
                className={
                  "pin" +
                  (isHere ? " pin--here" : "") +
                  (isThere ? " pin--there" : "") +
                  (isVisited ? " pin--visited" : "")
                }
              />
              <text x="1.4" y="0.7" className={
                "pin-label" +
                (isThere ? " pin-label--there" : "") +
                (!isHere && !isThere && !isVisited ? " pin-label--dim" : "")
              }>
                {l.name}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function NotesView({ chapters, onJump }) {
  if (!chapters || chapters.length === 0) {
    return (
      <div className="notes-empty">
        The case file is empty. The investigation has not begun.
      </div>
    );
  }
  return (
    <div className="notes-list">
      {chapters.map((c, i) => {
        const excerpt = (c.paragraphs?.[0]?.text || "")
          .replace(/\[([^|\]]+)\|[^\]]+\]/g, "$1");
        return (
          <div className="note" key={c.id + "-" + i} onClick={() => onJump(c)}>
            <div className="note-meta">
              <span>Entry {String(i + 1).padStart(2, "0")}</span>
              <span>{c.date_label}</span>
            </div>
            <h3 className="note-title">{c.location_name}</h3>
            <p className="note-excerpt">{excerpt}</p>
            <div className="note-link">↳ scroll to entry</div>
          </div>
        );
      })}
    </div>
  );
}

function WorldPane({
  mode, setMode, walking, walkFrom, walkTo,
  currentImage, currentLocId, locationName, dateLabel,
  visitedIds, chapters, onJumpToChapter,
}) {
  // During a walk, the map is always shown regardless of user-set
  // mode. After the walk, return to whatever mode was set.
  const showMap = walking || mode === "map";
  const showNotes = !walking && mode === "notes";
  const showLocation = !walking && mode === "location";

  return (
    <div className="world">
      {/* image slots — always rendered, just hidden behind other modes */}
      <ImageSlots currentUrl={currentImage} />
      <div className="world-tint" />
      <div className="world-grain" />

      {/* top tab strip — location / map / notes */}
      <div className="world-tabs" role="tablist" aria-label="left pane mode">
        {["location", "map", "notes"].map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            className={"world-tab" + (mode === m ? " active" : "")}
            onClick={() => setMode(m)}
            disabled={walking}
          >
            {m}
          </button>
        ))}
      </div>

      {/* mode-specific overlays */}
      <div className={"world-overlay world-overlay--map" + (showMap ? " in" : "")}>
        <div className="world-overlay-eyebrow">
          {walking ? "on the road" : "the survey · pant goch & the marches"}
        </div>
        <MapView
          currentLocId={currentLocId}
          visitedIds={visitedIds}
          walking={walking}
          fromId={walkFrom}
          toId={walkTo}
        />
        {walking && walkFrom && walkTo && (
          <div className="world-overlay-foot">
            {locsName(walkFrom)} <span className="arrow">→</span> {locsName(walkTo)}
          </div>
        )}
        {!walking && (
          <div className="world-overlay-foot foot-meta">
            {visitedIds.size} of {window.WORLD.locations.length} visited
          </div>
        )}
      </div>

      <div className={"world-overlay world-overlay--notes" + (showNotes ? " in" : "")}>
        <div className="world-overlay-eyebrow">case file · m. pritchard</div>
        <NotesView chapters={chapters} onJump={onJumpToChapter} />
      </div>

      {/* slate at bottom-left — only visible in LOCATION mode */}
      {showLocation && locationName && (
        <div className="world-slate">
          <div className="world-slate-eyebrow">present location</div>
          <div className="world-slate-name">{locationName}</div>
          {dateLabel && <div className="world-slate-time">{dateLabel}</div>}
        </div>
      )}
    </div>
  );
}

function locsName(id) {
  return window.WORLD.locations.find((l) => l.id === id)?.name ?? id;
}

// ──────────────────────────────────────────────────────────
// Chapter — content inside the right panel
// ──────────────────────────────────────────────────────────

function Chapter({ scene, isActive, onComplete, onPick, picked }) {
  const speedMs = 60;
  const stream = useChapterStream(scene.paragraphs, speedMs, isActive, onComplete);
  const isFullyDone = !isActive || stream.done;
  const paragraphs = scene.paragraphs;
  const paraRefs = useRef([]);
  const sectionRef = useRef(null);

  // Page-turn scroll: when a new paragraph starts and the previous
  // paragraph's bottom is past ~68% of the panel, smooth-scroll so
  // the previous paragraph sits at the top — leaving exactly one
  // paragraph of context above the streaming one. The text in view
  // does not move except at these explicit page breaks.
  useEffect(() => {
    if (!isActive) return;
    if (stream.paraIdx === 0) return; // first paragraph: no page-turn
    const prev = paraRefs.current[stream.paraIdx - 1];
    const section = sectionRef.current;
    if (!prev || !section) return;
    const panel = section.closest(".panel-scroll");
    if (!panel) return;
    const panelRect = panel.getBoundingClientRect();
    const prevRect = prev.getBoundingClientRect();
    const prevBottomInPanel = prevRect.bottom - panelRect.top;
    // Trigger a page-turn if the previous paragraph extends past
    // 68% of the panel height. The new paragraph would otherwise
    // start below the safe reading zone.
    if (prevBottomInPanel > panelRect.height * 0.68) {
      const padding = 56; // breathing room at top so prev para isn't flush
      const newTop = panel.scrollTop + (prevRect.top - panelRect.top) - padding;
      panel.scrollTo({ top: newTop, behavior: "smooth" });
    }
  }, [stream.paraIdx, isActive]);

  // Same page-turn logic when the choices block is about to appear —
  // the choices are a content block too and shouldn't pop in below
  // the fold.
  useEffect(() => {
    if (!stream.done || !isActive) return;
    const lastPara = paraRefs.current[paragraphs.length - 1];
    const section = sectionRef.current;
    if (!lastPara || !section) return;
    const panel = section.closest(".panel-scroll");
    if (!panel) return;
    const panelRect = panel.getBoundingClientRect();
    const lastRect = lastPara.getBoundingClientRect();
    const lastBottomInPanel = lastRect.bottom - panelRect.top;
    // Choices need ~50% of panel height for their block. If the
    // last paragraph extends past 55%, page-turn so the last para
    // is at the top and choices have room below.
    if (lastBottomInPanel > panelRect.height * 0.55) {
      const padding = 56;
      const newTop = panel.scrollTop + (lastRect.top - panelRect.top) - padding;
      panel.scrollTo({ top: newTop, behavior: "smooth" });
    }
  }, [stream.done, isActive, paragraphs.length]);

  return (
    <section className="chapter" id={`ch-${scene.id}`} ref={sectionRef}>
      <header className="chapter-head">
        <span className="chapter-no">Chapter {romanize(scene.chapter_no)}</span>
        <h2 className="chapter-title">{scene.location_name}</h2>
        <span className="chapter-date">{scene.date_label}</span>
      </header>
      <div className="chapter-prose">
        {paragraphs.map((p, i) => {
          if (isActive && i > stream.paraIdx) return null;
          const isCurrent = isActive && i === stream.paraIdx && !stream.done;
          const units = stream.allUnits[i] || [];
          const unitsToShow = isCurrent ? stream.unitIdx : units.length;
          return (
            <div
              className={"para " + (isCurrent ? "current" : "prior")}
              key={i}
              ref={(el) => (paraRefs.current[i] = el)}
            >
              <p>
                {units.slice(0, unitsToShow).map((u, j) => {
                  if (u.kind === 'gloss') {
                    return <GlossTrigger key={j} text={u.text} glossId={u.glossId} />;
                  }
                  if (u.kind === 'space') {
                    return <span key={j}>{u.text}</span>;
                  }
                  return <span key={j} className="word">{u.text}</span>;
                })}
                {isCurrent && <span className="caret" />}
              </p>
            </div>
          );
        })}
        {isFullyDone && (
          <ChoicesBlock
            options={scene.options}
            onPick={onPick}
            picked={picked}
          />
        )}
      </div>
    </section>
  );
}

function ChoicesBlock({ options, onPick, picked }) {
  if (picked) {
    return <div className="player-pick">» {picked.text}</div>;
  }
  return (
    <div className="choices-wrap">
      <div className="choices-eyebrow">— how I went on —</div>
      <ol className="choices">
        {options.map((opt, i) => {
          const kindLabel =
            opt.kind === "leave"  ? `leave for ${shortLoc(opt.leaves_for)}` :
            opt.kind === "risk"   ? "risk" :
            opt.kind === "intuit" ? "intuit" : "engage";
          return (
            <li key={i}>
              <button
                className="choice"
                data-kind={opt.kind ?? "engage"}
                onClick={() => onPick(opt, i)}
              >
                <span className="choice-num">{i + 1}.</span>
                <span>
                  {opt.text}
                  {opt.note && <span className="choice-note">{opt.note}</span>}
                  <span className="choice-kind">{kindLabel}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function shortLoc(id) {
  const l = window.WORLD.locations.find((x) => x.id === id);
  return l ? l.name.toLowerCase() : id?.replace(/_/g, " ");
}

function romanize(n) {
  const map = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
  return map[n] ?? String(n);
}

// ──────────────────────────────────────────────────────────
// App
// ──────────────────────────────────────────────────────────

function App() {
  const [started, setStarted] = useState(false);
  const [trail, setTrail] = useState([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [walk, setWalk] = useState(null);
  const [currentChapterIdx, setCurrentChapterIdx] = useState(0);
  const [worldMode, setWorldMode] = useState("location"); // location|map|notes
  const [activeGloss, setActiveGloss] = useState(null);
  const glossHideTimer = useRef(null);
  const panelScrollRef = useRef(null);

  const chapters = useMemo(
    () => trail.map((t) => window.SCENES[t.sceneId]).filter(Boolean),
    [trail]
  );

  const visitedIds = useMemo(() => {
    const s = new Set();
    chapters.forEach((c) => s.add(c.location_id));
    return s;
  }, [chapters]);

  const currentImageUrl = useMemo(() => {
    if (walk) {
      const toScene = window.SCENES[window.SCENE_FOR_LOCATION[walk.toId]];
      return toScene?.backdrop ?? window.WORLD.frontispiece_backdrop;
    }
    if (!started) return window.WORLD.frontispiece_backdrop;
    const c = chapters[currentChapterIdx];
    return c?.backdrop ?? window.WORLD.frontispiece_backdrop;
  }, [walk, started, chapters, currentChapterIdx]);

  const currentLocId = useMemo(() => {
    if (!started) return window.WORLD.player_starts_at;
    return chapters[currentChapterIdx]?.location_id ?? window.WORLD.player_starts_at;
  }, [started, chapters, currentChapterIdx]);

  const currentLocationName = useMemo(() => {
    if (!started) return "Approaching the Marches";
    return chapters[currentChapterIdx]?.location_name ?? "";
  }, [started, chapters, currentChapterIdx]);

  const currentDateLabel = useMemo(() => {
    if (!started) return "the welsh borders · spring 2026";
    return chapters[currentChapterIdx]?.date_label ?? "";
  }, [started, chapters, currentChapterIdx]);

  // gloss popup show/hide with a small grace period so moving from
  // trigger to popup (and back) doesn't flicker
  const glossApi = useMemo(() => ({
    show: (glossId, rect) => {
      if (glossHideTimer.current) clearTimeout(glossHideTimer.current);
      setActiveGloss({ glossId, rect });
    },
    hide: () => {
      if (glossHideTimer.current) clearTimeout(glossHideTimer.current);
      glossHideTimer.current = setTimeout(() => setActiveGloss(null), 160);
    },
  }), []);

  const startBook = useCallback(() => {
    const first = window.SCENE_FOR_LOCATION[window.WORLD.player_starts_at];
    setTrail([{ sceneId: first, picked: null }]);
    setActiveIdx(0);
    setStarted(true);
    setCurrentChapterIdx(0);
    setWorldMode("location");
  }, []);

  const onPick = useCallback((idx, opt) => {
    setTrail((t) => t.map((e, i) => i === idx ? { ...e, picked: opt } : e));
    if (opt.kind === "leave" && opt.leaves_for) {
      const nextSceneKey = window.SCENE_FOR_LOCATION[opt.leaves_for];
      if (!nextSceneKey || !window.SCENES[nextSceneKey]) return;
      const fromId = chapters[idx]?.location_id;
      setWalk({ fromId, toId: opt.leaves_for });
      const APPEND_AT_MS = 2400;
      const CLEAR_AT_MS  = 3000;
      setTimeout(() => {
        setTrail((t) => [...t, { sceneId: nextSceneKey, picked: null }]);
        setActiveIdx(idx + 1);
        setTimeout(() => {
          const el = document.getElementById(`ch-${nextSceneKey}`);
          el?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 60);
      }, APPEND_AT_MS);
      setTimeout(() => {
        setWalk(null);
        setWorldMode("location");
      }, CLEAR_AT_MS);
    }
  }, [chapters]);

  // Track which chapter is centred in the right-panel viewport,
  // so the left pane's photograph follows the reader's position.
  useEffect(() => {
    if (!started) return;
    const scrollEl = panelScrollRef.current;
    if (!scrollEl) return;
    const onScroll = () => {
      const panelRect = scrollEl.getBoundingClientRect();
      const centerY = panelRect.top + panelRect.height * 0.5;
      let bestIdx = 0;
      let bestDist = Infinity;
      chapters.forEach((c, i) => {
        const el = document.getElementById(`ch-${c.id}`);
        if (!el) return;
        const r = el.getBoundingClientRect();
        const mid = (r.top + r.bottom) / 2;
        const d = Math.abs(mid - centerY);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      });
      setCurrentChapterIdx(bestIdx);
    };
    onScroll();
    scrollEl.addEventListener("scroll", onScroll, { passive: true });
    return () => scrollEl.removeEventListener("scroll", onScroll);
  }, [started, chapters.length]);

  // jump to a chapter from notes view
  const onJumpToChapter = useCallback((scene) => {
    setWorldMode("location");
    setTimeout(() => {
      const el = document.getElementById(`ch-${scene.id}`);
      el?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  }, []);

  // close popup on Escape
  useEffect(() => {
    const h = (e) => {
      if (e.key === "Escape") setActiveGloss(null);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  return (
    <GlossContext.Provider value={glossApi}>
      <div className="stage">
        <WorldPane
          mode={worldMode}
          setMode={setWorldMode}
          walking={!!walk}
          walkFrom={walk?.fromId}
          walkTo={walk?.toId}
          currentImage={currentImageUrl}
          currentLocId={currentLocId}
          locationName={currentLocationName}
          dateLabel={currentDateLabel}
          visitedIds={visitedIds}
          chapters={chapters}
          onJumpToChapter={onJumpToChapter}
        />

        <div className="panel">
          <div className="panel-bar">
            <span>case file · m. pritchard · 2026</span>
            <div className="panel-bar-meta">
              <span>{started ? `${chapters.length} ${chapters.length === 1 ? "entry" : "entries"}` : "—"}</span>
              <span className="panel-bar-sep" />
              <a onClick={() => {
                setStarted(false); setTrail([]); setActiveIdx(0); setCurrentChapterIdx(0);
                setWorldMode("location");
              }}>↺ restart</a>
            </div>
          </div>

          <div className="panel-scroll" ref={panelScrollRef}>
            {!started ? (
              <header className="frontis">
                <div />
                <div>
                  <div className="frontis-mark">a folk-noir investigation</div>
                  <h1 className="frontis-title">Pant Goch.</h1>
                  <div className="frontis-rule" />
                  <p className="frontis-sub">
                    Six weeks. A village in the Welsh Marches that has held its
                    breath since 2019. A letter from a man who places flowers,
                    monthly, at the bend where his daughter went off the road.
                    The story is whether there is a story.
                  </p>
                  <p className="frontis-byline">
                    field notebook · m. pritchard · 2026
                  </p>
                </div>
                <div className="frontis-scroll">
                  <button onClick={startBook}>begin the investigation</button>
                </div>
              </header>
            ) : (
              <Fragment>
                {chapters.map((c, i) => (
                  <Chapter
                    key={c.id + "-" + i}
                    scene={c}
                    isActive={i === activeIdx && !trail[i].picked}
                    onComplete={() => {}}
                    onPick={(opt) => onPick(i, opt)}
                    picked={trail[i].picked}
                  />
                ))}
                <div className="colophon">
                  <div className="colophon-rule" />
                  in progress
                </div>
              </Fragment>
            )}
          </div>
        </div>

        <GlossPopup entry={activeGloss} />
      </div>
    </GlossContext.Provider>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
