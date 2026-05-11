import { useEffect, useState } from "react";
import { nowZulu } from "../lib/time";
import { useStore } from "../store";
import { Sidebar } from "./Sidebar";

export function DeskFrame({ children }: { children: React.ReactNode }) {
  const meta = useStore((s) => s.mission_meta);
  const [railOpen, setRailOpen] = useState(true);
  const [headerTime, setHeaderTime] = useState(nowZulu);

  useEffect(() => {
    const id = setInterval(() => {
      // The formatted string only changes at minute rollover; skip the
      // setState call when nothing actually moved so children don't re-render.
      const next = nowZulu();
      setHeaderTime((prev) => (prev === next ? prev : next));
    }, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className={`desk-root ${railOpen ? "rail-open" : "rail-closed"}`}>
      <header className="desk-head">
        <span className="desk-head-op">
          {meta ? `Op. ${meta.codename}` : "HANDLER"}
        </span>
        <span className="desk-head-time">{headerTime}</span>
      </header>
      <div className="desk-stage">
        <Sidebar open={railOpen} onToggle={() => setRailOpen((o) => !o)} />
        <div className="paper-wrap">{children}</div>
      </div>
    </div>
  );
}
