import { useEffect } from "react";
import { Authorize } from "./components/Authorize";
import { DispatchModal } from "./components/DispatchModal";
import { Inbox } from "./components/Inbox";
import { OutcomeCard } from "./components/OutcomeCard";
import { Paper } from "./components/Paper";
import { StateStrip } from "./components/StateStrip";
import { Topbar } from "./components/Topbar";
import { useStream } from "./hooks/useStream";
import { useStore } from "./store";

/** Root layout per the v.0.7 mockup: topbar → state strip → paper + authorize
 * → inbox cards. Outcome overlays on terminal. */
export default function App() {
  const startMission = useStore((s) => s.startMission);
  const error = useStore((s) => s.errorMessage);

  useEffect(() => {
    void startMission();
  }, [startMission]);

  useStream();

  return (
    <>
      <div className="grain" />
      <div className="vignette" />

      <Topbar />
      <StateStrip />

      <main className="flex-1 flex flex-col items-center gap-14 px-6 py-20">
        <Paper />
        <Authorize />
      </main>

      <Inbox />

      <DispatchModal />
      <OutcomeCard />

      {error && (
        <div
          role="alert"
          className="fixed bottom-6 right-6 z-50 max-w-md bg-stamp text-paper px-5 py-3 text-xs uppercase tracking-[0.2em] shadow-lg"
        >
          {error}
        </div>
      )}
    </>
  );
}
