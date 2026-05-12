import { useEffect } from "react";
import { DeskFrame } from "./components/DeskFrame";
import { DispatchModal } from "./components/DispatchModal";
import { DispatchOverlay } from "./components/DispatchOverlay";
import { OnboardingCard } from "./components/OnboardingCard";
import { OutcomeCard } from "./components/OutcomeCard";
import { Paper } from "./components/Paper";
import { useStream } from "./hooks/useStream";
import { useStore } from "./store";

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

      <DeskFrame>
        <Paper />
        <DispatchOverlay />
      </DeskFrame>

      <OnboardingCard />
      <DispatchModal />
      <OutcomeCard />

      {error && (
        <div role="alert" className="error-toast">
          {error}
        </div>
      )}
    </>
  );
}
