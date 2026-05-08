import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/inter/standard.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import App from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { DebugPreview } from "./debug/DebugPreview";
import "./index.css";

// DEV-only fixture harness. `?debug-preview=1` swaps the live app for a
// grid of debug docks fed by canned engine states (web/src/debug/fixtures/).
// Lets each tab be eyeballed without booting llama/api/SSE. Production
// builds tree-shake DebugPreview because the gate's static literal is
// always false outside DEV.
const previewMode = import.meta.env.DEV
  && new URLSearchParams(window.location.search).get("debug-preview") === "1";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      {previewMode ? <DebugPreview /> : <App />}
    </ErrorBoundary>
  </React.StrictMode>,
);
