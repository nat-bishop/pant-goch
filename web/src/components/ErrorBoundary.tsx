import React from "react";

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  message?: string;
}

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(err: unknown): State {
    return {
      hasError: true,
      message: err instanceof Error ? err.message : String(err),
    };
  }

  componentDidCatch(err: unknown, info: React.ErrorInfo) {
    console.error("render error", err, info);
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="flex min-h-screen w-full items-center justify-center bg-desk p-8">
        <div className="max-w-md border border-desk-rule bg-desk-2 p-6">
          <h2 className="text-base text-chrome-bright tracking-[0.16em] uppercase">
            Render error
          </h2>
          <p className="mt-3 text-sm text-chrome-dim">
            The desk hit a render error. Reloading usually clears it.
          </p>
          {this.state.message && (
            <pre className="mt-3 max-h-32 overflow-auto border border-desk-rule bg-desk p-2 font-mono text-xs text-chrome-dim">
              {this.state.message}
            </pre>
          )}
          <button
            type="button"
            className="mt-5 border-[1.5px] border-stamp bg-transparent px-5 py-2 text-xs font-bold uppercase tracking-[0.32em] text-stamp-soft hover:bg-stamp hover:text-paper transition-colors"
            onClick={() => location.reload()}
          >
            Reload
          </button>
        </div>
      </div>
    );
  }
}
