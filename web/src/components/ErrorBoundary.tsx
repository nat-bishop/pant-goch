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
      <div className="flex h-full w-full items-center justify-center bg-bg p-8">
        <div className="max-w-md rounded-lg border border-border bg-bg-bubble-them p-6 shadow-md">
          <h2 className="text-base font-medium text-text">
            Something went wrong
          </h2>
          <p className="mt-2 text-sm text-text-muted">
            The Inbox hit a render error. Reloading usually clears it.
          </p>
          {this.state.message && (
            <pre className="mt-3 max-h-32 overflow-auto rounded border border-border bg-bg p-2 font-mono text-xs text-text-muted">
              {this.state.message}
            </pre>
          )}
          <button
            type="button"
            className="mt-4 rounded bg-accent px-4 py-2 text-sm font-medium text-text-on-accent hover:bg-accent-hover"
            onClick={() => location.reload()}
          >
            Reload
          </button>
        </div>
      </div>
    );
  }
}
