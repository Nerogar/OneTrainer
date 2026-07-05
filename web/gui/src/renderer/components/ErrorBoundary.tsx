import React, { Component, type ErrorInfo, type ReactNode } from "react";

import { ERROR_500, ON_SURFACE_DARK, SURFACE_RAISED_DARK, SURFACE_SUNKEN_DARK } from "../../shared/brandColors";

export interface FallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

interface Props {
  children: ReactNode;
  /**
   * Static fallback UI. Does NOT receive the reset function — the user cannot
   * trigger a reset from this element. Prefer `fallbackRender` instead, which
   * passes `{ error, resetErrorBoundary }` to the callback.
   */
  fallback?: ReactNode;
  /** Render prop for fallback UI that receives the error and a reset function. */
  fallbackRender?: (props: FallbackProps) => ReactNode;
  /** Called after the boundary resets and children are about to remount. Use for app-level state cleanup. */
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  /** Incremented on each reset to force children to remount with fresh state. */
  resetCount: number;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, resetCount: 0 };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  resetErrorBoundary = (): void => {
    this.setState(
      (prev) => ({ hasError: false, error: null, resetCount: prev.resetCount + 1 }),
      () => {
        this.props.onReset?.();
      },
    );
  };

  render(): ReactNode {
    if (this.state.hasError) {
      const error = this.state.error;

      if (this.props.fallbackRender && error) {
        return this.props.fallbackRender({
          error,
          resetErrorBoundary: this.resetErrorBoundary,
        });
      }

      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div style={{ padding: "24px", textAlign: "center" }}>
          <h2 style={{ color: `var(--color-error-500, ${ERROR_500})`, margin: "0 0 8px" }}>Something went wrong</h2>
          <p style={{ color: "var(--color-on-surface-secondary)", margin: "0 0 16px" }}>
            {error?.message ?? "An unexpected error occurred."}
          </p>
          {error?.stack && (
            <details style={{ marginBottom: "16px", textAlign: "left" }}>
              <summary
                style={{
                  cursor: "pointer",
                  color: "var(--color-on-surface-secondary)",
                  marginBottom: "8px",
                }}
              >
                Error details
              </summary>
              <pre
                style={{
                  fontSize: "12px",
                  padding: "12px",
                  borderRadius: "6px",
                  background: `var(--color-surface-sunken, ${SURFACE_SUNKEN_DARK})`,
                  color: "var(--color-on-surface-secondary)",
                  overflow: "auto",
                  maxHeight: "200px",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}
              >
                {error?.stack}
              </pre>
            </details>
          )}
          <button
            onClick={this.resetErrorBoundary}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: "1px solid var(--color-border-subtle, rgba(255,255,255,0.12))",
              background: `var(--color-surface-raised, ${SURFACE_RAISED_DARK})`,
              color: `var(--color-on-surface, ${ON_SURFACE_DARK})`,
              cursor: "pointer",
            }}
          >
            Try again
          </button>
        </div>
      );
    }

    // Key the children wrapper by resetCount so that a reset forces React to
    // unmount and remount all children, discarding any corrupted component state.
    return <React.Fragment key={this.state.resetCount}>{this.props.children}</React.Fragment>;
  }
}
