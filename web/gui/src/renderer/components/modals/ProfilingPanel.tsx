import { useState } from "react";

import { systemApi } from "@/api/systemApi";
import { Button } from "@/components/shared";

import { ModalBase } from "./ModalBase";

export interface ProfilingPanelProps {
  open: boolean;
  onClose: () => void;
}

type Status =
  | { kind: "idle" }
  | { kind: "busy" }
  | { kind: "ok"; message: string }
  | { kind: "error"; message: string };

export function ProfilingPanel({ open, onClose }: ProfilingPanelProps) {
  const [scaleneActive, setScaleneActive] = useState(false);
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  const handleDumpStack = async () => {
    setStatus({ kind: "busy" });
    try {
      const res = await systemApi.dumpStacks();
      if (res.ok && res.path) {
        setStatus({ kind: "ok", message: `Stacks written to ${res.path}` });
      } else {
        setStatus({ kind: "error", message: res.error ?? "Unknown error dumping stacks" });
      }
    } catch (e) {
      setStatus({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const handleToggleProfiling = async () => {
    setStatus({ kind: "busy" });
    try {
      const res = await systemApi.toggleProfiling();
      if (res.ok) {
        setScaleneActive(res.active);
        setStatus({
          kind: "ok",
          message: res.active ? "Scalene profiling started" : "Scalene profiling stopped",
        });
      } else {
        setStatus({ kind: "error", message: res.error ?? "Toggle failed" });
      }
    } catch (e) {
      setStatus({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const messageColor =
    status.kind === "error"
      ? "var(--color-error-500)"
      : status.kind === "ok"
        ? "var(--color-success-500)"
        : "var(--color-on-surface-secondary)";
  const message =
    status.kind === "ok" || status.kind === "error"
      ? status.message
      : status.kind === "busy"
        ? "Working..."
        : "Inactive";

  return (
    <ModalBase open={open} onClose={onClose} title="Profiling" size="sm">
      <div className="flex flex-col gap-4">
        <Button variant="secondary" onClick={handleDumpStack} disabled={status.kind === "busy"}>
          Dump Python Thread Stacks
        </Button>
        <Button
          variant={scaleneActive ? "primary" : "secondary"}
          onClick={handleToggleProfiling}
          disabled={status.kind === "busy"}
        >
          {scaleneActive ? "Stop Scalene Profiling" : "Start Scalene Profiling"}
        </Button>
        <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)]">
          <p className="text-sm break-words" style={{ color: messageColor }}>
            {message}
          </p>
        </div>
      </div>

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
