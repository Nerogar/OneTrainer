import { useEffect, useRef, useState } from "react";

import { toolsApi, type ToolStatusResponse } from "@/api/toolsApi";

interface UseToolPollingResult {
  status: ToolStatusResponse | null;
  isRunning: boolean;
  error: string | null;
  start: () => void;
  stop: () => void;
  setError: (error: string | null) => void;
  progress: number;
  progressLabel: string;
}

export function useToolPolling(): UseToolPollingResult {
  const [status, setStatus] = useState<ToolStatusResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!isRunning) return;

    const poll = async () => {
      try {
        const s = await toolsApi.getStatus();
        setStatus(s);
        if (s.status === "completed" || s.status === "error" || s.status === "idle") {
          setIsRunning(false);
          if (s.status === "error" && s.error) setError(s.error);
        }
      } catch {
        /* ignore */
      }
    };

    pollRef.current = setInterval(poll, 500);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [isRunning]);

  const progress = status && status.max_progress > 0 ? (status.progress / status.max_progress) * 100 : 0;
  const progressLabel =
    status && status.max_progress > 0
      ? `${status.progress} / ${status.max_progress}`
      : isRunning
        ? "Starting..."
        : "0 / 0";

  return {
    status,
    isRunning,
    error,
    start: () => {
      setError(null);
      setIsRunning(true);
      setStatus(null);
    },
    stop: () => setIsRunning(false),
    setError,
    progress,
    progressLabel,
  };
}
