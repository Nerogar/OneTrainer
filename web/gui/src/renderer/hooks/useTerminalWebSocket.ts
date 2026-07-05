import { useEffect, useRef } from "react";

import { useReconnectingWebSocket } from "./useReconnectingWebSocket";

const FLUSH_INTERVAL_MS = 50;

interface LogData {
  text: string;
  ts: number;
}

interface WsMessage {
  type: "log";
  data: LogData;
}

export function useTerminalWebSocket(onData: (text: string) => void, enabled = true): void {
  const pendingRef = useRef<string[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onDataRef = useRef(onData);
  onDataRef.current = onData;

  useReconnectingWebSocket({
    path: "/ws/terminal",
    onMessage: (event: MessageEvent) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        if (msg.type === "log" && msg.data?.text) {
          pendingRef.current.push(msg.data.text);
        }
      } catch {
        // ignore unparseable
      }
    },
    enabled,
  });

  useEffect(() => {
    if (!enabled) return;

    flushTimerRef.current = setInterval(() => {
      if (pendingRef.current.length > 0) {
        const batch = pendingRef.current.join("");
        pendingRef.current = [];
        onDataRef.current(batch);
      }
    }, FLUSH_INTERVAL_MS);

    return () => {
      if (flushTimerRef.current) {
        clearInterval(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      if (pendingRef.current.length > 0) {
        const batch = pendingRef.current.join("");
        pendingRef.current = [];
        onDataRef.current(batch);
      }
    };
  }, [enabled]);
}
