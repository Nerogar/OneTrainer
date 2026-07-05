import { useEffect, useRef } from "react";

import { WS_BASE } from "@/api/request";

const INITIAL_RETRY_MS = 1000;
const MAX_RETRY_MS = 30000;
const BACKOFF_FACTOR = 2;

interface UseReconnectingWebSocketOptions {
  path: string;
  onMessage: (event: MessageEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
  enabled?: boolean;
}

export function useReconnectingWebSocket({
  path,
  onMessage,
  onOpen,
  onClose,
  enabled = true,
}: UseReconnectingWebSocketOptions): React.RefObject<WebSocket | null> {
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(INITIAL_RETRY_MS);

  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!enabled) return;

    let active = true;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const url = `${WS_BASE}${path}`;

    function connect() {
      if (!active) return;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        retryRef.current = INITIAL_RETRY_MS;
        onOpenRef.current?.();
      };

      ws.onmessage = (event) => {
        onMessageRef.current(event);
      };

      ws.onclose = () => {
        wsRef.current = null;
        onCloseRef.current?.();
        scheduleReconnect();
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    function scheduleReconnect() {
      if (!active) return;

      retryTimer = setTimeout(() => {
        retryRef.current = Math.min(retryRef.current * BACKOFF_FACTOR, MAX_RETRY_MS);
        connect();
      }, retryRef.current);
    }

    connect();

    return () => {
      active = false;
      if (retryTimer) {
        clearTimeout(retryTimer);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, path]);

  return wsRef;
}
