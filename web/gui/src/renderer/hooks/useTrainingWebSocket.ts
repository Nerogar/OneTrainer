import { useCallback, useEffect, useRef } from "react";

import { useTrainingStore } from "@/store/trainingStore";

import { useReconnectingWebSocket } from "./useReconnectingWebSocket";

interface ProgressData {
  epoch: number;
  epoch_step: number;
  epoch_sample: number;
  global_step: number;
  max_step: number;
  max_epoch: number;
}

interface StatusData {
  text: string;
}

interface SampleData {
  file_type: string;
  format: string;
  data: string | null;
}

interface SampleProgressData {
  step: number;
  max_step: number;
}

interface ErrorData {
  message: string;
}

type WsMessage =
  | { type: "progress"; data: ProgressData }
  | { type: "status"; data: StatusData }
  | { type: "sample"; data: SampleData }
  | { type: "sample_progress"; data: SampleProgressData }
  | { type: "error"; data: ErrorData };

export function useTrainingWebSocket(enabled = true): void {
  const progressBufferRef = useRef<ProgressData | null>(null);
  const rafRef = useRef<number | null>(null);

  const setStatus = useTrainingStore((s) => s.setStatus);
  const setProgress = useTrainingStore((s) => s.setProgress);
  const setError = useTrainingStore((s) => s.setError);
  const setStatusText = useTrainingStore((s) => s.setStatusText);
  const addSampleUrl = useTrainingStore((s) => s.addSampleUrl);

  const flushProgressBuffer = useCallback(() => {
    const data = progressBufferRef.current;
    if (data) {
      setProgress({
        step: data.epoch_step,
        maxStep: data.max_step,
        epoch: data.epoch,
        maxEpoch: data.max_epoch,
        loss: null,
        learningRate: null,
        elapsedTime: null,
        remainingTime: null,
      });
      setStatus("training");
      progressBufferRef.current = null;
    }
    rafRef.current = null;
  }, [setProgress, setStatus]);

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      let msg: WsMessage;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      switch (msg.type) {
        case "progress": {
          progressBufferRef.current = msg.data;
          if (rafRef.current === null) {
            rafRef.current = requestAnimationFrame(flushProgressBuffer);
          }
          break;
        }

        case "status": {
          const text = msg.data.text;
          setStatusText(text);

          if (text.startsWith("Error")) {
            setStatus("error");
          } else if (text === "Stopped") {
            setStatus("idle");
          } else if (text === "Stopping...") {
            // training thread still winding down
          } else if (text.startsWith("Starting")) {
            setStatus("preparing");
          }
          break;
        }

        case "sample": {
          const d = msg.data;
          if (d.data) {
            const mimeType =
              d.file_type === "IMAGE"
                ? "image/png"
                : d.file_type === "VIDEO"
                  ? "video/mp4"
                  : "application/octet-stream";
            const dataUrl = `data:${mimeType};base64,${d.data}`;
            addSampleUrl(dataUrl);
          }
          break;
        }

        case "sample_progress": {
          setStatusText(`Sampling: step ${msg.data.step}/${msg.data.max_step}`);
          break;
        }

        case "error": {
          setStatus("error");
          setError(msg.data.message);
          break;
        }
      }
    },
    [setStatus, setStatusText, setError, addSampleUrl, flushProgressBuffer],
  );

  useReconnectingWebSocket({
    path: "/ws/training",
    onMessage: handleMessage,
    enabled,
  });

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      progressBufferRef.current = null;
    };
  }, []);
}
