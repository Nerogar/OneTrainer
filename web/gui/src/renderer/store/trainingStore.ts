import { create } from "zustand";

import { trainingApi } from "@/api/trainingApi";
import { useUiStore } from "@/store/uiStore";

export type TrainingStatus = "idle" | "preparing" | "training" | "error";

export interface TrainingProgress {
  step: number;
  maxStep: number;
  epoch: number;
  maxEpoch: number;
  loss: number | null;
  learningRate: number | null;
  elapsedTime: number | null;
  remainingTime: number | null;
}

export interface SampleEntry {
  id: number;
  url: string;
}

interface TrainingState {
  status: TrainingStatus;
  progress: TrainingProgress | null;
  error: string | null;
  statusText: string;
  sampleUrls: SampleEntry[];
  latestSample: string | null;
  startTime: number | null;
  nextSampleId: number;

  setStatus: (status: TrainingStatus) => void;
  setProgress: (progress: TrainingProgress) => void;
  setError: (error: string | null) => void;
  setStatusText: (text: string) => void;
  addSampleUrl: (url: string) => void;
  clearSamples: () => void;
  reset: () => void;

  fetchStatus: () => Promise<void>;

  startTraining: (options?: { reattach?: boolean }) => Promise<void>;
  stopTraining: () => Promise<void>;
  sampleNow: () => Promise<void>;
  backupNow: () => Promise<void>;
  saveNow: () => Promise<void>;
}

const INITIAL_STATE: Pick<
  TrainingState,
  "status" | "progress" | "error" | "statusText" | "sampleUrls" | "latestSample" | "startTime" | "nextSampleId"
> = {
  status: "idle",
  progress: null,
  error: null,
  statusText: "",
  sampleUrls: [],
  latestSample: null,
  startTime: null,
  nextSampleId: 0,
};

export const useTrainingStore = create<TrainingState>((set, get) => ({
  ...INITIAL_STATE,

  setStatus: (status) => set({ status }),
  setProgress: (progress) => set({ progress }),
  setError: (error) => set({ error }),
  setStatusText: (text) => set({ statusText: text }),
  addSampleUrl: (url) =>
    set((s) => {
      const entry: SampleEntry = { id: s.nextSampleId, url };
      const urls = s.sampleUrls.length >= 50 ? [...s.sampleUrls.slice(1), entry] : [...s.sampleUrls, entry];
      return { sampleUrls: urls, latestSample: url, nextSampleId: s.nextSampleId + 1 };
    }),
  clearSamples: () => set({ sampleUrls: [], latestSample: null }),
  reset: () => set(INITIAL_STATE),

  fetchStatus: async () => {
    try {
      const res = await trainingApi.getStatus();
      if (res.status === "running") {
        set({ status: "training" });
      } else if (res.status === "stopping") {
        set({ status: "training", statusText: "Stopping..." });
      } else if (res.status === "error") {
        set({ status: "error", error: res.error });
      } else {
        set({ status: "idle" });
      }
    } catch {
      // backend unreachable
    }
  },

  startTraining: async (options?: { reattach?: boolean }) => {
    const { status } = get();
    if (status === "training" || status === "preparing") return;

    // Open the backend log panel so the terminal WebSocket mounts before the
    // trainer emits its first output — otherwise early logs are dropped until
    // the user opens the panel manually.
    useUiStore.getState().setTerminalOpen(true);

    set({
      status: "preparing",
      error: null,
      statusText: options?.reattach ? "Reattaching..." : "Starting training...",
    });

    try {
      const res = await trainingApi.start(options);
      if (!res.ok) {
        set({ status: "error", error: res.error, statusText: res.error ?? "Failed to start" });
      } else {
        set({ startTime: Date.now() });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start training";
      set({ status: "error", error: message, statusText: message });
    }
  },

  stopTraining: async () => {
    try {
      const res = await trainingApi.stop();
      if (res.ok) {
        set({ statusText: "Stopping..." });
      }
    } catch {
      // status updates via WebSocket
    }
  },

  sampleNow: async () => {
    try {
      await trainingApi.sample();
    } catch {
      // best effort
    }
  },

  backupNow: async () => {
    try {
      await trainingApi.backup();
    } catch {
      // best effort
    }
  },

  saveNow: async () => {
    try {
      await trainingApi.save();
    } catch {
      // best effort
    }
  },
}));
