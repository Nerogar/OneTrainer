import { request } from "@/api/request";

export interface TrainingActionResponse {
  ok: boolean;
  error: string | null;
}

export interface TrainingStatusResponse {
  status: "idle" | "running" | "stopping" | "error";
  error: string | null;
  start_time: number | null;
}

export interface CustomSampleRequest {
  prompt?: string;
  negative_prompt?: string;
  height?: number;
  width?: number;
  seed?: number;
  random_seed?: boolean;
  diffusion_steps?: number;
  cfg_scale?: number;
}

export const trainingApi = {
  start: (options?: { reattach?: boolean }) =>
    request<TrainingActionResponse>("/training/start", {
      method: "POST",
      body: JSON.stringify(options ?? {}),
    }),

  stop: () => request<TrainingActionResponse>("/training/stop", { method: "POST" }),

  sample: () => request<TrainingActionResponse>("/training/sample", { method: "POST" }),

  sampleCustom: (params: CustomSampleRequest) =>
    request<TrainingActionResponse>("/training/sample/custom", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  backup: () => request<TrainingActionResponse>("/training/backup", { method: "POST" }),

  save: () => request<TrainingActionResponse>("/training/save", { method: "POST" }),

  getStatus: () => request<TrainingStatusResponse>("/training/status"),
};
