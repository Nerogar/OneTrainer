import { request } from "@/api/request";

export interface SamplerActionResponse {
  ok: boolean;
  error: string | null;
}

export interface SampleData {
  file_type: "IMAGE" | "VIDEO" | "AUDIO" | string;
  format: string;
  data: string | null;
}

export interface SamplerSampleResponse {
  ok: boolean;
  error: string | null;
  sample: SampleData | null;
}

export interface SamplerStatusResponse {
  status: "idle" | "loading" | "ready" | "sampling" | "error";
  error: string | null;
  model_loaded: boolean;
  sample_progress: {
    step: number;
    max_step: number;
  };
}

export interface StandaloneSampleRequest {
  prompt?: string;
  negative_prompt?: string;
  height?: number;
  width?: number;
  seed?: number;
  random_seed?: boolean;
  diffusion_steps?: number;
  cfg_scale?: number;
  noise_scheduler?: string;
}

export const samplingApi = {
  loadModel: () => request<SamplerActionResponse>("/tools/sampling/load-model", { method: "POST" }),

  sample: (params: StandaloneSampleRequest) =>
    request<SamplerSampleResponse>("/tools/sampling/sample", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  unload: () => request<SamplerActionResponse>("/tools/sampling/unload", { method: "POST" }),

  getStatus: () => request<SamplerStatusResponse>("/tools/sampling/status"),
};
