import { API_BASE, request } from "@/api/request";

export interface ToolActionResponse {
  ok: boolean;
  error: string | null;
  task_id: string | null;
}

export interface ToolStatusResponse {
  status: "idle" | "running" | "completed" | "error";
  progress: number;
  max_progress: number;
  error: string | null;
  task_id: string | null;
}

export interface QuantizationParams {
  layer_filter: string;
  layer_filter_preset: string;
  layer_filter_regex: boolean;
  svd_dtype: string;
  svd_rank: number;
  cache_dir: string | null;
}

export interface ConvertModelRequest {
  model_type: string;
  training_method: string;
  input_name: string;
  output_dtype: string;
  output_model_format: string;
  output_model_destination: string;
  quantization?: QuantizationParams | null;
}

export interface ConvertModelResponse {
  ok: boolean;
  error: string | null;
}

export interface CaptionRequest {
  model?: string;
  folder: string;
  initial_caption?: string;
  caption_prefix?: string;
  caption_postfix?: string;
  mode?: string;
  include_subdirectories?: boolean;
}

export interface ApiCaptionRequest {
  backend: "openai" | "gemini";
  folder: string;
  prompt: string;
  additional_prompts?: string[];
  mode?: string;
  include_subdirectories?: boolean;
  caption_prefix?: string;
  caption_postfix?: string;
  api_url?: string;
  api_key?: string;
  model_name?: string;
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
  enable_thinking?: boolean;
  batch_size?: number;
  requests_per_minute?: number;
  timeout?: number;
  pass_filename?: boolean;
  pass_metadata?: boolean;
  pass_current_caption?: boolean;
}

export interface PreviewRequest {
  backend?: string;
  model?: string;
  folder?: string;
  image_path?: string;
  prompt?: string;
  include_subdirectories?: boolean;
  api_url?: string;
  api_key?: string;
  model_name?: string;
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
  enable_thinking?: boolean;
  timeout?: number;
  pass_filename?: boolean;
  pass_metadata?: boolean;
  pass_current_caption?: boolean;
}

export interface PreviewResponse {
  ok: boolean;
  caption: string;
  prompt_used: string;
  image_path: string;
  image_base64: string;
  error: string | null;
}

export interface CaptionKeys {
  openai_api_key: string;
  openai_api_url: string;
  gemini_api_key: string;
}

export interface CapabilitiesResponse {
  ultralytics_available: boolean;
  caption_models: string[];
  mask_models: string[];
}

export interface MaskRequest {
  model?: string;
  folder: string;
  prompt?: string;
  mode?: string;
  threshold?: number;
  smooth?: number;
  expand?: number;
  alpha?: number;
  include_subdirectories?: boolean;
  model_path?: string;
}

export interface ImageListItem {
  path: string;
  filename: string;
  has_mask: boolean;
}

export interface YoloDetection {
  class_name: string;
  confidence: number;
  polygon: number[][];
  bbox: number[];
}

export interface YoloPredictResponse {
  ok: boolean;
  detections: YoloDetection[];
  error: string | null;
}

export const toolsApi = {
  convertModel: (params: ConvertModelRequest) =>
    request<ConvertModelResponse>("/tools/convert", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  generateCaptions: (params: CaptionRequest) =>
    request<ToolActionResponse>("/tools/captions/generate", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  generateCaptionsApi: (params: ApiCaptionRequest) =>
    request<ToolActionResponse>("/tools/captions/generate-api", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  previewCaption: (params: PreviewRequest) =>
    request<PreviewResponse>("/tools/captions/preview", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getCaptionKeys: () => request<CaptionKeys>("/tools/captions/keys"),

  saveCaptionKeys: (params: CaptionKeys) =>
    request<CaptionKeys>("/tools/captions/keys", {
      method: "PUT",
      body: JSON.stringify(params),
    }),

  getCapabilities: () => request<CapabilitiesResponse>("/tools/capabilities"),

  generateMasks: (params: MaskRequest) =>
    request<ToolActionResponse>("/tools/masks/generate", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getStatus: () => request<ToolStatusResponse>("/tools/status"),

  cancel: () => request<ToolActionResponse>("/tools/cancel", { method: "POST" }),

  listMaskEditorImages: (folder: string, includeSubdirs: boolean) =>
    request<ImageListItem[]>(
      `/tools/mask-editor/images?folder=${encodeURIComponent(folder)}&include_subdirectories=${includeSubdirs}`,
    ),

  getMaskEditorImageUrl: (path: string) => `${API_BASE}/tools/mask-editor/image?path=${encodeURIComponent(path)}`,

  saveMask: async (imagePath: string, maskBlob: Blob, smooth = 0, expand = 0, threshold = 128) => {
    const form = new FormData();
    form.append("image_path", imagePath);
    form.append("smooth", String(smooth));
    form.append("expand", String(expand));
    form.append("threshold", String(threshold));
    form.append("mask", maskBlob, "mask.png");
    const res = await fetch(`${API_BASE}/tools/mask-editor/save-mask`, {
      method: "POST",
      body: form,
    });
    return res.json() as Promise<{ ok: boolean; mask_path?: string; detail?: string }>;
  },

  yoloPredict: (imagePath: string, modelPath: string) =>
    request<YoloPredictResponse>("/tools/mask-editor/yolo-predict", {
      method: "POST",
      body: JSON.stringify({ image_path: imagePath, model_path: modelPath }),
    }),

  downloadDebugPackage: async (): Promise<string> => {
    const res = await fetch(`${API_BASE}/tools/debug-package`, {
      method: "POST",
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Failed to generate debug package: ${text}`);
    }

    const disposition = res.headers.get("Content-Disposition") ?? "";
    const filenameMatch = disposition.match(/filename="?([^"]+)"?/);
    const filename = filenameMatch?.[1] ?? "OneTrainer_debug.zip";

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);

    return filename;
  },
};
