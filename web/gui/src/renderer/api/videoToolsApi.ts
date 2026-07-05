import { request } from "@/api/request";

export interface ExtractClipsParams {
  video_path?: string;
  directory?: string;
  batch_mode?: boolean;
  output_dir: string;
  time_start?: string;
  time_end?: string;
  output_subdirectories?: boolean;
  split_at_cuts?: boolean;
  max_length?: number;
  fps?: number;
  remove_borders?: boolean;
  crop_variation?: number;
}

export interface ExtractImagesParams {
  video_path?: string;
  directory?: string;
  batch_mode?: boolean;
  output_dir: string;
  time_start?: string;
  time_end?: string;
  output_subdirectories?: boolean;
  images_per_second?: number;
  blur_removal?: number;
  remove_borders?: boolean;
  crop_variation?: number;
}

export interface DownloadParams {
  url?: string;
  link_list_path?: string;
  batch_mode?: boolean;
  output_dir: string;
  additional_args?: string;
}

export interface VideoToolResponse {
  ok: boolean;
  error: string | null;
  message: string | null;
}

export interface VideoToolStatusResponse {
  status: "idle" | "running" | "completed" | "error";
  message: string | null;
  error: string | null;
}

export const videoToolsApi = {
  extractClips: (params: ExtractClipsParams) =>
    request<VideoToolResponse>("/tools/video/extract-clips", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  extractImages: (params: ExtractImagesParams) =>
    request<VideoToolResponse>("/tools/video/extract-images", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  download: (params: DownloadParams) =>
    request<VideoToolResponse>("/tools/video/download", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getStatus: () => request<VideoToolStatusResponse>("/tools/video/status"),
};
