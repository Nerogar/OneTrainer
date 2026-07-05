import { API_BASE, request } from "@/api/request";
import type { ConceptConfig, SampleConfig, SecretsConfig, TrainConfig } from "@/types/generated/config";

export interface HealthResponse {
  status: string;
  version: string;
}

export interface PresetInfo {
  name: string;
  path: string;
  is_builtin: boolean;
}

export interface FieldMetadata {
  type: string;
  default: unknown;
  nullable: boolean;
  enum_values?: string[];
  min?: number;
  max?: number;
  description?: string;
}

export type ConfigSchema = Record<string, FieldMetadata>;

export interface OptimizerParamDetail {
  title: string;
  tooltip: string;
  type: "bool" | "float" | "int" | "str" | "dict";
}

export interface OptimizerParamsResponse {
  optimizers: Record<
    string,
    {
      keys: string[];
      defaults: Record<string, unknown>;
    }
  >;
  detail_map: Record<string, OptimizerParamDetail>;
}

export const configApi = {
  health: () => request<HealthResponse>("/health"),

  getConfig: () => request<TrainConfig>("/config"),

  updateConfig: (config: TrainConfig) =>
    request<TrainConfig>("/config", {
      method: "PUT",
      body: JSON.stringify(config),
    }),

  getDefaults: () => request<TrainConfig>("/config/defaults"),

  getSchema: () => request<{ fields: ConfigSchema }>("/config/schema").then((r) => r.fields),

  exportConfig: () => request<TrainConfig>("/config/export", { method: "POST" }),

  changeOptimizer: (optimizer: string) =>
    request<TrainConfig>("/config/change-optimizer", {
      method: "POST",
      body: JSON.stringify({ optimizer }),
    }),

  getOptimizerParams: () => request<OptimizerParamsResponse>("/config/optimizer-params"),

  listPresets: () => request<PresetInfo[]>("/presets"),

  loadPreset: (path: string) =>
    request<TrainConfig>("/presets/load", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  savePreset: (name: string) =>
    request<{ name: string; path: string }>("/presets/save", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  deletePreset: (name: string) =>
    request<{ deleted: string }>(`/presets/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  getConcepts: () => request<ConceptConfig[]>("/concepts"),

  saveConcepts: (concepts: ConceptConfig[]) =>
    request<{ saved: number; path: string }>("/concepts", {
      method: "PUT",
      body: JSON.stringify(concepts),
    }),

  conceptImages: (path: string, includeSubdirectories = false, offset = 0, limit = 9999) =>
    request<{
      total: number;
      offset: number;
      images: Array<{ filename: string; path: string; caption: string | null }>;
    }>(
      `/concepts/images?path=${encodeURIComponent(path)}&offset=${offset}&limit=${limit}&include_subdirectories=${includeSubdirectories}`,
    ),

  conceptTextFile: (path: string) =>
    request<{ content: string }>(`/concepts/text-file?path=${encodeURIComponent(path)}`),

  conceptImageUrl: (path: string) => `${API_BASE}/concepts/image?path=${encodeURIComponent(path)}`,

  saveCaption: (image_path: string, caption: string) =>
    request<{ ok: boolean; saved: string }>("/concepts/save-caption", {
      method: "POST",
      body: JSON.stringify({ image_path, caption }),
    }),

  conceptStats: (path: string, includeSubdirectories: boolean, advanced: boolean) =>
    request<Record<string, unknown>>("/concepts/stats", {
      method: "POST",
      body: JSON.stringify({ path, include_subdirectories: includeSubdirectories, advanced }),
    }),

  cancelConceptStats: () => request<{ cancelled: boolean }>("/concepts/stats/cancel", { method: "DELETE" }),

  augmentationPreview: (req: { image_path: string; image: Record<string, unknown>; seed: number }) =>
    request<{ ok: boolean; image_base64: string; seed: number }>("/concepts/augmentation-preview", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  getSamples: () => request<SampleConfig[]>("/samples"),

  saveSamples: (samples: SampleConfig[]) =>
    request<{ saved: number; path: string }>("/samples", {
      method: "PUT",
      body: JSON.stringify(samples),
    }),

  getSecrets: () => request<SecretsConfig>("/secrets"),

  saveSecrets: (secrets: SecretsConfig) =>
    request<SecretsConfig>("/secrets", {
      method: "PUT",
      body: JSON.stringify(secrets),
    }),

  tensorboardLaunch: () =>
    request<{ ok: boolean; url?: string; port?: number; already_running?: boolean; error?: string }>(
      "/tensorboard/launch",
      { method: "POST" },
    ),

  tensorboardRuns: () => request<string[]>("/tensorboard/runs"),

  tensorboardTags: (run: string) => request<string[]>(`/tensorboard/scalars?run=${encodeURIComponent(run)}`),

  tensorboardScalars: (run: string, tag: string, afterStep?: number) =>
    request<Array<{ wall_time: number; step: number; value: number }>>(
      `/tensorboard/scalars/${encodeURIComponent(tag)}?run=${encodeURIComponent(run)}${afterStep != null ? `&after_step=${afterStep}` : ""}`,
    ),

  wikiPages: () => request<Array<{ title: string; pages: string[] }>>("/wiki/pages"),

  wikiPage: (slug: string) => request<{ slug: string; content: string }>(`/wiki/pages/${encodeURIComponent(slug)}`),
};
