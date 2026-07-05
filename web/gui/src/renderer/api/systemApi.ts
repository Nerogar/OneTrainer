import { request } from "@/api/request";

export interface ProfilingDumpResponse {
  ok: boolean;
  path: string | null;
  error: string | null;
}

export interface ProfilingToggleResponse {
  ok: boolean;
  active: boolean;
  error: string | null;
}

export const systemApi = {
  dumpStacks: () => request<ProfilingDumpResponse>("/system/profiling/dump-stacks", { method: "POST" }),

  toggleProfiling: () => request<ProfilingToggleResponse>("/system/profiling/toggle", { method: "POST" }),
};
