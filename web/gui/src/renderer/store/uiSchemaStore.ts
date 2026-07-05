import { create } from "zustand";

import type { UiSchema } from "@/types/uiSchema";

interface UiSchemaState {
  schema: UiSchema | null;
  loading: boolean;
  error: string | null;

  loadSchema: () => Promise<void>;
}

function resolveSchemaUrl(): string {
  if (typeof window !== "undefined" && window.location.protocol === "file:") {
    return "./ui-schema.json";
  }
  return "/ui-schema.json";
}

export const useUiSchemaStore = create<UiSchemaState>((set, get) => ({
  schema: null,
  loading: false,
  error: null,

  loadSchema: async () => {
    if (get().schema || get().loading) return;

    set({ loading: true, error: null });
    try {
      const url = resolveSchemaUrl();
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to load UI schema: ${res.status} ${res.statusText}`);
      }
      const schema: UiSchema = await res.json();
      set({ schema, loading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : String(err),
        loading: false,
      });
    }
  },
}));
