import { create } from "zustand";

import type { EnumDef, ModalDef, TabDef, UiSchema } from "@/types/uiSchema";

interface UiSchemaState {
  schema: UiSchema | null;
  loading: boolean;
  error: string | null;

  loadSchema: () => Promise<void>;
  getTab: (id: string) => TabDef | undefined;
  getModal: (id: string) => ModalDef | undefined;
  getEnum: (name: string) => EnumDef | undefined;
  getPredicate: (name: string) => string[] | undefined;
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

  getTab: (id: string) => get().schema?.tabs.find((t) => t.id === id),
  getModal: (id: string) => get().schema?.modals[id],
  getEnum: (name: string) => get().schema?.enums[name],
  getPredicate: (name: string) => get().schema?.predicates[name],
}));
