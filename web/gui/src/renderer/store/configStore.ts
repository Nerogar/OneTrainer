import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

import type { OptimizerParamsResponse } from "@/api/configApi";
import { configApi } from "@/api/configApi";
import type { TrainConfig } from "@/types/generated/config";

function setByPath(obj: Record<string, unknown>, path: string, value: unknown): void {
  if (import.meta.env.DEV) {
    const parentPath = path.split(".").slice(0, -1).join(".");
    if (parentPath && getByPath(obj, parentPath) === undefined) {
      console.warn(
        `[configStore] setByPath: intermediate path "${parentPath}" does not exist in config. ` +
          `Path "${path}" may be a typo.`,
      );
    }
  }

  const keys = path.split(".");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let current: any = obj;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (current[key] === undefined || current[key] === null) {
      current[key] = {};
    }
    current = current[key];
  }

  const lastKey = keys[keys.length - 1];
  current[lastKey] = value;
}

export function getByPath(obj: Record<string, unknown>, path: string): unknown {
  const keys = path.split(".");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let current: any = obj;

  for (const key of keys) {
    if (current === undefined || current === null) {
      return undefined;
    }
    current = current[key];
  }

  return current;
}

const SYNC_DEBOUNCE_MS = 500;

type ImmerSet = (fn: (draft: ConfigState) => void) => void;

async function withLoading(set: ImmerSet, fn: () => Promise<void>): Promise<void> {
  set((draft) => {
    draft.isLoading = true;
    draft.error = null;
  });
  try {
    await fn();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[configStore]", message);
    set((draft) => {
      draft.error = message;
    });
  } finally {
    set((draft) => {
      draft.isLoading = false;
    });
  }
}

interface ConfigState {
  config: TrainConfig | null;
  isDirty: boolean;
  isLoading: boolean;
  error: string | null;
  loadedPresetName: string | null;
  optimizerParams: OptimizerParamsResponse | null;
  optimizerParamsFetched: boolean;
  _optimizerParamsFetching: boolean;
  _syncTimer: ReturnType<typeof setTimeout> | null;
  _syncGeneration: number;
  _conceptSaveTimer: ReturnType<typeof setTimeout> | null;
  _sampleSaveTimer: ReturnType<typeof setTimeout> | null;
  _secretsSaveTimer: ReturnType<typeof setTimeout> | null;

  loadConfig: () => Promise<void>;
  updateField: (path: string, value: unknown) => void;
  updateConfig: (partial: Partial<TrainConfig>) => void;
  syncToBackend: () => Promise<void>;
  loadConcepts: () => Promise<void>;
  loadSamples: () => Promise<void>;
  loadPreset: (presetPath: string, presetName?: string) => Promise<void>;
  savePreset: (name: string) => Promise<void>;
  changeOptimizer: (optimizer: string) => Promise<void>;
  loadDefaults: () => Promise<void>;
  exportConfig: () => Promise<TrainConfig>;
  fetchOptimizerParams: () => Promise<void>;
  flushPendingChanges: () => Promise<void>;
  clearError: () => void;
  destroy: () => void;
}

function cancelPendingSync(get: () => ConfigState, set: ImmerSet): void {
  const timer = get()._syncTimer;
  if (timer !== null) {
    clearTimeout(timer);
    set((draft) => {
      draft._syncTimer = null;
    });
  }
}

function scheduleDebouncedSync(get: () => ConfigState, set: ImmerSet): void {
  cancelPendingSync(get, set);
  const timer = setTimeout(() => {
    set((draft) => {
      draft._syncTimer = null;
    });
    void get().syncToBackend();
  }, SYNC_DEBOUNCE_MS);
  set((draft) => {
    draft._syncTimer = timer;
  });
}

function incrementGeneration(get: () => ConfigState, set: ImmerSet): number {
  const next = get()._syncGeneration + 1;
  set((draft) => {
    draft._syncGeneration = next;
  });
  return next;
}

export const useConfigStore = create<ConfigState>()(
  immer((set, get) => ({
    config: null,
    isDirty: false,
    isLoading: false,
    error: null,
    loadedPresetName: null,
    _syncTimer: null,
    _syncGeneration: 0,
    _conceptSaveTimer: null,
    _sampleSaveTimer: null,
    _secretsSaveTimer: null,
    optimizerParams: null,
    optimizerParamsFetched: false,
    _optimizerParamsFetching: false,

    loadConfig: async () => {
      await withLoading(set, async () => {
        const config = await configApi.getConfig();
        cancelPendingSync(get, set);
        set((draft) => {
          draft.config = config;
          draft.isDirty = false;
        });
      });
      await Promise.all([get().loadConcepts(), get().loadSamples()]);
    },

    updateField: (path: string, value: unknown) => {
      set((draft) => {
        if (draft.config === null) return;
        setByPath(draft.config as unknown as Record<string, unknown>, path, value);
        draft.isDirty = true;
        draft.error = null;
      });
      scheduleDebouncedSync(get, set);

      if (path === "concepts") {
        const timer = get()._conceptSaveTimer;
        if (timer !== null) clearTimeout(timer);
        const saveTimer = setTimeout(() => {
          set((draft) => {
            draft._conceptSaveTimer = null;
          });
          const concepts = get().config?.concepts;
          if (concepts != null) {
            configApi.saveConcepts(concepts).catch((err) => {
              console.error("[configStore] Failed to save concepts file:", err);
            });
          }
        }, SYNC_DEBOUNCE_MS);
        set((draft) => {
          draft._conceptSaveTimer = saveTimer;
        });
      }

      if (path === "samples" || path.startsWith("samples.")) {
        const timer = get()._sampleSaveTimer;
        if (timer !== null) clearTimeout(timer);
        const saveTimer = setTimeout(() => {
          set((draft) => {
            draft._sampleSaveTimer = null;
          });
          const samples = get().config?.samples;
          if (samples != null) {
            configApi.saveSamples(samples).catch((err) => {
              console.error("[configStore] Failed to save samples file:", err);
            });
          }
        }, SYNC_DEBOUNCE_MS);
        set((draft) => {
          draft._sampleSaveTimer = saveTimer;
        });
      }

      if (path === "secrets" || path.startsWith("secrets.")) {
        const timer = get()._secretsSaveTimer;
        if (timer !== null) clearTimeout(timer);
        const saveTimer = setTimeout(() => {
          set((draft) => {
            draft._secretsSaveTimer = null;
          });
          const secrets = get().config?.secrets;
          if (secrets != null) {
            configApi.saveSecrets(secrets).catch((err) => {
              console.error("[configStore] Failed to save secrets file:", err);
            });
          }
        }, SYNC_DEBOUNCE_MS);
        set((draft) => {
          draft._secretsSaveTimer = saveTimer;
        });
      }
    },

    updateConfig: (partial: Partial<TrainConfig>) => {
      set((draft) => {
        if (draft.config === null) return;
        Object.assign(draft.config, partial);
        draft.isDirty = true;
        draft.error = null;
      });
      scheduleDebouncedSync(get, set);
    },

    syncToBackend: async () => {
      const { config, isDirty } = get();
      if (config === null || !isDirty) return;

      cancelPendingSync(get, set);

      set((draft) => {
        draft.isLoading = true;
        draft.error = null;
      });

      const gen = incrementGeneration(get, set);

      try {
        const reconciled = await configApi.updateConfig(config);
        if (gen === get()._syncGeneration) {
          set((draft) => {
            draft.config = reconciled;
            draft.isDirty = false;
            draft.isLoading = false;
          });
        }
      } catch (err) {
        if (gen === get()._syncGeneration) {
          set((draft) => {
            draft.error = err instanceof Error ? err.message : String(err);
            draft.isLoading = false;
          });
        }
      }
    },

    flushPendingChanges: async () => {
      const { _syncTimer, _conceptSaveTimer, _sampleSaveTimer, _secretsSaveTimer } = get();

      const tasks: Array<Promise<unknown>> = [];

      if (_syncTimer !== null) {
        clearTimeout(_syncTimer);
        set((draft) => {
          draft._syncTimer = null;
        });
      }
      if (get().isDirty) {
        tasks.push(get().syncToBackend());
      }

      if (_conceptSaveTimer !== null) {
        clearTimeout(_conceptSaveTimer);
        set((draft) => {
          draft._conceptSaveTimer = null;
        });
        const concepts = get().config?.concepts;
        if (concepts != null) {
          tasks.push(
            configApi.saveConcepts(concepts).catch((err) => {
              console.error("[configStore] flush: failed to save concepts:", err);
            }),
          );
        }
      }

      if (_sampleSaveTimer !== null) {
        clearTimeout(_sampleSaveTimer);
        set((draft) => {
          draft._sampleSaveTimer = null;
        });
        const samples = get().config?.samples;
        if (samples != null) {
          tasks.push(
            configApi.saveSamples(samples).catch((err) => {
              console.error("[configStore] flush: failed to save samples:", err);
            }),
          );
        }
      }

      if (_secretsSaveTimer !== null) {
        clearTimeout(_secretsSaveTimer);
        set((draft) => {
          draft._secretsSaveTimer = null;
        });
        const secrets = get().config?.secrets;
        if (secrets != null) {
          tasks.push(
            configApi.saveSecrets(secrets).catch((err) => {
              console.error("[configStore] flush: failed to save secrets:", err);
            }),
          );
        }
      }

      await Promise.all(tasks);
    },

    loadConcepts: async () => {
      try {
        const concepts = await configApi.getConcepts();
        set((draft) => {
          if (draft.config) {
            draft.config.concepts = concepts;
          }
        });
      } catch {
        // concept file may not exist yet
      }
    },

    loadSamples: async () => {
      try {
        const samples = await configApi.getSamples();
        set((draft) => {
          if (draft.config) {
            draft.config.samples = samples;
          }
        });
      } catch {
        // sample file may not exist yet
      }
    },

    loadPreset: async (presetPath: string, presetName?: string) => {
      cancelPendingSync(get, set);
      incrementGeneration(get, set);

      await withLoading(set, async () => {
        const config = await configApi.loadPreset(presetPath);
        const name = presetName ?? presetPath.replace(/.*[/\\]/, "").replace(/\.json$/, "");
        set((draft) => {
          draft.config = config;
          draft.isDirty = false;
          draft.loadedPresetName = name;
        });
      });
      await Promise.all([get().loadConcepts(), get().loadSamples()]);
    },

    savePreset: async (name: string) => {
      const { isDirty } = get();
      if (isDirty) {
        cancelPendingSync(get, set);
        incrementGeneration(get, set);
        await get().syncToBackend();
      }

      await withLoading(set, async () => {
        await configApi.savePreset(name);
      });
    },

    changeOptimizer: async (optimizer: string) => {
      cancelPendingSync(get, set);
      incrementGeneration(get, set);

      await withLoading(set, async () => {
        const config = await configApi.changeOptimizer(optimizer);
        set((draft) => {
          draft.config = config;
          draft.isDirty = false;
        });
      });
    },

    loadDefaults: async () => {
      cancelPendingSync(get, set);
      incrementGeneration(get, set);

      await withLoading(set, async () => {
        const defaults = await configApi.getDefaults();
        set((draft) => {
          draft.config = defaults;
          draft.isDirty = false;
        });
      });
    },

    exportConfig: async () => {
      const { isDirty } = get();
      if (isDirty) {
        cancelPendingSync(get, set);
        await get().syncToBackend();
      }

      set((draft) => {
        draft.error = null;
      });

      try {
        return await configApi.exportConfig();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        set((draft) => {
          draft.error = message;
        });
        throw err;
      }
    },

    fetchOptimizerParams: async () => {
      const { optimizerParams, _optimizerParamsFetching } = get();
      if (optimizerParams || _optimizerParamsFetching) return;

      set((draft) => {
        draft._optimizerParamsFetching = true;
      });
      try {
        const data = await configApi.getOptimizerParams();
        set((draft) => {
          draft.optimizerParams = data;
          draft.optimizerParamsFetched = true;
          draft._optimizerParamsFetching = false;
        });
      } catch {
        set((draft) => {
          draft.optimizerParamsFetched = true;
          draft._optimizerParamsFetching = false;
        });
      }
    },

    clearError: () => {
      set((draft) => {
        draft.error = null;
      });
    },

    destroy: () => {
      const syncTimer = get()._syncTimer;
      if (syncTimer !== null) clearTimeout(syncTimer);
      const conceptTimer = get()._conceptSaveTimer;
      if (conceptTimer !== null) clearTimeout(conceptTimer);
      const sampleTimer = get()._sampleSaveTimer;
      if (sampleTimer !== null) clearTimeout(sampleTimer);
      const secretsTimer = get()._secretsSaveTimer;
      if (secretsTimer !== null) clearTimeout(secretsTimer);
      set((draft) => {
        draft.config = null;
        draft.isDirty = false;
        draft.isLoading = false;
        draft.error = null;
        draft.loadedPresetName = null;
        draft._syncTimer = null;
        draft._syncGeneration = 0;
        draft._conceptSaveTimer = null;
        draft._sampleSaveTimer = null;
        draft._secretsSaveTimer = null;
      });
    },
  })),
);
