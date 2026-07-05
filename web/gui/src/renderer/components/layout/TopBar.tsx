import { Moon, Save, Sun, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { configApi, type PresetInfo } from "@/api/configApi";
import { SavePresetModal } from "@/components/modals/SavePresetModal";
import { useConfigField } from "@/hooks/useConfigField";
import { useConfigStore } from "@/store/configStore";
import { useUiStore } from "@/store/uiStore";
import type { ModelType, TrainingMethod } from "@/types/generated/enums";
import { ModelTypeValues } from "@/types/generated/enums";
import { TRAINING_METHODS_BY_MODEL } from "@/types/generated/modelTypeInfo";
import { enumLabel } from "@/utils/enumLabels";

export default function TopBar() {
  const { theme, toggleTheme, backendConnected } = useUiStore();
  const loadPreset = useConfigStore((s) => s.loadPreset);
  const savePreset = useConfigStore((s) => s.savePreset);
  const loadedPresetName = useConfigStore((s) => s.loadedPresetName);
  const [modelType, setModelType] = useConfigField<ModelType>("model_type");
  const [trainingMethod, setTrainingMethod] = useConfigField<TrainingMethod>("training_method");
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [showSaveModal, setShowSaveModal] = useState(false);

  useEffect(() => {
    if (backendConnected) {
      configApi
        .listPresets()
        .then(setPresets)
        .catch((err) => {
          console.error("[TopBar] Failed to fetch presets:", err);
        });
    }
  }, [backendConnected]);

  const availableMethods = TRAINING_METHODS_BY_MODEL[modelType ?? "STABLE_DIFFUSION_15"];

  const handlePresetLoad = (path: string) => {
    const preset = presets.find((p) => p.path === path);
    loadPreset(path, preset?.name);
  };
  const handleSavePreset = () => {
    setShowSaveModal(true);
  };
  const handleSavePresetConfirm = async (name: string) => {
    await savePreset(name);
    setShowSaveModal(false);
    try {
      const updated = await configApi.listPresets();
      setPresets(updated);
    } catch {
      /* best-effort refresh */
    }
  };
  const handleModelChange = (val: string) => {
    setModelType(val as ModelType);
    // If current training method is not compatible with new model, reset to FINE_TUNE
    const methods = TRAINING_METHODS_BY_MODEL[val as ModelType];
    if (trainingMethod && !methods.includes(trainingMethod)) {
      setTrainingMethod("FINE_TUNE");
    }
  };

  const [presetMenuOpen, setPresetMenuOpen] = useState(false);
  const presetMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!presetMenuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (presetMenuRef.current && !presetMenuRef.current.contains(e.target as Node)) {
        setPresetMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [presetMenuOpen]);

  const handleDeletePreset = async (name: string) => {
    if (!confirm(`Delete preset "${name}"? This cannot be undone.`)) return;
    try {
      await configApi.deletePreset(name);
      const updated = await configApi.listPresets();
      setPresets(updated);
    } catch (err) {
      alert(`Failed to delete preset: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  return (
    <header className="top-bar">
      <div className="top-bar-left">
        <h1 className="top-bar-title">OneTrainer</h1>
        {new URLSearchParams(window.location.search).get("dev") === "1" && (
          <span className={`connection-status ${backendConnected ? "connected" : "disconnected"}`}>
            {backendConnected ? "Connected" : "Disconnected"}
          </span>
        )}
      </div>
      <div className="top-bar-right">
        <div ref={presetMenuRef} className="relative">
          <button
            type="button"
            className="top-bar-select cursor-pointer text-left"
            onClick={() => setPresetMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={presetMenuOpen}
          >
            {loadedPresetName ? `Preset: ${loadedPresetName.replace(/^#/, "")}` : "Load Preset..."}
          </button>
          {presetMenuOpen && (
            <div
              role="menu"
              className="absolute z-50 right-0 mt-1 min-w-[220px] max-h-80 overflow-y-auto rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] shadow-lg"
            >
              {presets.filter((p) => p.is_builtin).length > 0 && (
                <div>
                  <div className="px-3 py-1 text-xs uppercase tracking-wide text-[var(--color-on-surface-secondary)]">
                    Built-in
                  </div>
                  {presets
                    .filter((p) => p.is_builtin)
                    .map((p) => (
                      <button
                        key={p.path}
                        type="button"
                        className="block w-full px-3 py-1.5 text-left text-sm text-[var(--color-on-surface)] hover:bg-[var(--color-border-subtle)]"
                        onClick={() => {
                          handlePresetLoad(p.path);
                          setPresetMenuOpen(false);
                        }}
                      >
                        {p.name}
                      </button>
                    ))}
                </div>
              )}
              {presets.filter((p) => !p.is_builtin).length > 0 && (
                <div>
                  <div className="px-3 py-1 text-xs uppercase tracking-wide text-[var(--color-on-surface-secondary)] mt-1">
                    User
                  </div>
                  {presets
                    .filter((p) => !p.is_builtin)
                    .map((p) => (
                      <div
                        key={p.path}
                        className="flex items-center justify-between px-3 py-1.5 text-sm text-[var(--color-on-surface)] hover:bg-[var(--color-border-subtle)]"
                      >
                        <button
                          type="button"
                          onClick={() => {
                            handlePresetLoad(p.path);
                            setPresetMenuOpen(false);
                          }}
                          className="text-left flex-1 truncate"
                        >
                          {p.name}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeletePreset(p.name)}
                          className="ml-2 p-1 rounded hover:bg-[rgba(239,68,68,0.15)] text-[var(--color-error-500)]"
                          aria-label={`Delete preset ${p.name}`}
                          title={`Delete preset "${p.name}"`}
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                </div>
              )}
              {presets.length === 0 && (
                <div className="px-3 py-2 text-sm text-[var(--color-on-surface-secondary)]">No presets available</div>
              )}
            </div>
          )}
        </div>
        <button onClick={handleSavePreset} className="theme-toggle" aria-label="Save preset" title="Save preset">
          <Save className="w-4 h-4" />
        </button>

        <div className="top-bar-separator" aria-hidden="true" />

        <select
          value={modelType ?? "STABLE_DIFFUSION_15"}
          onChange={(e) => handleModelChange(e.target.value)}
          className="top-bar-select"
        >
          {ModelTypeValues.map((mt) => (
            <option key={mt} value={mt}>
              {enumLabel(mt)}
            </option>
          ))}
        </select>

        <select
          value={trainingMethod ?? "FINE_TUNE"}
          onChange={(e) => setTrainingMethod(e.target.value as TrainingMethod)}
          className="top-bar-select"
        >
          {availableMethods.map((m) => (
            <option key={m} value={m}>
              {enumLabel(m)}
            </option>
          ))}
        </select>

        <div className="top-bar-separator" aria-hidden="true" />

        <button
          onClick={toggleTheme}
          className="theme-toggle"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
      <SavePresetModal
        isOpen={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        onSave={handleSavePresetConfirm}
      />
    </header>
  );
}
