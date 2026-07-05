import { useEffect, useState } from "react";

import { toolsApi } from "@/api/toolsApi";
import { Button, DirPicker, FilePicker, FormEntry, ProgressBar, Select, Toggle } from "@/components/shared";
import { useToolPolling } from "@/hooks/useToolPolling";
import { MASK_MODELS, MASK_MODES } from "@/types/generated/dropdownSources";

import { ModalBase } from "./ModalBase";

export interface MaskToolModalProps {
  open: boolean;
  onClose: () => void;
}

interface MaskState {
  model: string;
  folder: string;
  prompt: string;
  mode: string;
  threshold: number;
  smooth: number;
  expand: number;
  alpha: number;
  include_subdirectories: boolean;
  model_path: string;
}

const DEFAULT_STATE: MaskState = {
  model: "ClipSeg",
  folder: "",
  prompt: "",
  mode: "fill",
  threshold: 0.3,
  smooth: 5,
  expand: 10,
  alpha: 1.0,
  include_subdirectories: false,
  model_path: "",
};

export function MaskToolModal({ open, onClose }: MaskToolModalProps) {
  const [state, setState] = useState<MaskState>({ ...DEFAULT_STATE });
  const polling = useToolPolling();
  const { status, isRunning, error } = polling;
  const [ultralyticsAvailable, setUltralyticsAvailable] = useState<boolean | null>(null);

  const update = <K extends keyof MaskState>(field: K, value: MaskState[K]) => {
    setState((prev) => ({ ...prev, [field]: value }));
  };

  useEffect(() => {
    if (!open) return;
    toolsApi
      .getCapabilities()
      .then((caps) => {
        setUltralyticsAvailable(caps.ultralytics_available);
      })
      .catch(() => {
        /* ignore */
      });
  }, [open]);

  const handleGenerate = async () => {
    if (!state.folder) {
      polling.setError("Please select a folder.");
      return;
    }
    if (state.model === "YOLO" && !state.model_path) {
      polling.setError("Please select a YOLO model file (.pt).");
      return;
    }
    polling.start();

    try {
      const result = await toolsApi.generateMasks({
        model: state.model,
        folder: state.folder,
        prompt: state.prompt,
        mode: state.mode,
        threshold: state.threshold,
        smooth: state.smooth,
        expand: state.expand,
        alpha: state.alpha,
        include_subdirectories: state.include_subdirectories,
        model_path: state.model === "YOLO" ? state.model_path : undefined,
      });

      if (!result.ok) {
        polling.setError(result.error ?? "Failed to start mask generation");
        polling.stop();
      }
    } catch (err) {
      polling.setError(err instanceof Error ? err.message : "Unknown error");
      polling.stop();
    }
  };

  const handleCancel = async () => {
    try {
      await toolsApi.cancel();
      polling.stop();
    } catch {
      /* ignore */
    }
  };

  const handleOpenEditor = async () => {
    const api = window.electronAPI;
    if (api?.openMaskEditor) {
      await api.openMaskEditor(state.folder);
    } else {
      window.open(`/?route=mask-editor&folder=${encodeURIComponent(state.folder)}`, "_blank");
    }
  };

  return (
    <ModalBase open={open} onClose={onClose} title="Batch Generate Masks" size="md" closeOnBackdrop={!isRunning}>
      <div className="flex flex-col gap-4">
        <Select
          label="Model"
          options={MASK_MODELS}
          value={state.model}
          onChange={(v) => update("model", v)}
          disabled={isRunning}
        />

        {state.model === "YOLO" && (
          <>
            <FilePicker
              label="YOLO Model File"
              value={state.model_path}
              onChange={(v) => update("model_path", v)}
              disabled={isRunning}
              filters={[{ name: "YOLO Models", extensions: ["pt"] }]}
              tooltip="Select a YOLO .pt model file for segmentation"
            />
            {ultralyticsAvailable === false && (
              <div className="p-2 rounded-[var(--radius-sm)] bg-[var(--color-error-500)]/10 border border-[var(--color-error-500)]/30">
                <p className="text-xs text-[var(--color-error-500)]">
                  Ultralytics is not installed. Install with: pip install ultralytics
                </p>
              </div>
            )}
          </>
        )}

        <DirPicker label="Folder" value={state.folder} onChange={(v) => update("folder", v)} disabled={isRunning} />

        {state.model !== "YOLO" && (
          <FormEntry
            label="Prompt"
            value={state.prompt}
            onChange={(v) => update("prompt", String(v))}
            placeholder="Masking prompt (for ClipSeg)..."
            disabled={isRunning}
          />
        )}

        <Select
          label="Mode"
          options={MASK_MODES}
          value={state.mode}
          onChange={(v) => update("mode", v)}
          disabled={isRunning}
        />

        <div className="grid grid-cols-2 gap-3">
          <FormEntry
            label="Threshold"
            value={state.threshold}
            onChange={(v) => update("threshold", Number(v))}
            type="number"
            placeholder="0.0 - 1.0"
            disabled={isRunning}
          />
          <FormEntry
            label="Smooth"
            value={state.smooth}
            onChange={(v) => update("smooth", Number(v))}
            type="number"
            placeholder="5"
            disabled={isRunning}
          />
          <FormEntry
            label="Expand"
            value={state.expand}
            onChange={(v) => update("expand", Number(v))}
            type="number"
            placeholder="10"
            disabled={isRunning}
          />
          <FormEntry
            label="Alpha"
            value={state.alpha}
            onChange={(v) => update("alpha", Number(v))}
            type="number"
            placeholder="1.0"
            disabled={isRunning}
          />
        </div>

        <Toggle
          label="Include subfolders"
          value={state.include_subdirectories}
          onChange={(v) => update("include_subdirectories", v)}
          disabled={isRunning}
        />

        <div className="pt-2">
          <ProgressBar
            value={polling.progress}
            label={polling.progressLabel}
            indeterminate={isRunning && polling.progress === 0}
          />
        </div>

        {error && (
          <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-error-500)]/10 border border-[var(--color-error-500)]/30">
            <p className="text-sm text-[var(--color-error-500)]">{error}</p>
          </div>
        )}

        {status?.status === "completed" && (
          <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-cobalt-600)]/10 border border-[var(--color-cobalt-600)]/30">
            <p className="text-sm text-[var(--color-cobalt-600)]">Mask generation completed successfully.</p>
          </div>
        )}
      </div>

      <div className="flex justify-between mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <div className="flex gap-2">
          <Button variant="primary" onClick={handleGenerate} disabled={isRunning}>
            {isRunning ? "Generating..." : "Create Masks"}
          </Button>
          <Button variant="secondary" onClick={handleOpenEditor} disabled={isRunning}>
            Open Interactive Editor
          </Button>
          {isRunning && (
            <Button variant="danger" onClick={handleCancel}>
              Cancel
            </Button>
          )}
        </div>
        <Button variant="secondary" onClick={onClose} disabled={isRunning}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
