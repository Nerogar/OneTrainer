import { useCallback, useEffect, useRef, useState } from "react";

import { type SamplerStatusResponse, samplingApi, type StandaloneSampleRequest } from "@/api/samplingApi";
import { Button, Card, FormEntry, ProgressBar, Select, Toggle } from "@/components/shared";
import { NoiseSchedulerValues } from "@/types/generated/enums";
import { TEXTAREA_FULL } from "@/utils/inputStyles";

import { ModalBase } from "./ModalBase";

export interface StandaloneSamplingModalProps {
  open: boolean;
  onClose: () => void;
}

interface SampleState {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  seed: number;
  random_seed: boolean;
  diffusion_steps: number;
  cfg_scale: number;
  noise_scheduler: string;
}

const DEFAULT_SAMPLE: SampleState = {
  prompt: "",
  negative_prompt: "",
  width: 512,
  height: 512,
  seed: 42,
  random_seed: false,
  diffusion_steps: 20,
  cfg_scale: 7,
  noise_scheduler: "DDIM",
};

type SamplerStatus = SamplerStatusResponse["status"];

const STATUS_LABELS: Record<SamplerStatus, string> = {
  idle: "Not loaded",
  loading: "Loading model...",
  ready: "Ready",
  sampling: "Generating sample...",
  error: "Error",
};

const STATUS_COLORS: Record<SamplerStatus, string> = {
  idle: "var(--color-on-surface-secondary)",
  loading: "var(--color-warning-500, #f59e0b)",
  ready: "var(--color-success-500)",
  sampling: "var(--color-cobalt-600)",
  error: "var(--color-error-500)",
};

export function StandaloneSamplingModal({ open, onClose }: StandaloneSamplingModalProps) {
  const [sample, setSample] = useState<SampleState>({ ...DEFAULT_SAMPLE });
  const [status, setStatus] = useState<SamplerStatus>("idle");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sampleImage, setSampleImage] = useState<string | null>(null);
  const [progress, setProgress] = useState({ step: 0, max_step: 0 });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const update = <K extends keyof SampleState>(field: K, value: SampleState[K]) => {
    setSample((prev) => ({ ...prev, [field]: value }));
  };

  const fetchStatus = useCallback(async () => {
    try {
      const res = await samplingApi.getStatus();
      setStatus(res.status);
      setModelLoaded(res.model_loaded);
      setErrorMessage(res.error);
      setProgress(res.sample_progress);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!open) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    fetchStatus();
    pollRef.current = setInterval(fetchStatus, 1000);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [open, fetchStatus]);

  useEffect(() => {
    if (!open) return;
    if (status !== "loading" && status !== "sampling") return;

    if (pollRef.current) {
      clearInterval(pollRef.current);
    }
    pollRef.current = setInterval(fetchStatus, 300);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [open, status, fetchStatus]);

  const handleLoadModel = async () => {
    setErrorMessage(null);
    try {
      const res = await samplingApi.loadModel();
      if (!res.ok) {
        setErrorMessage(res.error);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to load model");
    }
    fetchStatus();
  };

  const handleUnloadModel = async () => {
    setErrorMessage(null);
    setSampleImage(null);
    try {
      const res = await samplingApi.unload();
      if (!res.ok) {
        setErrorMessage(res.error);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to unload model");
    }
    fetchStatus();
  };

  const handleSample = async () => {
    setErrorMessage(null);
    try {
      const params: StandaloneSampleRequest = {
        prompt: sample.prompt,
        negative_prompt: sample.negative_prompt,
        width: sample.width,
        height: sample.height,
        seed: sample.seed,
        random_seed: sample.random_seed,
        diffusion_steps: sample.diffusion_steps,
        cfg_scale: sample.cfg_scale,
        noise_scheduler: sample.noise_scheduler,
      };
      const res = await samplingApi.sample(params);
      if (!res.ok) {
        setErrorMessage(res.error);
      } else if (res.sample?.data && res.sample.file_type === "IMAGE") {
        setSampleImage(`data:image/png;base64,${res.sample.data}`);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Sampling failed");
    }
    fetchStatus();
  };

  const isLoading = status === "loading";
  const isSampling = status === "sampling";
  const isBusy = isLoading || isSampling;
  const canSample = modelLoaded && !isBusy;
  const canLoad = !modelLoaded && !isBusy;
  const canUnload = modelLoaded && !isBusy;
  const progressPercent = progress.max_step > 0 ? (progress.step / progress.max_step) * 100 : 0;

  return (
    <ModalBase open={open} onClose={onClose} title="Standalone Sampling" size="xl" closeOnBackdrop={false}>
      <Card hoverable={false} padding="sm" className="mb-4">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: STATUS_COLORS[status] }} />
            <div className="flex flex-col">
              <span className="text-sm font-medium text-[var(--color-on-surface)]">{STATUS_LABELS[status]}</span>
              {errorMessage && (
                <span className="text-xs text-[var(--color-error-500)] mt-0.5 break-all">{errorMessage}</span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="primary" size="sm" onClick={handleLoadModel} disabled={!canLoad} loading={isLoading}>
              Load Model
            </Button>
            <Button variant="secondary" size="sm" onClick={handleUnloadModel} disabled={!canUnload}>
              Unload Model
            </Button>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Prompt</span>
            <textarea
              value={sample.prompt}
              onChange={(e) => update("prompt", e.target.value)}
              rows={3}
              disabled={isBusy}
              className={`${TEXTAREA_FULL} resize-y`}
              placeholder="Enter prompt..."
            />
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Negative Prompt</span>
            <textarea
              value={sample.negative_prompt}
              onChange={(e) => update("negative_prompt", e.target.value)}
              rows={2}
              disabled={isBusy}
              className={`${TEXTAREA_FULL} resize-y`}
              placeholder="Enter negative prompt..."
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <FormEntry
              label="Width"
              value={sample.width}
              onChange={(v) => update("width", Number(v))}
              type="number"
              disabled={isBusy}
            />
            <FormEntry
              label="Height"
              value={sample.height}
              onChange={(v) => update("height", Number(v))}
              type="number"
              disabled={isBusy}
            />
            <FormEntry
              label="Steps"
              value={sample.diffusion_steps}
              onChange={(v) => update("diffusion_steps", Number(v))}
              type="number"
              disabled={isBusy}
            />
            <FormEntry
              label="CFG Scale"
              value={sample.cfg_scale}
              onChange={(v) => update("cfg_scale", Number(v))}
              type="number"
              disabled={isBusy}
            />
            <Select
              label="Scheduler"
              options={[...NoiseSchedulerValues]}
              value={sample.noise_scheduler}
              onChange={(v) => update("noise_scheduler", v)}
              disabled={isBusy}
            />
            <FormEntry
              label="Seed"
              value={sample.seed}
              onChange={(v) => update("seed", Number(v))}
              type="number"
              disabled={isBusy}
            />
            <Toggle
              label="Random Seed"
              value={sample.random_seed}
              onChange={(v) => update("random_seed", v)}
              disabled={isBusy}
            />
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="aspect-square bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] rounded-[var(--radius-sm)] flex items-center justify-center overflow-hidden">
            {sampleImage ? (
              <img src={sampleImage} alt="Generated sample" className="max-w-full max-h-full object-contain" />
            ) : (
              <span className="text-sm text-[var(--color-on-surface-secondary)]">
                {isSampling
                  ? "Generating..."
                  : modelLoaded
                    ? "Model ready -- click Generate to sample"
                    : "Load a model to begin sampling"}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-3 mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        {isSampling && (
          <ProgressBar value={progressPercent} label={`Sampling step ${progress.step} / ${progress.max_step}`} />
        )}
        {isLoading && <ProgressBar value={0} indeterminate label="Loading model..." />}

        <div className="flex justify-between">
          <Button variant="primary" onClick={handleSample} disabled={!canSample} loading={isSampling}>
            {isSampling ? "Generating..." : "Generate Sample"}
          </Button>
          <Button variant="secondary" onClick={onClose} disabled={isBusy}>
            Close
          </Button>
        </div>
      </div>
    </ModalBase>
  );
}
