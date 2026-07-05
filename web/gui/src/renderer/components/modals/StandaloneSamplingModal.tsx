import { useCallback, useEffect, useRef, useState } from "react";

import { type SamplerStatusResponse, samplingApi, type StandaloneSampleRequest } from "@/api/samplingApi";
import { Button, Card, ProgressBar } from "@/components/shared";

import { ModalBase } from "./ModalBase";
import { DEFAULT_SAMPLE, SampleParamsForm, type SampleState } from "./sampleForm";

export interface StandaloneSamplingModalProps {
  open: boolean;
  onClose: () => void;
}

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

      <SampleParamsForm
        state={sample}
        onChange={update}
        sampleImage={sampleImage}
        isSampling={isSampling}
        disabled={isBusy}
        placeholder={modelLoaded ? "Model ready -- click Generate to sample" : "Load a model to begin sampling"}
      />

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
