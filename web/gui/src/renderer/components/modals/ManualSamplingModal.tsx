import { useEffect, useRef, useState } from "react";

import { trainingApi } from "@/api/trainingApi";
import { Button, FormEntry, Select, Toggle } from "@/components/shared";
import { useTrainingStore } from "@/store/trainingStore";
import { NoiseSchedulerValues } from "@/types/generated/enums";
import { TEXTAREA_FULL } from "@/utils/inputStyles";

import { ModalBase } from "./ModalBase";

export interface ManualSamplingModalProps {
  open: boolean;
  onClose: () => void;
}

interface ManualSampleState {
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

const DEFAULT_SAMPLE: ManualSampleState = {
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

export function ManualSamplingModal({ open, onClose }: ManualSamplingModalProps) {
  const [sample, setSample] = useState<ManualSampleState>({ ...DEFAULT_SAMPLE });
  const [isSampling, setIsSampling] = useState(false);
  const [sampleImage, setSampleImage] = useState<string | null>(null);
  const waitingRef = useRef(false);
  const latestSample = useTrainingStore((s) => s.latestSample);

  useEffect(() => {
    if (waitingRef.current && latestSample) {
      setSampleImage(latestSample);
      waitingRef.current = false;
    }
  }, [latestSample]);

  const update = <K extends keyof ManualSampleState>(field: K, value: ManualSampleState[K]) => {
    setSample((prev) => ({ ...prev, [field]: value }));
  };

  const handleSample = async () => {
    setIsSampling(true);
    waitingRef.current = true;
    try {
      await trainingApi.sampleCustom({
        prompt: sample.prompt,
        negative_prompt: sample.negative_prompt,
        height: sample.height,
        width: sample.width,
        seed: sample.seed,
        random_seed: sample.random_seed,
        diffusion_steps: sample.diffusion_steps,
        cfg_scale: sample.cfg_scale,
      });
    } catch {
      waitingRef.current = false;
    } finally {
      setIsSampling(false);
    }
  };

  return (
    <ModalBase open={open} onClose={onClose} title="Manual Sample" size="xl" closeOnBackdrop={false}>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Prompt</span>
            <textarea
              value={sample.prompt}
              onChange={(e) => update("prompt", e.target.value)}
              rows={3}
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
              className={`${TEXTAREA_FULL} resize-y`}
              placeholder="Enter negative prompt..."
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <FormEntry label="Width" value={sample.width} onChange={(v) => update("width", Number(v))} type="number" />
            <FormEntry
              label="Height"
              value={sample.height}
              onChange={(v) => update("height", Number(v))}
              type="number"
            />
            <FormEntry
              label="Steps"
              value={sample.diffusion_steps}
              onChange={(v) => update("diffusion_steps", Number(v))}
              type="number"
            />
            <FormEntry
              label="CFG Scale"
              value={sample.cfg_scale}
              onChange={(v) => update("cfg_scale", Number(v))}
              type="number"
            />
            <Select
              label="Scheduler"
              options={[...NoiseSchedulerValues]}
              value={sample.noise_scheduler}
              onChange={(v) => update("noise_scheduler", v)}
            />
            <FormEntry label="Seed" value={sample.seed} onChange={(v) => update("seed", Number(v))} type="number" />
            <Toggle label="Random Seed" value={sample.random_seed} onChange={(v) => update("random_seed", v)} />
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="aspect-square bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] rounded-[var(--radius-sm)] flex items-center justify-center overflow-hidden">
            {sampleImage ? (
              <img src={sampleImage} alt="Generated sample" className="max-w-full max-h-full object-contain" />
            ) : (
              <span className="text-sm text-[var(--color-on-surface-secondary)]">
                {isSampling ? "Generating..." : "No sample generated yet"}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex justify-between mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="primary" onClick={handleSample} disabled={isSampling}>
          {isSampling ? "Sampling..." : "Sample"}
        </Button>
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
