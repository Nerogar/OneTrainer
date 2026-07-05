import type { ReactNode } from "react";

import { FormEntry, Select, Toggle } from "@/components/shared";
import { NoiseSchedulerValues } from "@/types/generated/enums";
import { TEXTAREA_FULL } from "@/utils/inputStyles";

export interface SampleState {
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

export const DEFAULT_SAMPLE: SampleState = {
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

export type SampleUpdater = <K extends keyof SampleState>(field: K, value: SampleState[K]) => void;

export interface SampleParamsFormProps {
  state: SampleState;
  onChange: SampleUpdater;
  sampleImage: string | null;
  isSampling: boolean;
  placeholder: ReactNode;
  disabled?: boolean;
}

export function SampleParamsForm({
  state,
  onChange,
  sampleImage,
  isSampling,
  placeholder,
  disabled = false,
}: SampleParamsFormProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-[var(--color-on-surface)]">Prompt</span>
          <textarea
            value={state.prompt}
            onChange={(e) => onChange("prompt", e.target.value)}
            rows={3}
            disabled={disabled}
            className={`${TEXTAREA_FULL} resize-y`}
            placeholder="Enter prompt..."
          />
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-[var(--color-on-surface)]">Negative Prompt</span>
          <textarea
            value={state.negative_prompt}
            onChange={(e) => onChange("negative_prompt", e.target.value)}
            rows={2}
            disabled={disabled}
            className={`${TEXTAREA_FULL} resize-y`}
            placeholder="Enter negative prompt..."
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <FormEntry
            label="Width"
            value={state.width}
            onChange={(v) => onChange("width", Number(v))}
            type="number"
            disabled={disabled}
          />
          <FormEntry
            label="Height"
            value={state.height}
            onChange={(v) => onChange("height", Number(v))}
            type="number"
            disabled={disabled}
          />
          <FormEntry
            label="Steps"
            value={state.diffusion_steps}
            onChange={(v) => onChange("diffusion_steps", Number(v))}
            type="number"
            disabled={disabled}
          />
          <FormEntry
            label="CFG Scale"
            value={state.cfg_scale}
            onChange={(v) => onChange("cfg_scale", Number(v))}
            type="number"
            disabled={disabled}
          />
          <Select
            label="Scheduler"
            options={[...NoiseSchedulerValues]}
            value={state.noise_scheduler}
            onChange={(v) => onChange("noise_scheduler", v)}
            disabled={disabled}
          />
          <FormEntry
            label="Seed"
            value={state.seed}
            onChange={(v) => onChange("seed", Number(v))}
            type="number"
            disabled={disabled}
          />
          <Toggle
            label="Random Seed"
            value={state.random_seed}
            onChange={(v) => onChange("random_seed", v)}
            disabled={disabled}
          />
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="aspect-square bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] rounded-[var(--radius-sm)] flex items-center justify-center overflow-hidden">
          {sampleImage ? (
            <img src={sampleImage} alt="Generated sample" className="max-w-full max-h-full object-contain" />
          ) : (
            <span className="text-sm text-[var(--color-on-surface-secondary)]">
              {isSampling ? "Generating..." : placeholder}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
