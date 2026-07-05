import { Dice5 } from "lucide-react";

import { Button, FilePicker, FormEntry, Select, Toggle } from "@/components/shared";
import { useConfigField } from "@/hooks/useConfigField";
import { NoiseSchedulerValues } from "@/types/generated/enums";
import { MODEL_TYPE_GROUPS } from "@/types/generated/modelTypeInfo";
import { TEXTAREA_FULL } from "@/utils/inputStyles";

import { ModalBase } from "./ModalBase";

export interface SampleParamsModalProps {
  open: boolean;
  onClose: () => void;
  sampleIndex: number;
}

export function SampleParamsModal({ open, onClose, sampleIndex }: SampleParamsModalProps) {
  const p = `samples.${sampleIndex}`;
  const [prompt, setPrompt] = useConfigField<string>(`${p}.prompt`);
  const [negativePrompt, setNegativePrompt] = useConfigField<string>(`${p}.negative_prompt`);
  const [, setSeed] = useConfigField<number>(`${p}.seed`);
  const [modelType] = useConfigField<string>("model_type");
  const isFlowMatching = modelType
    ? (MODEL_TYPE_GROUPS.is_flow_matching as readonly string[]).includes(modelType)
    : false;

  const rerollSeed = () => setSeed(Math.floor(Math.random() * 2 ** 30));

  return (
    <ModalBase open={open} onClose={onClose} title={`Sample ${sampleIndex + 1} Parameters`} size="xl">
      <div className="flex flex-col gap-3">
        <div>
          <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">Prompt</label>
          <textarea
            value={prompt ?? ""}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter the prompt to sample with..."
            className={TEXTAREA_FULL}
            rows={4}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">Negative Prompt</label>
          <textarea
            value={negativePrompt ?? ""}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="(Optional) Enter what to avoid..."
            className={TEXTAREA_FULL}
            rows={2}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <FormEntry label="Width" configPath={`${p}.width`} type="number" />
        <FormEntry label="Height" configPath={`${p}.height`} type="number" />
        <div className="flex items-end gap-2">
          <FormEntry label="Seed" configPath={`${p}.seed`} type="number" />
          <Button variant="secondary" size="sm" onClick={rerollSeed} title="Reroll seed">
            <Dice5 className="w-4 h-4" />
          </Button>
        </div>
        <Toggle configPath={`${p}.random_seed`} label="Random Seed" />
        <FormEntry label="Diffusion Steps" configPath={`${p}.diffusion_steps`} type="number" />
        <FormEntry label="CFG Scale" configPath={`${p}.cfg_scale`} type="number" />
        {!isFlowMatching && (
          <Select label="Noise Scheduler" configPath={`${p}.noise_scheduler`} options={[...NoiseSchedulerValues]} />
        )}

        <FormEntry label="Frames" configPath={`${p}.frames`} type="number" tooltip="Frame count for video models" />
        <FormEntry
          label="Audio Length (s)"
          configPath={`${p}.length`}
          type="number"
          tooltip="Length in seconds for audio models"
        />

        <FormEntry label="TE1 Layer Skip" configPath={`${p}.text_encoder_1_layer_skip`} type="number" />
        <FormEntry label="TE1 Seq Length" configPath={`${p}.text_encoder_1_sequence_length`} type="number" nullable />
        <FormEntry label="TE2 Layer Skip" configPath={`${p}.text_encoder_2_layer_skip`} type="number" />
        <FormEntry label="TE2 Seq Length" configPath={`${p}.text_encoder_2_sequence_length`} type="number" nullable />
        <FormEntry label="TE3 Layer Skip" configPath={`${p}.text_encoder_3_layer_skip`} type="number" />
        <FormEntry label="TE4 Layer Skip" configPath={`${p}.text_encoder_4_layer_skip`} type="number" />

        <Toggle configPath={`${p}.transformer_attention_mask`} label="Attention Mask" />
        <Toggle configPath={`${p}.force_last_timestep`} label="Force Last Timestep" />
        <Toggle configPath={`${p}.sample_inpainting`} label="Sample Inpainting" />
      </div>

      <div className="grid grid-cols-1 gap-4 mt-4">
        <FilePicker label="Base Image" configPath={`${p}.base_image_path`} tooltip="Base image for inpainting" />
        <FilePicker label="Mask Image" configPath={`${p}.mask_image_path`} tooltip="Mask image for inpainting" />
      </div>

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
