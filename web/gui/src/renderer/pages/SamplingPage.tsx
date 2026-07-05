import { useCallback, useMemo, useState } from "react";

import { SampleParamsModal } from "@/components/modals/SampleParamsModal";
import { SampleList } from "@/components/sampling/SampleList";
import { FormEntry, SectionCard, Select, Toggle } from "@/components/shared";
import { useConfigField } from "@/hooks/useConfigField";
import type { SampleConfig } from "@/types/generated/config";
import { ImageFormatValues, TimeUnitValues } from "@/types/generated/enums";

const DEFAULT_SAMPLE: SampleConfig = {
  enabled: true,
  prompt: "",
  negative_prompt: "",
  height: 512,
  width: 512,
  frames: 1,
  length: 1,
  seed: 42,
  random_seed: false,
  diffusion_steps: 20,
  cfg_scale: 7.0,
  noise_scheduler: "DDIM",
  text_encoder_1_layer_skip: 0,
  text_encoder_1_sequence_length: null,
  text_encoder_2_layer_skip: 0,
  text_encoder_2_sequence_length: null,
  text_encoder_3_layer_skip: 0,
  text_encoder_4_layer_skip: 0,
  transformer_attention_mask: false,
  force_last_timestep: false,
  sample_inpainting: false,
  base_image_path: "",
  mask_image_path: "",
};

const newSample = (): SampleConfig => ({
  ...JSON.parse(JSON.stringify(DEFAULT_SAMPLE)),
  seed: Math.floor(Math.random() * 2 ** 30),
});

export default function SamplingPage() {
  const [samples, setSamples] = useConfigField<SampleConfig[] | null>("samples");
  const list = useMemo<SampleConfig[]>(() => samples ?? [], [samples]);
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number>(0);

  const handleAdd = useCallback(() => {
    setSamples([...list, newSample()]);
  }, [list, setSamples]);

  const handleRemove = useCallback(
    (index: number) => {
      setSamples(list.filter((_, i) => i !== index));
    },
    [list, setSamples],
  );

  const handleClone = useCallback(
    (index: number) => {
      const cloned = JSON.parse(JSON.stringify(list[index])) as SampleConfig;
      cloned.seed = Math.floor(Math.random() * 2 ** 30);
      const next = [...list];
      next.splice(index + 1, 0, cloned);
      setSamples(next);
    },
    [list, setSamples],
  );

  const handleToggle = useCallback(
    (index: number, enabled: boolean) => {
      setSamples(list.map((s, i) => (i === index ? { ...s, enabled } : s)));
    },
    [list, setSamples],
  );

  const handleOpen = useCallback((index: number) => {
    setEditingIndex(index);
    setEditorOpen(true);
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <SectionCard title="Sampling Settings">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormEntry
            label="Sample After"
            configPath="sample_after"
            type="number"
            tooltip="Interval between automatic samples during training"
          />
          <Select label="Sample After Unit" configPath="sample_after_unit" options={[...TimeUnitValues]} />
          <FormEntry
            label="Skip First"
            configPath="sample_skip_first"
            type="number"
            tooltip="Wait this long before the first auto-sample"
          />
          <Select label="Image Format" configPath="sample_image_format" options={[...ImageFormatValues]} />
          <Toggle
            configPath="non_ema_sampling"
            label="Non-EMA Sampling"
            tooltip="Include non-EMA sampling alongside EMA samples"
          />
          <Toggle
            configPath="samples_to_tensorboard"
            label="Samples to Tensorboard"
            tooltip="Include sample images in Tensorboard"
          />
        </div>
      </SectionCard>

      <SectionCard title="Sample Prompts">
        <SampleList
          samples={list}
          onAdd={handleAdd}
          onOpen={handleOpen}
          onClone={handleClone}
          onRemove={handleRemove}
          onToggle={handleToggle}
        />
      </SectionCard>

      {editorOpen && (
        <SampleParamsModal open={editorOpen} onClose={() => setEditorOpen(false)} sampleIndex={editingIndex} />
      )}
    </div>
  );
}
