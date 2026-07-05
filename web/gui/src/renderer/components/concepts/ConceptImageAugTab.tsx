import { FormEntry, SliderEntry, Toggle } from "@/components/shared";
import type { ConceptConfig, ConceptImageConfig } from "@/types/generated/config";

import { ImagePreviewPanel } from "./ImagePreviewPanel";

export interface ConceptImageAugTabProps {
  draft: ConceptConfig;
  updateImage: (field: keyof ConceptImageConfig, value: unknown) => void;
}

interface AugRow {
  label: string;
  random?: keyof ConceptImageConfig;
  fixed?: keyof ConceptImageConfig;
  value?: keyof ConceptImageConfig;
  valuePlaceholder?: string;
  valueType?: "number" | "text";
  slider?: { min: number; max: number; step: number };
}

const AUG_ROWS: AugRow[] = [
  { label: "Crop Jitter", random: "enable_crop_jitter" },
  { label: "Random Flip", random: "enable_random_flip", fixed: "enable_fixed_flip" },
  {
    label: "Random Rotation",
    random: "enable_random_rotate",
    fixed: "enable_fixed_rotate",
    value: "random_rotate_max_angle",
    valuePlaceholder: "Max angle",
    valueType: "number",
    slider: { min: 0, max: 360, step: 1 },
  },
  {
    label: "Random Brightness",
    random: "enable_random_brightness",
    fixed: "enable_fixed_brightness",
    value: "random_brightness_max_strength",
    valuePlaceholder: "Max strength",
    valueType: "number",
    slider: { min: 0, max: 1, step: 0.01 },
  },
  {
    label: "Random Contrast",
    random: "enable_random_contrast",
    fixed: "enable_fixed_contrast",
    value: "random_contrast_max_strength",
    valuePlaceholder: "Max strength",
    valueType: "number",
    slider: { min: 0, max: 1, step: 0.01 },
  },
  {
    label: "Random Saturation",
    random: "enable_random_saturation",
    fixed: "enable_fixed_saturation",
    value: "random_saturation_max_strength",
    valuePlaceholder: "Max strength",
    valueType: "number",
    slider: { min: 0, max: 1, step: 0.01 },
  },
  {
    label: "Random Hue",
    random: "enable_random_hue",
    fixed: "enable_fixed_hue",
    value: "random_hue_max_strength",
    valuePlaceholder: "Max strength",
    valueType: "number",
    slider: { min: 0, max: 1, step: 0.01 },
  },
  { label: "Circular Mask Generation", random: "enable_random_circular_mask_shrink" },
  { label: "Random Rotate & Crop", random: "enable_random_mask_rotate_crop" },
  {
    label: "Resolution Override",
    fixed: "enable_resolution_override",
    value: "resolution_override",
    valuePlaceholder: "e.g. 512 or 768x512",
    valueType: "text",
  },
];

export function ConceptImageAugTab({ draft, updateImage }: ConceptImageAugTabProps) {
  return (
    <div className="flex gap-6">
      <div className="flex-1 min-w-0 flex flex-col gap-2">
        <div className="grid grid-cols-[minmax(160px,auto)_48px_48px_1fr] gap-x-3 gap-y-0 items-center text-xs font-semibold text-[var(--color-on-surface-secondary)] uppercase">
          <span>Augmentation</span>
          <span className="text-center">Random</span>
          <span className="text-center">Fixed</span>
          <span>Value</span>
        </div>

        {AUG_ROWS.map((row) => {
          const { random, fixed, value } = row;
          return (
            <div
              key={row.label}
              className="grid grid-cols-[minmax(160px,auto)_48px_48px_1fr] gap-x-3 items-center py-1"
            >
              <span className="text-sm font-medium text-[var(--color-on-surface)]">{row.label}</span>

              <div className="flex justify-center">
                {random ? (
                  <Toggle value={draft.image[random] as boolean} onChange={(v) => updateImage(random, v)} />
                ) : (
                  <span />
                )}
              </div>

              <div className="flex justify-center">
                {fixed ? (
                  <Toggle value={draft.image[fixed] as boolean} onChange={(v) => updateImage(fixed, v)} />
                ) : (
                  <span />
                )}
              </div>

              <div>
                {value && row.slider ? (
                  <SliderEntry
                    label=""
                    value={draft.image[value] as number}
                    onChange={(v) => updateImage(value, v)}
                    min={row.slider.min}
                    max={row.slider.max}
                    step={row.slider.step}
                  />
                ) : value ? (
                  <FormEntry
                    label=""
                    type={row.valueType ?? "text"}
                    value={draft.image[value] as string | number}
                    onChange={(v) => updateImage(value, v)}
                    placeholder={row.valuePlaceholder}
                  />
                ) : (
                  <span />
                )}
              </div>
            </div>
          );
        })}
      </div>

      <ImagePreviewPanel
        conceptPath={draft.path}
        includeSubdirectories={draft.include_subdirectories}
        textConfig={draft.text}
        imageConfig={draft.image}
      />
    </div>
  );
}
