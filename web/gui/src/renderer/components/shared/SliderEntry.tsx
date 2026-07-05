import { type ChangeEvent, useCallback, useEffect, useState } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { INPUT_BASE, PLACEHOLDER } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface SliderEntryProps {
  label: string;
  configPath?: string;
  value?: number;
  onChange?: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  tooltip?: string;
  disabled?: boolean;
}

export function SliderEntry({
  label,
  configPath,
  value: controlledValue,
  onChange,
  min,
  max,
  step = 0.01,
  tooltip,
  disabled,
}: SliderEntryProps) {
  const [configValue, setConfigValue] = useConfigField<number>(configPath);

  const externalValue = configPath ? configValue : controlledValue;
  const [localValue, setLocalValue] = useState<number>(externalValue ?? min);

  useEffect(() => {
    setLocalValue(externalValue ?? min);
  }, [externalValue, min]);

  const commit = useCallback(
    (v: number) => {
      const clamped = Math.min(max, Math.max(min, v));
      setLocalValue(clamped);
      if (configPath && setConfigValue) setConfigValue(clamped);
      if (onChange) onChange(clamped);
    },
    [configPath, setConfigValue, onChange, min, max],
  );

  const handleSlider = (e: ChangeEvent<HTMLInputElement>) => {
    commit(Number(e.target.value));
  };

  const handleInput = (e: ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    if (raw === "") {
      setLocalValue(min);
      return;
    }
    const parsed = Number(raw);
    if (!isNaN(parsed)) commit(parsed);
  };

  // Compute fill percentage for the slider track gradient.
  const pct = max > min ? ((localValue - min) / (max - min)) * 100 : 0;

  return (
    <FormFieldWrapper label={label} tooltip={tooltip} configPath={configPath}>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleSlider}
          disabled={disabled}
          className="flex-[2] h-2 rounded-full appearance-none cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--color-cobalt-600)] [&::-webkit-slider-thumb]:shadow-[0_1px_4px_var(--color-cobalt-600-alpha-30)] [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:duration-150 [&::-webkit-slider-thumb]:hover:scale-110 [&::-webkit-slider-thumb]:active:scale-95 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-[var(--color-cobalt-600)] [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:shadow-[0_1px_4px_var(--color-cobalt-600-alpha-30)] [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:transition-transform [&::-moz-range-thumb]:duration-150 [&::-moz-range-thumb]:hover:scale-110 [&::-moz-range-thumb]:active:scale-95"
          style={{
            background: `linear-gradient(to right, var(--color-cobalt-600) 0%, var(--color-cobalt-600) ${pct}%, var(--color-border-subtle) ${pct}%, var(--color-border-subtle) 100%)`,
          }}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleInput}
          disabled={disabled}
          className={`flex-1 min-w-[70px] text-center ${INPUT_BASE} ${PLACEHOLDER}`}
        />
      </div>
    </FormFieldWrapper>
  );
}
