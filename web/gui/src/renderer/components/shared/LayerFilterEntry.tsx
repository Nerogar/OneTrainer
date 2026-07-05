import { type ChangeEvent, useEffect, useState } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { EMPTY_PRESETS, LAYER_PRESETS_BY_MODEL, type LayerPresetMap } from "@/types/generated/layerPresets";
import { INPUT_FULL, PLACEHOLDER, SELECT_BASE } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";
import { Toggle } from "./Toggle";

export interface LayerFilterEntryProps {
  filterPath: string;
  presetPath: string;
  regexPath: string;
  tooltip?: string;
}

const CUSTOM_KEY = "custom";

export function LayerFilterEntry({ filterPath, presetPath, regexPath, tooltip }: LayerFilterEntryProps) {
  const [filterValue, setFilterValue] = useConfigField<string>(filterPath);
  const [presetValue, setPresetValue] = useConfigField<string>(presetPath);
  const [, setRegexValue] = useConfigField<boolean>(regexPath);
  const [modelType] = useConfigField<string>("model_type");

  const resolvedMap = modelType ? LAYER_PRESETS_BY_MODEL[modelType as keyof typeof LAYER_PRESETS_BY_MODEL] : undefined;
  const presetMap: LayerPresetMap = resolvedMap ?? EMPTY_PRESETS;
  const presetKeys = Object.keys(presetMap);
  const selected =
    presetValue && (presetKeys.includes(presetValue) || presetValue === CUSTOM_KEY)
      ? presetValue
      : (presetKeys[0] ?? CUSTOM_KEY);
  const isCustom = selected === CUSTOM_KEY;
  const selectedDef = isCustom ? undefined : presetMap[selected];
  const inputHidden = !isCustom && (selectedDef?.patterns.length ?? 0) === 0;

  const [localFilter, setLocalFilter] = useState(filterValue ?? "");
  useEffect(() => {
    setLocalFilter(filterValue ?? "");
  }, [filterValue]);

  const handlePresetChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const key = e.target.value;
    setPresetValue(key);
    if (key === CUSTOM_KEY) return;
    const def = presetMap[key];
    if (!def) return;
    const joined = def.patterns.join(",");
    setFilterValue(joined);
    setRegexValue(def.regex);
    setLocalFilter(joined);
  };

  const handleFilterChange = (e: ChangeEvent<HTMLInputElement>) => {
    setLocalFilter(e.target.value);
    setFilterValue(e.target.value);
  };

  return (
    <FormFieldWrapper label="Layer Filter" tooltip={tooltip} configPath={presetPath} className="flex flex-col gap-2">
      <div className="flex gap-2 items-center">
        <select value={selected} onChange={handlePresetChange} className={SELECT_BASE}>
          {presetKeys.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
          <option value={CUSTOM_KEY}>custom</option>
        </select>
        {isCustom && <Toggle configPath={regexPath} label="Regex" labelPosition="right" />}
      </div>
      {!inputHidden && (
        <input
          type="text"
          value={localFilter}
          onChange={handleFilterChange}
          placeholder="Comma-separated layer names or regex..."
          disabled={!isCustom}
          className={`${INPUT_FULL} ${PLACEHOLDER}`}
        />
      )}
    </FormFieldWrapper>
  );
}
