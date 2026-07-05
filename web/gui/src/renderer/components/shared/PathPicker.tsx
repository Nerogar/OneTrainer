import { FolderOpen } from "lucide-react";
import { type ChangeEvent, type MouseEvent, useEffect, useState } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { INPUT_FLEX, SIDE_BUTTON } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface PathPickerProps {
  label: string;
  mode: "file" | "directory";
  configPath?: string;
  value?: string;
  onChange?: (value: string) => void;
  filters?: Array<{ name: string; extensions: string[] }>;
  tooltip?: string;
  disabled?: boolean;
}

export function PathPicker({
  label,
  mode,
  configPath,
  value: controlledValue,
  onChange,
  filters,
  tooltip,
  disabled,
}: PathPickerProps) {
  const [configValue, setConfigValue] = useConfigField<string>(configPath);

  const externalValue = configPath ? (configValue ?? "") : (controlledValue ?? "");
  const [localValue, setLocalValue] = useState(externalValue);

  useEffect(() => {
    setLocalValue(externalValue);
  }, [externalValue]);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setLocalValue(val);
    if (configPath && setConfigValue) setConfigValue(val);
    if (onChange) onChange(val);
  };

  const handleBrowse = async (e: MouseEvent<HTMLButtonElement>) => {
    // Stop the click from propagating to any ancestor label/form that might
    // swallow it — this bug recurred with the FormFieldWrapper <label> and
    // defensively guarding it here keeps the picker working even if a caller
    // wraps it in a new label layer later.
    e.preventDefault();
    e.stopPropagation();
    const api = window.electronAPI;
    if (!api) {
      console.warn("[PathPicker] window.electronAPI is unavailable; browse aborted.");
      return;
    }
    const result = mode === "file" ? await api.openFile(filters) : await api.openDirectory();
    if (result) {
      setLocalValue(result);
      if (configPath && setConfigValue) setConfigValue(result);
      if (onChange) onChange(result);
    }
  };

  return (
    <FormFieldWrapper label={label} tooltip={tooltip} configPath={configPath}>
      <div className="flex gap-1">
        <input type="text" value={localValue} onChange={handleChange} disabled={disabled} className={INPUT_FLEX} />
        <button
          type="button"
          onClick={handleBrowse}
          disabled={disabled}
          className={SIDE_BUTTON}
          aria-label={mode === "file" ? "Browse files" : "Browse directory"}
        >
          <FolderOpen className="w-4 h-4" />
        </button>
      </div>
    </FormFieldWrapper>
  );
}
