import { MoreHorizontal } from "lucide-react";
import { type ChangeEvent, forwardRef } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { enumLabel } from "@/utils/enumLabels";
import { SELECT_FLEX, SELECT_FULL, SIDE_BUTTON } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface SelectKVOption {
  label: string;
  value: string;
}

export interface SelectProps {
  label: string;
  options: string[] | SelectKVOption[];
  configPath?: string;
  value?: string;
  onChange?: (value: string) => void;
  onAdvancedClick?: () => void;
  tooltip?: string;
  disabled?: boolean;
  /** Custom label formatter for string option display. Defaults to `enumLabel`. */
  formatLabel?: (value: string) => string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      label,
      options,
      configPath,
      value: controlledValue,
      onChange,
      onAdvancedClick,
      tooltip,
      disabled,
      formatLabel = enumLabel,
    },
    ref,
  ) => {
    const [configValue, setConfigValue] = useConfigField<string>(configPath);

    const currentValue = configPath ? (configValue ?? "") : (controlledValue ?? "");

    const handleChange = (e: ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      if (configPath && setConfigValue) setConfigValue(val);
      if (onChange) onChange(val);
    };

    const isKV = options.length > 0 && typeof options[0] !== "string";
    const selectClass = onAdvancedClick ? SELECT_FLEX : SELECT_FULL;

    const selectEl = (
      <select ref={ref} value={currentValue} onChange={handleChange} disabled={disabled} className={selectClass}>
        {isKV
          ? (options as SelectKVOption[]).map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))
          : (options as string[]).map((opt) => (
              <option key={opt} value={opt}>
                {formatLabel(opt)}
              </option>
            ))}
      </select>
    );

    return (
      <FormFieldWrapper label={label} tooltip={tooltip} configPath={configPath}>
        {onAdvancedClick ? (
          <div className="flex gap-1">
            {selectEl}
            <button
              type="button"
              onClick={onAdvancedClick}
              disabled={disabled}
              className={SIDE_BUTTON}
              aria-label="Advanced settings"
            >
              <MoreHorizontal className="w-4 h-4" />
            </button>
          </div>
        ) : (
          selectEl
        )}
      </FormFieldWrapper>
    );
  },
);
Select.displayName = "Select";
