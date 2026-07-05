import { type ChangeEvent, forwardRef, useEffect, useRef, useState } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { INPUT_FULL, PLACEHOLDER } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface FormEntryProps {
  label: string;
  configPath?: string;
  value?: string | number | null;
  onChange?: (value: string | number) => void;
  type?: "text" | "number";
  placeholder?: string;
  tooltip?: string;
  disabled?: boolean;
  nullable?: boolean;
  width?: string;
}

export const FormEntry = forwardRef<HTMLInputElement, FormEntryProps>(
  (
    {
      label,
      configPath,
      value: controlledValue,
      onChange,
      type = "text",
      placeholder,
      tooltip,
      disabled,
      nullable,
      width,
    },
    ref,
  ) => {
    const [configValue, setConfigValue] = useConfigField<string | number | null>(configPath);

    const externalValue = configPath ? configValue : controlledValue;
    const [localValue, setLocalValue] = useState<string>(externalValue != null ? String(externalValue) : "");
    const isFocusedRef = useRef(false);

    useEffect(() => {
      // Don't clobber what the user is actively typing. Pushing a parsed number
      // to the config makes it flow back here as `externalValue`; re-syncing
      // mid-type would rewrite the raw text (e.g. "-0" -> "0", "1e-4" -> "0.0001").
      // External-driven updates (preset load, reset, config switch) still apply
      // when the field isn't focused.
      if (isFocusedRef.current) return;
      setLocalValue(externalValue != null ? String(externalValue) : "");
    }, [externalValue]);

    const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
      const raw = e.target.value;
      setLocalValue(raw);
      if (raw === "" && nullable) {
        if (configPath && setConfigValue) setConfigValue(null as unknown as string | number);
        return;
      }
      if (type === "number") {
        // Don't push an empty field as 0 — let the user keep typing.
        // The config will be updated once a valid number is entered.
        if (raw === "" || raw === "-") return;
        const num = Number(raw);
        if (isNaN(num)) return;
        if (configPath && setConfigValue) setConfigValue(num);
        if (onChange) onChange(num);
      } else {
        if (configPath && setConfigValue) setConfigValue(raw);
        if (onChange) onChange(raw);
      }
    };

    const handleFocus = () => {
      isFocusedRef.current = true;
    };

    const handleBlur = () => {
      isFocusedRef.current = false;
      // Validate/normalize on blur: snap the display to the canonical stored
      // value (e.g. "007" -> "7", "5." -> "5"). Incomplete entries like "-" or
      // "" that were never pushed simply revert to the last valid value.
      setLocalValue(externalValue != null ? String(externalValue) : "");
    };

    return (
      <FormFieldWrapper label={label} tooltip={tooltip} configPath={configPath} style={width ? { width } : undefined}>
        <input
          ref={ref}
          type={type}
          value={localValue}
          onChange={handleChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          disabled={disabled}
          className={`${INPUT_FULL} ${PLACEHOLDER}`}
        />
      </FormFieldWrapper>
    );
  },
);
FormEntry.displayName = "FormEntry";
