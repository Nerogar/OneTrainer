import { type ChangeEvent, useEffect, useState } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import type { TimeUnit } from "@/types/generated/enums";
import { TimeUnitValues } from "@/types/generated/enums";
import { INPUT_FLEX, SELECT_BASE } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface TimeEntryProps {
  label: string;
  valuePath: string;
  unitPath: string;
  tooltip?: string;
  disabled?: boolean;
}

export function TimeEntry({ label, valuePath, unitPath, tooltip, disabled }: TimeEntryProps) {
  const [numValue, setNumValue] = useConfigField<number>(valuePath);
  const [unitValue, setUnitValue] = useConfigField<TimeUnit>(unitPath);
  const [localNum, setLocalNum] = useState(String(numValue ?? ""));

  useEffect(() => {
    setLocalNum(numValue != null ? String(numValue) : "");
  }, [numValue]);

  const handleNumChange = (e: ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setLocalNum(raw);
    const parsed = Number(raw);
    if (!isNaN(parsed)) setNumValue(parsed);
  };

  const handleUnitChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setUnitValue(e.target.value as TimeUnit);
  };

  return (
    <FormFieldWrapper label={label} tooltip={tooltip} configPath={valuePath}>
      <div className="flex gap-1">
        <input type="number" value={localNum} onChange={handleNumChange} disabled={disabled} className={INPUT_FLEX} />
        <select value={unitValue ?? "NEVER"} onChange={handleUnitChange} disabled={disabled} className={SELECT_BASE}>
          {TimeUnitValues.map((u) => (
            <option key={u} value={u}>
              {u}
            </option>
          ))}
        </select>
      </div>
    </FormFieldWrapper>
  );
}
