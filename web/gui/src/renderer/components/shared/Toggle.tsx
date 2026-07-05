import { useConfigField } from "@/hooks/useConfigField";
import { getTooltip } from "@/utils/tooltips";

import { Tooltip } from "./Tooltip";

export interface ToggleProps {
  label?: string;
  labelPosition?: "left" | "right";
  configPath?: string;
  value?: boolean;
  onChange?: (value: boolean) => void;
  disabled?: boolean;
  tooltip?: string;
}

export function Toggle({
  label,
  labelPosition = "right",
  configPath,
  value,
  onChange,
  disabled,
  tooltip,
}: ToggleProps) {
  const resolvedTooltip = tooltip ?? (configPath ? getTooltip(configPath) : undefined);
  const [configValue, setConfigValue] = useConfigField<boolean>(configPath);
  const checked = configPath ? (configValue ?? false) : (value ?? false);
  const handleChange = () => {
    const next = !checked;
    if (configPath && setConfigValue) setConfigValue(next);
    if (onChange) onChange(next);
  };

  const switchEl = (
    <button
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={handleChange}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-[background,box-shadow,transform] duration-200 ease-out cursor-pointer hover:scale-105
        disabled:opacity-40 disabled:cursor-not-allowed
        ${checked ? "" : "bg-[var(--color-border-subtle)]"}`}
      style={{
        background: checked ? "linear-gradient(135deg, var(--color-cobalt-600), var(--color-azure-500))" : undefined,
        boxShadow: checked ? "0 0 8px var(--color-cobalt-600-alpha-25)" : "inset 0 1px 3px rgba(0,0,0,0.15)",
      }}
    >
      <span
        className={`inline-block h-4 w-4 rounded-full bg-white transition-transform duration-200 ease-out
          ${checked ? "translate-x-[17px]" : "translate-x-[3px]"}`}
        style={{ boxShadow: "0 1px 3px rgba(0,0,0,0.15)" }}
      />
    </button>
  );

  const content = label ? (
    <label className="inline-flex items-center gap-3 cursor-pointer text-sm text-[var(--color-on-surface)]">
      {labelPosition === "left" && <span>{label}</span>}
      {switchEl}
      {labelPosition === "right" && <span>{label}</span>}
    </label>
  ) : (
    switchEl
  );

  if (resolvedTooltip) return <Tooltip content={resolvedTooltip}>{content}</Tooltip>;
  return content;
}
