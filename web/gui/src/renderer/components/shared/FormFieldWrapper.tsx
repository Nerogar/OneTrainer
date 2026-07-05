import { Info } from "lucide-react";
import { type CSSProperties, type ReactNode } from "react";

import { getTooltip } from "@/utils/tooltips";

import { Tooltip } from "./Tooltip";

export interface FormFieldWrapperProps {
  label?: string;
  tooltip?: string;
  configPath?: string;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}

export function FormFieldWrapper({ label, tooltip, configPath, className, style, children }: FormFieldWrapperProps) {
  const resolvedTooltip = tooltip ?? (configPath ? getTooltip(configPath) : undefined);

  const wrapperClass =
    className ??
    (label
      ? "flex flex-col gap-1.5 pl-3 border-l-2 border-transparent hover:border-[color-mix(in_srgb,var(--color-cobalt-600)_20%,transparent)] transition-[border-color] duration-300"
      : "flex flex-col gap-1");

  // Uses <div> rather than <label>: a <label> containing an <input> and a
  // <button> forwards any click inside it to the first labelable descendant
  // (the input), which on Electron/Chromium can swallow the button's onClick
  // and left PathPicker's browse button dead. The label text is still
  // readable and the tooltip still reachable; the (minor) loss is that
  // clicking the label text no longer focuses the input.
  if (!label) {
    return (
      <div className={wrapperClass} style={style}>
        {children}
      </div>
    );
  }

  return (
    <div className={wrapperClass} style={style}>
      <div className="flex items-center gap-1">
        <span className="text-sm font-medium text-[var(--color-on-surface)]">{label}</span>
        {resolvedTooltip && (
          <Tooltip content={resolvedTooltip}>
            <Info className="w-4 h-4 text-[var(--color-on-surface-secondary)] hover:text-[var(--color-cobalt-600)] transition-colors duration-200 cursor-help" />
          </Tooltip>
        )}
      </div>
      {children}
    </div>
  );
}
