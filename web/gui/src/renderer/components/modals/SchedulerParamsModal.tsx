import { Plus, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Button, FormEntry } from "@/components/shared";
import { useConfigField } from "@/hooks/useConfigField";

import { ModalBase } from "./ModalBase";

interface SchedulerParam {
  key: string;
  value: string;
}

export interface SchedulerParamsModalProps {
  open: boolean;
  onClose: () => void;
}

export function SchedulerParamsModal({ open, onClose }: SchedulerParamsModalProps) {
  const [storedParams, setStoredParams] = useConfigField<Array<Record<string, string>> | null>("scheduler_params");
  const [scheduler] = useConfigField<string>("learning_rate_scheduler");
  const isCustom = scheduler === "CUSTOM";
  const [params, setParams] = useState<SchedulerParam[]>([]);

  useEffect(() => {
    if (open && storedParams) {
      const entries = storedParams.flatMap((obj) => Object.entries(obj).map(([key, value]) => ({ key, value })));
      setParams(entries.length > 0 ? entries : [{ key: "", value: "" }]);
    } else if (open) {
      setParams([{ key: "", value: "" }]);
    }
  }, [open, storedParams]);

  const addParam = () => setParams([...params, { key: "", value: "" }]);

  const removeParam = (index: number) => {
    const next = params.filter((_, i) => i !== index);
    setParams(next.length > 0 ? next : [{ key: "", value: "" }]);
  };

  const updateParam = (index: number, field: "key" | "value", val: string) => {
    setParams(params.map((p, i) => (i === index ? { ...p, [field]: val } : p)));
  };

  const handleSave = () => {
    const filtered = params.filter((p) => p.key.trim() !== "");
    const result = filtered.map((p) => ({ [p.key]: p.value }));
    setStoredParams(result.length > 0 ? result : null);
    onClose();
  };

  const variables = ["%LR%", "%EPOCHS%", "%STEPS_PER_EPOCH%", "%TOTAL_STEPS%", "%SCHEDULER_STEPS%"];

  return (
    <ModalBase open={open} onClose={onClose} title="Scheduler Parameters" size="md">
      {isCustom && (
        <div className="mb-4">
          <FormEntry
            label="Custom Scheduler Class"
            configPath="custom_learning_rate_scheduler"
            tooltip="Python class path for a custom learning rate scheduler"
          />
        </div>
      )}

      <div className="mb-4 p-3 rounded-[var(--radius-sm)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)]">
        <p className="text-xs font-medium text-[var(--color-on-surface-secondary)] mb-1">Available Variables:</p>
        <div className="flex flex-wrap gap-2">
          {variables.map((v) => (
            <code
              key={v}
              className="text-xs px-2 py-0.5 rounded bg-[var(--color-surface)] text-[var(--color-cobalt-600)]"
            >
              {v}
            </code>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <div className="grid grid-cols-[1fr_1fr_40px] gap-2 text-xs font-medium text-[var(--color-on-surface-secondary)] uppercase tracking-wide">
          <span>Key</span>
          <span>Value</span>
          <span></span>
        </div>
        {params.map((param, i) => (
          <div key={i} className="grid grid-cols-[1fr_1fr_40px] gap-2">
            <input
              value={param.key}
              onChange={(e) => updateParam(i, "key", e.target.value)}
              placeholder="parameter_name"
              className="px-3 py-2 rounded-[var(--radius-sm)] text-sm bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] text-[var(--color-on-surface)] focus:outline-none focus:ring-2 focus:ring-[var(--color-cobalt-600)]"
            />
            <input
              value={param.value}
              onChange={(e) => updateParam(i, "value", e.target.value)}
              placeholder="value"
              className="px-3 py-2 rounded-[var(--radius-sm)] text-sm bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] text-[var(--color-on-surface)] focus:outline-none focus:ring-2 focus:ring-[var(--color-cobalt-600)]"
            />
            <button
              onClick={() => removeParam(i)}
              className="flex items-center justify-center text-[var(--color-error-500)] hover:opacity-80 cursor-pointer"
              aria-label="Remove parameter"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>

      <button
        onClick={addParam}
        className="mt-3 flex items-center gap-1 text-sm text-[var(--color-cobalt-600)] hover:text-[var(--color-cobalt-600)] cursor-pointer"
      >
        <Plus className="w-4 h-4" /> Add Parameter
      </button>

      <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" size="sm" onClick={onClose}>
          Cancel
        </Button>
        <Button variant="primary" size="sm" onClick={handleSave}>
          Save
        </Button>
      </div>
    </ModalBase>
  );
}
