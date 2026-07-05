import { Pencil } from "lucide-react";

import { ArrayItemHeader, Toggle } from "@/components/shared";
import type { SampleConfig } from "@/types/generated/config";

export interface SampleCardProps {
  sample: SampleConfig;
  index: number;
  onOpen: (index: number) => void;
  onClone: (index: number) => void;
  onRemove: (index: number) => void;
  onToggle: (index: number, enabled: boolean) => void;
}

export function SampleCard({ sample, index, onOpen, onClone, onRemove, onToggle }: SampleCardProps) {
  const promptPreview = sample.prompt?.trim() ? (
    sample.prompt.length > 80 ? (
      `${sample.prompt.slice(0, 80)}…`
    ) : (
      sample.prompt
    )
  ) : (
    <span className="italic opacity-60">(no prompt)</span>
  );

  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] p-3">
      <ArrayItemHeader title={`Sample ${index + 1}`} onClone={() => onClone(index)} onRemove={() => onRemove(index)}>
        <Toggle value={sample.enabled} onChange={(v) => onToggle(index, v)} />
      </ArrayItemHeader>

      <div className="text-sm text-[var(--color-on-surface)] mb-2 break-words">{promptPreview}</div>
      {sample.negative_prompt?.trim() && (
        <div className="text-xs text-[var(--color-on-surface-secondary)] mb-2 break-words">
          neg: {sample.negative_prompt.length > 60 ? `${sample.negative_prompt.slice(0, 60)}…` : sample.negative_prompt}
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-[var(--color-on-surface-secondary)]">
        <span>
          {sample.width}×{sample.height} · seed {sample.random_seed ? "random" : sample.seed} · cfg {sample.cfg_scale} ·{" "}
          {sample.diffusion_steps} steps
        </span>
        <button
          onClick={() => onOpen(index)}
          className="flex items-center gap-1 text-[var(--color-cobalt-600)] hover:underline cursor-pointer"
        >
          <Pencil className="w-3.5 h-3.5" /> Edit
        </button>
      </div>
    </div>
  );
}
