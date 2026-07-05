import { Plus } from "lucide-react";

import { Button, EmptyState } from "@/components/shared";
import type { SampleConfig } from "@/types/generated/config";

import { SampleCard } from "./SampleCard";

export interface SampleListProps {
  samples: SampleConfig[];
  onAdd: () => void;
  onOpen: (index: number) => void;
  onClone: (index: number) => void;
  onRemove: (index: number) => void;
  onToggle: (index: number, enabled: boolean) => void;
}

export function SampleList({ samples, onAdd, onOpen, onClone, onRemove, onToggle }: SampleListProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <Button variant="primary" size="sm" onClick={onAdd}>
          <Plus className="w-4 h-4" /> Add Sample
        </Button>
        <span className="text-xs text-[var(--color-on-surface-secondary)] ml-auto">
          {samples.length} sample{samples.length === 1 ? "" : "s"}
        </span>
      </div>

      {samples.length === 0 ? (
        <EmptyState title="No samples yet" description="Add a sample to generate previews during training." />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {samples.map((sample, idx) => (
            <SampleCard
              key={idx}
              sample={sample}
              index={idx}
              onOpen={onOpen}
              onClone={onClone}
              onRemove={onRemove}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}
