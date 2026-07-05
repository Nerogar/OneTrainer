import { Plus } from "lucide-react";

import { Button, EmptyState } from "@/components/shared";
import type { TrainEmbeddingConfig } from "@/types/generated/config";

import { EmbeddingCard } from "./EmbeddingCard";

export interface EmbeddingListProps {
  embeddings: TrainEmbeddingConfig[];
  onAdd: () => void;
  onClone: (index: number) => void;
  onRemove: (index: number) => void;
  onToggleTrain: (index: number, value: boolean) => void;
}

export function EmbeddingList({ embeddings, onAdd, onClone, onRemove, onToggleTrain }: EmbeddingListProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <Button variant="primary" size="sm" onClick={onAdd}>
          <Plus className="w-4 h-4" /> Add Embedding
        </Button>
        <span className="text-xs text-[var(--color-on-surface-secondary)] ml-auto">
          {embeddings.length} embedding{embeddings.length === 1 ? "" : "s"}
        </span>
      </div>

      {embeddings.length === 0 ? (
        <EmptyState title="No additional embeddings" description="Add an embedding to train it alongside your model." />
      ) : (
        <div className="flex flex-col gap-3">
          {embeddings.map((emb, idx) => (
            <EmbeddingCard
              key={emb.uuid || idx}
              embedding={emb}
              index={idx}
              onClone={onClone}
              onRemove={onRemove}
              onToggleTrain={onToggleTrain}
            />
          ))}
        </div>
      )}
    </div>
  );
}
