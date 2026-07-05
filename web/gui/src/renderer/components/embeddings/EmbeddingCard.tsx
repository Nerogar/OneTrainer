import { ArrayItemHeader, FormEntry, PathPicker, Toggle } from "@/components/shared";
import type { TrainEmbeddingConfig } from "@/types/generated/config";

export interface EmbeddingCardProps {
  embedding: TrainEmbeddingConfig;
  index: number;
  onClone: (index: number) => void;
  onRemove: (index: number) => void;
  onToggleTrain: (index: number, value: boolean) => void;
}

export function EmbeddingCard({ embedding, index, onClone, onRemove, onToggleTrain }: EmbeddingCardProps) {
  const p = `additional_embeddings.${index}`;

  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] p-4">
      <ArrayItemHeader
        title={embedding.placeholder?.trim() || `Embedding ${index + 1}`}
        onClone={() => onClone(index)}
        onRemove={() => onRemove(index)}
      >
        <Toggle value={embedding.train} onChange={(v) => onToggleTrain(index, v)} />
      </ArrayItemHeader>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <PathPicker
          label="Base Embedding"
          mode="file"
          configPath={`${p}.model_name`}
          tooltip="The base embedding to train on. Leave empty to create a new embedding."
        />
        <FormEntry
          label="Placeholder"
          configPath={`${p}.placeholder`}
          tooltip="The placeholder used when using the embedding in a prompt."
        />
        <FormEntry
          label="Token Count"
          configPath={`${p}.token_count`}
          type="number"
          nullable
          tooltip="Token count for new embedding. Leave empty to auto-detect from the initial text."
        />
        <FormEntry
          label="Initial Embedding Text"
          configPath={`${p}.initial_embedding_text`}
          tooltip="The initial embedding text used when creating a new embedding."
        />
        <Toggle
          configPath={`${p}.is_output_embedding`}
          label="Output Embedding"
          tooltip="Output embeddings are calculated at the output of the text encoder, not the input. Better for larger TEs and lower VRAM."
        />
        <FormEntry
          label="Stop Training After"
          configPath={`${p}.stop_training_after`}
          type="number"
          nullable
          tooltip="When to stop training the embedding."
        />
      </div>
    </div>
  );
}
