import { useCallback, useMemo } from "react";

import { EmbeddingList } from "@/components/embeddings/EmbeddingList";
import { SectionCard } from "@/components/shared";
import { useConfigField } from "@/hooks/useConfigField";
import type { TrainEmbeddingConfig } from "@/types/generated/config";

const DEFAULT_EMBEDDING: Omit<TrainEmbeddingConfig, "uuid"> = {
  model_name: "",
  placeholder: "<embedding>",
  train: true,
  stop_training_after: null,
  stop_training_after_unit: "NEVER",
  token_count: null,
  initial_embedding_text: "",
  is_output_embedding: false,
};

function newUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 10)}`;
}

const newEmbedding = (): TrainEmbeddingConfig => ({
  ...JSON.parse(JSON.stringify(DEFAULT_EMBEDDING)),
  uuid: newUuid(),
});

export default function EmbeddingsPage() {
  const [embeddings, setEmbeddings] = useConfigField<TrainEmbeddingConfig[] | null>("additional_embeddings");
  const list = useMemo<TrainEmbeddingConfig[]>(() => embeddings ?? [], [embeddings]);

  const handleAdd = useCallback(() => {
    setEmbeddings([...list, newEmbedding()]);
  }, [list, setEmbeddings]);

  const handleRemove = useCallback(
    (index: number) => {
      setEmbeddings(list.filter((_, i) => i !== index));
    },
    [list, setEmbeddings],
  );

  const handleClone = useCallback(
    (index: number) => {
      const cloned = JSON.parse(JSON.stringify(list[index])) as TrainEmbeddingConfig;
      cloned.uuid = newUuid();
      const next = [...list];
      next.splice(index + 1, 0, cloned);
      setEmbeddings(next);
    },
    [list, setEmbeddings],
  );

  const handleToggleTrain = useCallback(
    (index: number, value: boolean) => {
      setEmbeddings(list.map((e, i) => (i === index ? { ...e, train: value } : e)));
    },
    [list, setEmbeddings],
  );

  return (
    <SectionCard title="Additional Embeddings">
      <EmbeddingList
        embeddings={list}
        onAdd={handleAdd}
        onClone={handleClone}
        onRemove={handleRemove}
        onToggleTrain={handleToggleTrain}
      />
    </SectionCard>
  );
}
