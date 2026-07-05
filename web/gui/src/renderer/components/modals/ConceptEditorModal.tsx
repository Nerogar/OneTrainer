import { produce } from "immer";
import { useEffect, useState } from "react";

import { ConceptGeneralTab } from "@/components/concepts/ConceptGeneralTab";
import { ConceptImageAugTab } from "@/components/concepts/ConceptImageAugTab";
import { ConceptStatsPanel } from "@/components/concepts/ConceptStatsPanel";
import { ConceptTextAugTab } from "@/components/concepts/ConceptTextAugTab";
import { Button } from "@/components/shared";
import type { ConceptConfig } from "@/types/generated/config";

import { ModalBase } from "./ModalBase";

export interface ConceptEditorModalProps {
  open: boolean;
  onClose: () => void;
  concept: ConceptConfig | null;
  onSave: (updated: ConceptConfig) => void;
}

type Tab = "general" | "image_aug" | "text_aug" | "statistics";

const tabs: Array<{ id: Tab; label: string }> = [
  { id: "general", label: "General" },
  { id: "image_aug", label: "Image Augmentation" },
  { id: "text_aug", label: "Text Augmentation" },
  { id: "statistics", label: "Statistics" },
];

export function ConceptEditorModal({ open, onClose, concept, onSave }: ConceptEditorModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>("general");
  const [draft, setDraft] = useState<ConceptConfig | null>(null);

  useEffect(() => {
    if (open && concept) {
      setDraft(JSON.parse(JSON.stringify(concept)));
      setActiveTab("general");
    }
  }, [open, concept]);

  if (!draft) return null;

  const update = (path: string, value: unknown) => {
    setDraft((prev) => {
      if (!prev) return prev;
      return produce(prev, (d) => {
        const keys = path.split(".");
        let obj: Record<string, unknown> = d as unknown as Record<string, unknown>;
        for (let i = 0; i < keys.length - 1; i++) {
          obj = obj[keys[i]] as Record<string, unknown>;
        }
        obj[keys[keys.length - 1]] = value;
      });
    });
  };

  const updateImage = (field: keyof ConceptConfig["image"], value: unknown) => update(`image.${field}`, value);
  const updateText = (field: keyof ConceptConfig["text"], value: unknown) => update(`text.${field}`, value);

  const handleSave = () => {
    if (draft) {
      onSave(draft);
      onClose();
    }
  };

  // Use wider modal for image augmentation tab (needs preview panel space)
  const modalSize = activeTab === "image_aug" ? ("2xl" as const) : ("xl" as const);

  return (
    <ModalBase
      open={open}
      onClose={onClose}
      title={`Edit: ${draft.name || "Concept"}`}
      size={modalSize}
      closeOnBackdrop={false}
    >
      <div className="flex gap-1 mb-4 border-b border-[var(--color-border-subtle)]">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium transition-colors cursor-pointer ${
              activeTab === tab.id
                ? "text-[var(--color-cobalt-600)] border-b-2 border-[var(--color-cobalt-600)]"
                : "text-[var(--color-on-surface-secondary)] hover:text-[var(--color-on-surface)]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="min-h-[400px] max-h-[60vh] overflow-y-auto">
        {activeTab === "general" && <ConceptGeneralTab draft={draft} update={update} updateText={updateText} />}
        {activeTab === "image_aug" && <ConceptImageAugTab draft={draft} updateImage={updateImage} />}
        {activeTab === "text_aug" && <ConceptTextAugTab draft={draft} updateText={updateText} />}
        {activeTab === "statistics" && (
          <ConceptStatsPanel conceptPath={draft.path} includeSubdirectories={draft.include_subdirectories} />
        )}
      </div>

      <div className="flex justify-end gap-3 mt-4 pt-4 border-t border-[var(--color-border-subtle)]">
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
