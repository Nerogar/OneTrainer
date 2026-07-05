import { Copy, ImageIcon, X } from "lucide-react";
import { useEffect, useState } from "react";

import { API_BASE } from "@/api/request";
import { Toggle } from "@/components/shared";
import type { ConceptConfig } from "@/types/generated/config";
import type { ConceptType } from "@/types/generated/enums";

export interface ConceptCardProps {
  concept: ConceptConfig;
  index: number;
  onOpen: () => void;
  onRemove: () => void;
  onClone: () => void;
  onToggle: (enabled: boolean) => void;
}

/** Color config per concept type: border, badge background, badge text, overlay tint */
const typeStyles: Record<
  ConceptType,
  {
    border: string;
    badgeBg: string;
    badgeText: string;
    overlay: string;
    label: string;
  }
> = {
  STANDARD: {
    border: "var(--color-cobalt-600)",
    badgeBg: "var(--color-cobalt-600-alpha-15)",
    badgeText: "var(--color-cobalt-600)",
    overlay: "var(--color-cobalt-600-alpha-06)",
    label: "Standard",
  },
  VALIDATION: {
    border: "var(--color-success-500)",
    badgeBg: "var(--color-success-500-alpha-15)",
    badgeText: "var(--color-success-500)",
    overlay: "var(--color-success-500-alpha-06)",
    label: "Validation",
  },
  PRIOR_PREDICTION: {
    border: "var(--color-warning-500)",
    badgeBg: "var(--color-warning-500-alpha-15)",
    badgeText: "var(--color-warning-500)",
    overlay: "var(--color-warning-500-alpha-06)",
    label: "Prior",
  },
};

export function ConceptCard({ concept, index, onOpen, onRemove, onClone, onToggle }: ConceptCardProps) {
  const displayName = concept.name || (concept.path ? concept.path.split(/[/\\]/).pop() : `Concept ${index + 1}`);
  const style = typeStyles[concept.type] ?? typeStyles.STANDARD;

  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);
  const [thumbnailError, setThumbnailError] = useState(false);

  useEffect(() => {
    if (!concept.path) {
      setThumbnailUrl(null);
      setThumbnailError(false);
      return;
    }
    const url = `${API_BASE}/concepts/thumbnail?path=${encodeURIComponent(concept.path)}&include_subdirectories=${concept.include_subdirectories ?? false}`;
    setThumbnailUrl(url);
    setThumbnailError(false);
  }, [concept.path, concept.include_subdirectories]);

  return (
    <div
      className="card relative group flex flex-col min-w-[220px] max-w-[280px] flex-1 cursor-pointer transition-transform duration-200 hover:translate-y-[-2px]"
      style={{
        opacity: concept.enabled ? 1 : 0.5,
        borderLeft: `3px solid ${style.border}`,
      }}
    >
      <div
        className="relative h-[140px] flex items-center justify-center rounded-tr-[var(--radius-sm)] bg-[var(--color-surface-raised)] overflow-hidden"
        style={{ background: style.overlay }}
        onClick={onOpen}
      >
        {thumbnailUrl && !thumbnailError ? (
          <img
            src={thumbnailUrl}
            alt={displayName}
            className="w-full h-full object-cover"
            onError={() => setThumbnailError(true)}
          />
        ) : (
          <ImageIcon className="w-12 h-12 text-[var(--color-on-surface-secondary)]" strokeWidth={1} />
        )}
      </div>

      <div className="px-3 py-2.5 flex flex-col gap-1.5">
        <span
          className="text-sm font-medium text-[var(--color-on-surface)] break-words leading-snug"
          title={displayName}
        >
          {displayName}
        </span>
        <span
          className="inline-flex self-start px-2 py-0.5 rounded-[var(--radius-full)] text-[10px] font-semibold uppercase tracking-wider"
          style={{
            backgroundColor: style.badgeBg,
            color: style.badgeText,
          }}
        >
          {style.label}
        </span>
      </div>

      <div className="absolute top-1 left-1 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className="w-6 h-6 flex items-center justify-center rounded-[var(--radius-sm)] bg-[var(--color-error-500)] text-white text-xs hover:brightness-110 cursor-pointer"
          aria-label="Remove concept"
        >
          <X className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onClone();
          }}
          className="w-6 h-6 flex items-center justify-center rounded-[var(--radius-sm)] bg-[var(--color-success-500)] text-white text-xs hover:brightness-110 cursor-pointer"
          aria-label="Clone concept"
        >
          <Copy className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="absolute top-1 right-1" onClick={(e) => e.stopPropagation()}>
        <Toggle value={concept.enabled} onChange={onToggle} />
      </div>
    </div>
  );
}
