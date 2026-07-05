import { FilterX, Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/shared";
import type { ConceptConfig } from "@/types/generated/config";

import { ConceptCard } from "./ConceptCard";

export interface ConceptGridProps {
  concepts: ConceptConfig[];
  onAdd: () => void;
  onOpen: (index: number) => void;
  onRemove: (index: number) => void;
  onClone: (index: number) => void;
  onToggle: (index: number, enabled: boolean) => void;
}

export function ConceptGrid({ concepts, onAdd, onOpen, onRemove, onClone, onToggle }: ConceptGridProps) {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("ALL");
  const [showDisabled, setShowDisabled] = useState(true);

  const disabledCount = concepts.filter((c) => !c.enabled).length;

  const filtered = useMemo(() => {
    return concepts
      .map((c, i) => ({ concept: c, originalIndex: i }))
      .filter(({ concept }) => {
        if (!showDisabled && !concept.enabled) return false;
        if (typeFilter !== "ALL" && concept.type !== typeFilter) return false;
        if (search) {
          const q = search.toLowerCase();
          const name = (concept.name || "").toLowerCase();
          const path = (concept.path || "").toLowerCase();
          if (!name.includes(q) && !path.includes(q)) return false;
        }
        return true;
      });
  }, [concepts, search, typeFilter, showDisabled]);

  const clearFilters = () => {
    setSearch("");
    setTypeFilter("ALL");
    setShowDisabled(true);
  };
  const hasFilters = search !== "" || typeFilter !== "ALL" || !showDisabled;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" size="sm" onClick={onAdd}>
          <Plus className="w-4 h-4" /> Add Concept
        </Button>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)]">
          <Search className="w-4 h-4 text-[var(--color-on-surface-secondary)]" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search concepts..."
            className="bg-transparent text-sm text-[var(--color-on-surface)] outline-none w-48"
          />
        </div>

        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="px-3 py-1.5 rounded-[var(--radius-sm)] text-sm bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] text-[var(--color-on-surface)] cursor-pointer"
        >
          <option value="ALL">All Types</option>
          <option value="STANDARD">Standard</option>
          <option value="VALIDATION">Validation</option>
          <option value="PRIOR_PREDICTION">Prior Prediction</option>
        </select>

        <label className="flex items-center gap-2 text-sm text-[var(--color-on-surface-secondary)] cursor-pointer">
          <input
            type="checkbox"
            checked={showDisabled}
            onChange={(e) => setShowDisabled(e.target.checked)}
            className="accent-[var(--color-cobalt-600)]"
          />
          Show Disabled{disabledCount > 0 ? ` (${disabledCount})` : ""}
        </label>

        {hasFilters && (
          <button
            onClick={clearFilters}
            className="flex items-center gap-1 text-sm text-[var(--color-cobalt-600)] hover:text-[var(--color-cobalt-600)] cursor-pointer"
          >
            <FilterX className="w-4 h-4" /> Clear
          </button>
        )}

        <span className="ml-auto text-xs text-[var(--color-on-surface-secondary)]">
          {filtered.length} / {concepts.length} concepts
        </span>
      </div>

      {filtered.length > 0 ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4">
          {filtered.map(({ concept, originalIndex }) => (
            <ConceptCard
              key={originalIndex}
              concept={concept}
              index={originalIndex}
              onOpen={() => onOpen(originalIndex)}
              onRemove={() => onRemove(originalIndex)}
              onClone={() => onClone(originalIndex)}
              onToggle={(enabled) => onToggle(originalIndex, enabled)}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-16 text-[var(--color-on-surface-secondary)]">
          <p className="text-sm">
            {concepts.length === 0 ? "No concepts yet. Add one to get started." : "No concepts match current filters."}
          </p>
        </div>
      )}
    </div>
  );
}
