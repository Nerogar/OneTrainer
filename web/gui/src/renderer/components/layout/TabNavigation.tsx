import { useCallback, useMemo, useRef } from "react";

import { useConfigStore } from "@/store/configStore";
import { useUiSchemaStore } from "@/store/uiSchemaStore";
import { type TabId, useUiStore } from "@/store/uiStore";
import { evaluateCondition } from "@/utils/conditionEval";

/** Hardcoded fallback tabs used when schema hasn't loaded yet. */
const FALLBACK_TABS: Array<{ id: TabId; label: string }> = [
  { id: "general", label: "General" },
  { id: "model", label: "Model" },
  { id: "data", label: "Data" },
  { id: "concepts", label: "Concepts" },
  { id: "training", label: "Training" },
  { id: "sampling", label: "Sampling" },
  { id: "backup", label: "Backup" },
  { id: "tools", label: "Tools" },
  { id: "lora", label: "LoRA" },
  { id: "embedding", label: "Embedding" },
  { id: "additionalembeddings", label: "Additional Embeddings" },
  { id: "cloud", label: "Cloud" },
  { id: "performance", label: "Performance" },
  { id: "run", label: "Run" },
  { id: "help", label: "Help" },
];

export default function TabNavigation() {
  const { activeTab, setActiveTab } = useUiStore();
  const schema = useUiSchemaStore((s) => s.schema);
  const config = useConfigStore((s) => s.config);
  const navRef = useRef<HTMLElement>(null);

  const configRecord = useMemo(() => (config ?? {}) as Record<string, unknown>, [config]);
  const predicates = useMemo(() => schema?.predicates ?? {}, [schema]);

  const visibleTabs = useMemo(() => {
    // Use schema tabs if available, otherwise fallback
    const tabList = schema?.tabs
      ? schema.tabs.map((t) => ({ id: t.id as TabId, label: t.label, visibility: t.visibility }))
      : FALLBACK_TABS.map((t) => ({ ...t, visibility: undefined }));

    return tabList.filter((t) => evaluateCondition(t.visibility, configRecord, predicates));
  }, [schema, configRecord, predicates]);

  const focusTab = useCallback(
    (tabId: TabId) => {
      setActiveTab(tabId);
      const el = navRef.current?.querySelector<HTMLElement>(`#tab-${tabId}`);
      el?.focus();
    },
    [setActiveTab],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLElement>) => {
      const currentIndex = visibleTabs.findIndex((t) => t.id === activeTab);
      if (currentIndex === -1) return;

      let nextIndex: number | null = null;

      switch (e.key) {
        case "ArrowRight":
          nextIndex = (currentIndex + 1) % visibleTabs.length;
          break;
        case "ArrowLeft":
          nextIndex = (currentIndex - 1 + visibleTabs.length) % visibleTabs.length;
          break;
        case "Home":
          nextIndex = 0;
          break;
        case "End":
          nextIndex = visibleTabs.length - 1;
          break;
        default:
          return;
      }

      e.preventDefault();
      focusTab(visibleTabs[nextIndex].id);
    },
    [visibleTabs, activeTab, focusTab],
  );

  return (
    <nav className="tab-nav" role="tablist" ref={navRef} onKeyDown={handleKeyDown}>
      {visibleTabs.map((tab) => (
        <button
          key={tab.id}
          id={`tab-${tab.id}`}
          role="tab"
          aria-selected={activeTab === tab.id}
          aria-controls={`tabpanel-${tab.id}`}
          tabIndex={activeTab === tab.id ? 0 : -1}
          className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
          onClick={() => setActiveTab(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
