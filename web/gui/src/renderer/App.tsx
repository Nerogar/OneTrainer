import { lazy, Suspense, useEffect } from "react";

import { configApi } from "./api/configApi";
import { ErrorBoundary } from "./components/ErrorBoundary";
import BottomBar from "./components/layout/BottomBar";
import TabNavigation from "./components/layout/TabNavigation";
import TerminalPanel from "./components/layout/TerminalPanel";
import TopBar from "./components/layout/TopBar";
import { SchemaTabRenderer } from "./components/schema/SchemaTabRenderer";
import { useTrainingWebSocket } from "./hooks/useTrainingWebSocket";
import { useConfigStore } from "./store/configStore";
import { useTrainingStore } from "./store/trainingStore";
import { useUiSchemaStore } from "./store/uiSchemaStore";
import { useUiStore } from "./store/uiStore";

const ConceptsPage = lazy(() => import("./pages/ConceptsPage"));
const SamplingPage = lazy(() => import("./pages/SamplingPage"));
const EmbeddingsPage = lazy(() => import("./pages/EmbeddingsPage"));
const ToolsPage = lazy(() => import("./pages/ToolsPage"));
const PerformancePage = lazy(() => import("./pages/PerformancePage"));
const RunPage = lazy(() => import("./pages/RunPage"));
const HelpPage = lazy(() => import("./pages/HelpPage"));

const CUSTOM_TAB_COMPONENTS: Record<string, React.LazyExoticComponent<React.ComponentType>> = {
  concepts: ConceptsPage,
  sampling: SamplingPage,
  additionalembeddings: EmbeddingsPage,
  tools: ToolsPage,
  performance: PerformancePage,
  run: RunPage,
  help: HelpPage,
};

function TabContent() {
  const activeTab = useUiStore((s) => s.activeTab);
  const schema = useUiSchemaStore((s) => s.schema);
  const tabDef = schema?.tabs.find((t) => t.id === activeTab);

  const CustomComponent = CUSTOM_TAB_COMPONENTS[activeTab];
  if (CustomComponent || tabDef?.renderer === "custom") {
    if (CustomComponent) return <CustomComponent />;
    return null;
  }

  if (tabDef && tabDef.renderer === "schema") {
    const hasSections =
      (tabDef.sections && tabDef.sections.length > 0) ||
      (tabDef.sectionDefs && Object.keys(tabDef.sectionDefs).length > 0);

    if (hasSections) {
      return <SchemaTabRenderer tab={tabDef} />;
    }
  }

  return (
    <div className="card" style={{ padding: "24px" }}>
      <h2 style={{ margin: 0 }}>{activeTab}</h2>
      <p style={{ color: "var(--color-on-surface-secondary)", marginTop: "8px" }}>This tab will be implemented soon.</p>
    </div>
  );
}

export default function App() {
  const activeTab = useUiStore((s) => s.activeTab);
  const setBackendConnected = useUiStore((s) => s.setBackendConnected);
  const backendConnected = useUiStore((s) => s.backendConnected);
  const terminalOpen = useUiStore((s) => s.terminalOpen);
  const loadConfig = useConfigStore((s) => s.loadConfig);
  const flushPendingChanges = useConfigStore((s) => s.flushPendingChanges);
  const fetchTrainingStatus = useTrainingStore((s) => s.fetchStatus);
  const loadSchema = useUiSchemaStore((s) => s.loadSchema);

  useEffect(() => {
    loadSchema();
  }, [loadSchema]);

  useTrainingWebSocket(backendConnected);

  useEffect(() => {
    let cancelled = false;
    const checkHealth = async () => {
      try {
        const data = await configApi.health();
        if (!cancelled && data.status === "ok") {
          setBackendConnected(true);
        }
      } catch {
        if (!cancelled) setBackendConnected(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [setBackendConnected]);

  useEffect(() => {
    if (!backendConnected) return;
    const init = async () => {
      await loadConfig();
      await fetchTrainingStatus();
    };
    init().catch((err) => {
      console.error("App initialization failed:", err);
    });
  }, [backendConnected, loadConfig, fetchTrainingStatus]);

  useEffect(() => {
    const api = window.electronAPI;
    if (!api) return;

    return api.onFlushRequest(async (requestId) => {
      try {
        await flushPendingChanges();
      } catch (err) {
        console.error("[App] flush failed:", err);
      } finally {
        api.signalFlushComplete(requestId);
      }
    });
  }, [flushPendingChanges]);

  return (
    <div className="app-shell">
      <TopBar />
      <TabNavigation />
      <main className="tab-content" role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
        <div className="tab-content-inner">
          <Suspense
            fallback={<div className="skeleton" style={{ height: 200 }} aria-busy="true" aria-label="Loading page" />}
          >
            <ErrorBoundary>
              <TabContent />
            </ErrorBoundary>
          </Suspense>
        </div>
      </main>
      {terminalOpen && <TerminalPanel isOpen={terminalOpen} backendConnected={backendConnected} />}
      <BottomBar />
    </div>
  );
}
