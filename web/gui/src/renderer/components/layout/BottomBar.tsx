import {
  BarChart3,
  Bug,
  Camera,
  Clock,
  Download,
  HardDrive,
  Loader2,
  Play,
  RefreshCw,
  Save,
  Square,
  Terminal,
  Timer,
} from "lucide-react";

import { configApi } from "@/api/configApi";
import { toolsApi } from "@/api/toolsApi";
import { DualProgress } from "@/components/shared";
import { useElapsedTime } from "@/hooks/useElapsedTime";
import { getByPath, useConfigStore } from "@/store/configStore";
import { useTrainingStore } from "@/store/trainingStore";
import { useUiStore } from "@/store/uiStore";
import { formatDuration } from "@/utils/formatDuration";

export default function BottomBar() {
  const exportConfig = useConfigStore((s) => s.exportConfig);
  const isDirty = useConfigStore((s) => s.isDirty);
  const loadedPresetName = useConfigStore((s) => s.loadedPresetName);
  const configError = useConfigStore((s) => s.error);
  const clearConfigError = useConfigStore((s) => s.clearError);

  const config = useConfigStore((s) => s.config);
  const cloudEnabled = config
    ? getByPath(config as unknown as Record<string, unknown>, "cloud.enabled") === true
    : false;

  const backendConnected = useUiStore((s) => s.backendConnected);
  const terminalOpen = useUiStore((s) => s.terminalOpen);
  const toggleTerminal = useUiStore((s) => s.toggleTerminal);
  const status = useTrainingStore((s) => s.status);
  const progress = useTrainingStore((s) => s.progress);
  const statusText = useTrainingStore((s) => s.statusText);
  const error = useTrainingStore((s) => s.error);
  const latestSample = useTrainingStore((s) => s.latestSample);
  const startTraining = useTrainingStore((s) => s.startTraining);
  const stopTraining = useTrainingStore((s) => s.stopTraining);
  const sampleNow = useTrainingStore((s) => s.sampleNow);
  const backupNow = useTrainingStore((s) => s.backupNow);
  const saveNow = useTrainingStore((s) => s.saveNow);

  const startTime = useTrainingStore((s) => s.startTime);

  const isActive = status === "training" || status === "preparing";
  const isStopping = statusText === "Stopping...";

  // Elapsed time ticker (updates every second during training)
  const elapsed = useElapsedTime(startTime, status === "training");

  const epochProgress = progress ? (progress.maxEpoch > 0 ? (progress.epoch / progress.maxEpoch) * 100 : 0) : 0;
  const stepProgress = progress ? (progress.maxStep > 0 ? (progress.step / progress.maxStep) * 100 : 0) : 0;

  const handleExport = async () => {
    try {
      const config = await exportConfig();
      const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "train_config.json";
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      /* handled by store */
    }
  };

  const handleDebugPackage = async () => {
    try {
      await toolsApi.downloadDebugPackage();
    } catch {
      /* surfaced via console by fetch failure; no dedicated error UI for this action */
    }
  };

  const handleLaunchTensorboard = async () => {
    try {
      const result = await configApi.tensorboardLaunch();
      if (result.ok && result.url) {
        window.open(result.url, "_blank", "noopener");
      } else {
        alert(`Tensorboard launch failed: ${result.error ?? "unknown error"}`);
      }
    } catch (e) {
      alert(`Tensorboard launch failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleTrainingButton = () => {
    if (isActive) {
      stopTraining();
    } else {
      startTraining();
    }
  };

  return (
    <footer className="bottom-bar">
      <div className="bottom-bar-left" role="status" aria-live="polite" aria-atomic="true">
        {isActive ? (
          <>
            <div className="progress-glow">
              <DualProgress
                epochProgress={epochProgress}
                stepProgress={stepProgress}
                epochLabel={progress ? `Epoch ${progress.epoch}/${progress.maxEpoch}` : "Epoch"}
                stepLabel={progress ? `Step ${progress.step}/${progress.maxStep}` : "Step"}
              />
            </div>
            <span className="text-sm text-[var(--color-on-surface-secondary)] ml-2">{statusText || "Training..."}</span>
            {status === "training" && elapsed > 0 && (
              <span className="flex items-center gap-3 ml-3 text-xs tabular-nums text-[var(--color-on-surface-secondary)]">
                <span className="flex items-center gap-1" title="Elapsed time">
                  <Clock className="w-3 h-3 opacity-60" />
                  {formatDuration(elapsed)}
                </span>
                {progress?.remainingTime != null && progress.remainingTime > 0 && (
                  <span className="flex items-center gap-1" title="Estimated time remaining">
                    <Timer className="w-3 h-3 opacity-60" />
                    ETA {formatDuration(progress.remainingTime)}
                  </span>
                )}
              </span>
            )}
          </>
        ) : status === "error" ? (
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--color-error-500)] inline-block" />
            <span className="text-sm font-medium text-[var(--color-error-500)]">
              {error || statusText || "Training error"}
            </span>
          </div>
        ) : configError ? (
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--color-error-500)] inline-block flex-shrink-0" />
            <span className="text-sm text-[var(--color-error-500)] truncate max-w-[500px]" title={configError}>
              {configError}
            </span>
            <button
              onClick={clearConfigError}
              className="text-xs text-[var(--color-on-surface-secondary)] hover:text-[var(--color-on-surface)] ml-1 flex-shrink-0"
              aria-label="Dismiss error"
            >
              Dismiss
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--color-success-500)] inline-block" />
            <span className="text-sm font-medium text-[var(--color-on-surface)]">Ready to Train</span>
            {loadedPresetName && (
              <span className="text-xs text-[var(--color-on-surface-secondary)] ml-2">
                {loadedPresetName.replace(/^#/, "")}
              </span>
            )}
            {isDirty && <span className="text-xs text-[var(--color-cobalt-600)] ml-1">(modified)</span>}
          </div>
        )}
      </div>

      <div className="bottom-bar-right">
        {isActive && (
          <>
            <button onClick={sampleNow} className="theme-toggle gap-1.5" aria-label="Take sample" title="Sample Now">
              <Camera className="w-4 h-4" />
              <span className="text-xs hidden lg:inline">Sample</span>
            </button>
            <button onClick={backupNow} className="theme-toggle gap-1.5" aria-label="Create backup" title="Backup Now">
              <HardDrive className="w-4 h-4" />
              <span className="text-xs hidden lg:inline">Backup</span>
            </button>
            <button onClick={saveNow} className="theme-toggle gap-1.5" aria-label="Save model" title="Save Now">
              <Save className="w-4 h-4" />
              <span className="text-xs hidden lg:inline">Save</span>
            </button>
          </>
        )}

        {latestSample && (
          <img
            src={latestSample}
            alt="Latest sample"
            className="h-8 w-8 rounded object-cover border border-[var(--color-border-subtle)]"
            title="Latest training sample"
          />
        )}

        <button onClick={handleExport} className="theme-toggle" aria-label="Export config" title="Export config">
          <Download className="w-4 h-4" />
        </button>
        <button
          onClick={handleLaunchTensorboard}
          className="theme-toggle"
          aria-label="Open Tensorboard in browser"
          title="Open Tensorboard"
        >
          <BarChart3 className="w-4 h-4" />
        </button>
        <button
          onClick={handleDebugPackage}
          className="theme-toggle"
          aria-label="Generate debug package"
          title="Generate a zip file with config.json, system info and logs -- use this to report bugs or issues"
        >
          <Bug className="w-4 h-4" />
        </button>
        <button
          className={`theme-toggle${terminalOpen ? " terminal-toggle-active" : ""}`}
          onClick={toggleTerminal}
          aria-label={terminalOpen ? "Hide terminal panel" : "Show terminal panel"}
          aria-pressed={terminalOpen}
          title={terminalOpen ? "Hide Terminal" : "Show Terminal"}
        >
          <Terminal className="w-4 h-4" />
        </button>

        {cloudEnabled && !isActive && (
          <button
            className="action-button"
            onClick={() => startTraining({ reattach: true })}
            disabled={!backendConnected}
            aria-label="Reattach to cloud training"
            title="Reattach to a detached cloud training run"
          >
            <RefreshCw className="w-4 h-4 inline mr-1" /> Reattach
          </button>
        )}

        <button
          className="action-button"
          onClick={handleTrainingButton}
          disabled={isStopping || (!isActive && !backendConnected)}
          aria-label={isStopping ? "Stopping training" : isActive ? "Stop training" : "Start training"}
        >
          {isStopping ? (
            <>
              <Loader2 className="w-4 h-4 inline mr-1 animate-spin" /> Stopping...
            </>
          ) : isActive ? (
            <>
              <Square className="w-4 h-4 inline mr-1" /> Stop Training
            </>
          ) : (
            <>
              <Play className="w-4 h-4 inline mr-1" /> Start Training
            </>
          )}
        </button>
      </div>
    </footer>
  );
}
