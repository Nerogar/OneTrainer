import { CheckIcon, CirclePlay, Pause, Play, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { configApi } from "@/api/configApi";
import { ProgressBar } from "@/components/shared";
import { ScalarChart } from "@/components/shared/ScalarChart";
import { useElapsedTime } from "@/hooks/useElapsedTime";
import { type TrainingStatus, useTrainingStore } from "@/store/trainingStore";
import { formatStep } from "@/utils/chartUtils";
import { formatDuration } from "@/utils/formatDuration";

interface ScalarPoint {
  wall_time: number;
  step: number;
  value: number;
}

interface TagData {
  tag: string;
  points: ScalarPoint[];
  lastStep: number;
}

const ACTIVE_POLL_MS = 2000;
const PREPARING_POLL_MS = 4000;
const IDLE_POLL_MS = 5000;

const TAG_REDETECT_INTERVAL = 10;

const STATUS_CLASSES: Record<TrainingStatus, { badge: string; label: string }> = {
  idle: {
    badge: "bg-[var(--color-border-subtle)] text-[var(--color-on-surface-secondary)]",
    label: "Idle",
  },
  preparing: {
    badge: "bg-[var(--color-warning-500-alpha-12)] text-[var(--color-warning-500)]",
    label: "Preparing",
  },
  training: {
    badge: "bg-[var(--color-cobalt-600-alpha-12)] text-[var(--color-cobalt-600)]",
    label: "Training",
  },
  error: {
    badge: "bg-[var(--color-error-500-alpha-12)] text-[var(--color-error-500)]",
    label: "Error",
  },
};

function getLineColor(tag: string): string {
  const lower = tag.toLowerCase();
  if (lower.includes("loss") || lower.includes("smooth_loss")) {
    return "var(--color-cobalt-600)";
  }
  if (lower.includes("lr") || lower.includes("learning_rate")) {
    return "var(--color-azure-500)";
  }
  return "var(--color-cobalt-600)";
}

function StatusBadge({ status }: { status: TrainingStatus }) {
  const { badge, label } = STATUS_CLASSES[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-label font-semibold tracking-wide uppercase ${badge}`}
    >
      {(status === "training" || status === "preparing") && (
        <span className="status-badge-pulse w-1.5 h-1.5 rounded-full inline-block" />
      )}
      {label}
    </span>
  );
}

export default function RunPage() {
  const status = useTrainingStore((s) => s.status);
  const progress = useTrainingStore((s) => s.progress);
  const statusText = useTrainingStore((s) => s.statusText);
  const latestSample = useTrainingStore((s) => s.latestSample);
  const sampleUrls = useTrainingStore((s) => s.sampleUrls);
  const error = useTrainingStore((s) => s.error);
  const startTime = useTrainingStore((s) => s.startTime);

  const [runs, setRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [tagData, setTagData] = useState<TagData[]>([]);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [trainingCompleted, setTrainingCompleted] = useState(false);

  const tagDataRef = useRef<TagData[]>([]);
  const selectedRunRef = useRef<string>("");
  const statusRef = useRef<TrainingStatus>(status);
  const userManuallySelectedRun = useRef(false);
  const pollCountRef = useRef(0);

  tagDataRef.current = tagData;
  selectedRunRef.current = selectedRun;
  statusRef.current = status;

  const prevStatusRef = useRef<TrainingStatus>(status);
  useEffect(() => {
    if (prevStatusRef.current === "training" && status === "idle") {
      setTrainingCompleted(true);
    } else if (status === "training" || status === "preparing") {
      setTrainingCompleted(false);
    }
    prevStatusRef.current = status;
  }, [status]);

  const elapsed = useElapsedTime(startTime, status === "training");

  const loadRunData = useCallback(async (runName: string) => {
    if (!runName) return;
    setLoading(true);
    setFetchError(null);
    try {
      const tags = await configApi.tensorboardTags(runName);
      const results = await Promise.all(
        tags.map(async (tag) => {
          const points = await configApi.tensorboardScalars(runName, tag);
          const lastStep = points.length > 0 ? points[points.length - 1].step : 0;
          return { tag, points, lastStep } as TagData;
        }),
      );
      setTagData(results);
      pollCountRef.current = 0;
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchIncremental = useCallback(async () => {
    const runName = selectedRunRef.current;
    if (!runName) return;

    const currentData = tagDataRef.current;

    pollCountRef.current += 1;
    if (currentData.length === 0 || pollCountRef.current >= TAG_REDETECT_INTERVAL) {
      pollCountRef.current = 0;
      await loadRunData(runName);
      return;
    }

    try {
      const updatedData = await Promise.all(
        currentData.map(async (td) => {
          const newPoints = await configApi.tensorboardScalars(runName, td.tag, td.lastStep);
          if (newPoints.length === 0) return td;
          const allPoints = [...td.points, ...newPoints];
          return {
            tag: td.tag,
            points: allPoints,
            lastStep: allPoints[allPoints.length - 1].step,
          } as TagData;
        }),
      );
      setTagData(updatedData);
    } catch {
      /* ignore */
    }
  }, [loadRunData]);

  const detectAndSwitchToLatest = useCallback(async () => {
    try {
      const freshRuns = await configApi.tensorboardRuns();
      setRuns(freshRuns);
      if (freshRuns.length > 0 && !userManuallySelectedRun.current) {
        const latest = freshRuns[0];
        if (latest !== selectedRunRef.current) {
          setSelectedRun(latest);
        }
      }
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    configApi
      .tensorboardRuns()
      .then((r) => {
        setRuns(r);
        if (r.length > 0 && !selectedRunRef.current) {
          setSelectedRun(r[0]);
        }
      })
      .catch((err) => setFetchError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    if (status === "training" || status === "preparing") {
      userManuallySelectedRun.current = false;
      detectAndSwitchToLatest();
    }
  }, [status, detectAndSwitchToLatest]);

  useEffect(() => {
    if (selectedRun) {
      loadRunData(selectedRun);
    } else {
      setTagData([]);
    }
  }, [selectedRun, loadRunData]);

  useEffect(() => {
    const isTraining = status === "training";
    const isPreparing = status === "preparing";

    if (isTraining) {
      const interval = setInterval(() => {
        detectAndSwitchToLatest();
        fetchIncremental();
      }, ACTIVE_POLL_MS);
      return () => clearInterval(interval);
    }

    if (isPreparing) {
      const interval = setInterval(detectAndSwitchToLatest, PREPARING_POLL_MS);
      return () => clearInterval(interval);
    }

    if (autoRefresh && selectedRun) {
      const interval = setInterval(fetchIncremental, IDLE_POLL_MS);
      return () => clearInterval(interval);
    }
  }, [status, autoRefresh, selectedRun, fetchIncremental, detectAndSwitchToLatest]);

  const isActive = status === "training" || status === "preparing";
  const hasData = tagData.length > 0 && tagData.some((td) => td.points.length > 0);
  const noRunsExist = runs.length === 0;

  const handleRunChange = (runName: string) => {
    userManuallySelectedRun.current = true;
    setSelectedRun(runName);
  };

  return (
    <div className="flex flex-col gap-6">
      <div
        className={`card card-static px-5 py-4 ${
          isActive ? "run-card-active" : trainingCompleted ? "border-[var(--color-success-500-alpha-20)]" : ""
        }`}
      >
        <div className="flex items-center gap-4 flex-wrap">
          <StatusBadge status={status} />

          {isActive && progress && (
            <>
              <div className="flex-1 min-w-[120px]">
                <ProgressBar value={progress.maxStep > 0 ? (progress.step / progress.maxStep) * 100 : 0} />
              </div>
              <div className="mono tabular-nums flex items-center gap-4 text-micro text-[var(--color-on-surface-secondary)] whitespace-nowrap">
                <span>
                  Step <span className="text-[var(--color-on-surface)] font-semibold">{formatStep(progress.step)}</span>
                  /{formatStep(progress.maxStep)}
                </span>
                <span>
                  Epoch <span className="text-[var(--color-on-surface)] font-semibold">{progress.epoch}</span>/
                  {progress.maxEpoch}
                </span>
                {startTime && status === "training" && elapsed > 0 && <span>{formatDuration(elapsed)}</span>}
              </div>
            </>
          )}

          {isActive && statusText && !progress && (
            <span className="text-caption text-[var(--color-on-surface-secondary)]">{statusText}</span>
          )}

          <div className="flex-1" />

          {runs.length > 0 && (
            <div className="flex items-center gap-2">
              <label
                htmlFor="run-select"
                className="text-caption font-semibold text-[var(--color-on-surface)] whitespace-nowrap"
              >
                Run
              </label>
              <select
                id="run-select"
                className="top-bar-select max-w-[400px] min-w-[200px]"
                value={selectedRun}
                onChange={(e) => handleRunChange(e.target.value)}
              >
                {runs.map((run) => (
                  <option key={run} value={run}>
                    {run}
                  </option>
                ))}
              </select>
            </div>
          )}

          {isActive ? (
            <span className="text-label text-[var(--color-success-500)] inline-flex items-center gap-1">
              <span className="status-badge-pulse w-1.5 h-1.5 rounded-full inline-block" />
              Live
            </span>
          ) : (
            <div className="flex items-center gap-2">
              <button
                className="theme-toggle py-[5px] px-3 text-xs font-medium inline-flex items-center gap-1.5"
                onClick={() => setAutoRefresh(!autoRefresh)}
                title={autoRefresh ? "Pause auto-refresh" : "Resume auto-refresh"}
              >
                {autoRefresh ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                {autoRefresh ? "Live" : "Paused"}
              </button>

              <button
                className="theme-toggle py-[5px] px-3 text-xs font-medium"
                onClick={() => loadRunData(selectedRun)}
                title="Refresh data"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            </div>
          )}
        </div>

        {trainingCompleted && (
          <div className="mt-3 px-3 py-2 rounded-md flex items-center gap-2 text-caption font-medium text-[var(--color-success-500)] bg-[var(--color-success-500-alpha-08)] border border-[var(--color-success-500-alpha-15)]">
            <CheckIcon className="w-3.5 h-3.5" />
            Training completed. Results shown below.
          </div>
        )}

        {status === "error" && error && (
          <div className="mt-3 px-3 py-2 rounded-md text-caption font-medium text-[var(--color-error-500)] bg-[var(--color-error-500-alpha-08)] border border-[var(--color-error-500-alpha-15)]">
            {error}
          </div>
        )}
      </div>

      {fetchError && (
        <div className="card card-static px-4 py-3 border-[var(--color-error-500)] text-[var(--color-error-500)] text-caption">
          {fetchError}
        </div>
      )}

      {loading && (
        <div className="card card-static p-6 flex items-center justify-center text-[var(--color-on-surface-secondary)] text-small">
          Loading metrics data...
        </div>
      )}

      {!loading && noRunsExist && !fetchError && (
        <div className="card card-static p-8 text-center">
          <CirclePlay className="mx-auto mb-4 block opacity-50 w-12 h-12 text-[var(--color-on-surface-secondary)]" />
          <h3 className="m-0 mb-2 text-[var(--color-on-surface)] text-body font-semibold">No Training Runs Found</h3>
          <p className="m-0 mx-auto text-[var(--color-on-surface-secondary)] text-small leading-relaxed max-w-[420px]">
            TensorBoard event files will appear here once you start a training run. The log directory is derived from
            your workspace configuration.
          </p>
        </div>
      )}

      {!loading && selectedRun && tagData.length === 0 && !fetchError && !noRunsExist && (
        <div className="card card-static p-6 text-center text-[var(--color-on-surface-secondary)] text-small">
          No scalar data found for this run. Training may not have started logging yet.
        </div>
      )}

      {isActive && !hasData && !loading && !noRunsExist && (
        <div className="card card-static p-8 text-center flex flex-col items-center gap-3">
          <div className="skeleton w-[200px] h-2 rounded" />
          <span className="text-[var(--color-on-surface-secondary)] text-small">Waiting for training data...</span>
        </div>
      )}

      {!loading && hasData && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,480px),1fr))] gap-4">
          {tagData
            .filter((td) => td.points.length > 0)
            .map((td) => (
              <ScalarChart key={td.tag} tag={td.tag} points={td.points} lineColor={getLineColor(td.tag)} />
            ))}
        </div>
      )}

      {sampleUrls.length > 0 && (
        <div className="card card-static p-4">
          <div className="flex justify-between items-baseline mb-3">
            <h4 className="m-0 text-caption font-semibold text-[var(--color-on-surface)]">Training Samples</h4>
            <span className="mono tabular-nums text-label text-[var(--color-on-surface-secondary)]">
              {sampleUrls.length}
            </span>
          </div>

          <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
            {latestSample && (
              <div className="rounded-md overflow-hidden border-2 border-[var(--color-cobalt-600-alpha-20)] bg-[var(--color-input-bg)]">
                <img
                  src={latestSample}
                  alt="Latest training sample"
                  className="block w-full max-h-[300px] object-contain"
                />
              </div>
            )}

            {[...sampleUrls]
              .reverse()
              .slice(1)
              .map((sample) => (
                <div
                  key={sample.id}
                  className="rounded overflow-hidden border border-[var(--color-border-subtle)] bg-[var(--color-input-bg)]"
                >
                  <img src={sample.url} alt="Training sample" className="block w-full max-h-[220px] object-contain" />
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
