import { useCallback, useMemo, useState } from "react";

import { ScalarChart } from "@/components/shared/ScalarChart";
import { useReconnectingWebSocket } from "@/hooks/useReconnectingWebSocket";
import { useUiStore } from "@/store/uiStore";

interface GpuMetrics {
  index: number;
  name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  vram_percent: number;
  temperature: number | null;
  utilization: number | null;
}

interface MetricsSnapshot {
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  gpus: GpuMetrics[];
}

interface TimestampedMetrics extends MetricsSnapshot {
  timestamp: number;
}

const MAX_POINTS = 300;

const GPU_COLORS = [
  "var(--color-cobalt-600)",
  "var(--color-azure-500)",
  "var(--color-success-500)",
  "var(--color-warning-500)",
  "var(--color-info-500)",
  "var(--color-error-500)",
];

function formatTime(secondsAgo: number): string {
  if (secondsAgo < 60) return `${Math.round(secondsAgo)}s`;
  return `${Math.floor(secondsAgo / 60)}m${Math.round(secondsAgo % 60)}s`;
}

interface UseSystemWebSocketResult {
  connected: boolean;
  latest: MetricsSnapshot | null;
  history: TimestampedMetrics[];
}

function useSystemWebSocket(): UseSystemWebSocketResult {
  const backendConnected = useUiStore((s) => s.backendConnected);
  const [connected, setConnected] = useState(false);
  const [latest, setLatest] = useState<MetricsSnapshot | null>(null);
  const [history, setHistory] = useState<TimestampedMetrics[]>([]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "metrics" && msg.data) {
        const snapshot: MetricsSnapshot = msg.data;
        const timestamped: TimestampedMetrics = {
          ...snapshot,
          timestamp: Date.now() / 1000,
        };

        setLatest(snapshot);
        setHistory((prev) => {
          const next = [...prev, timestamped];
          if (next.length > MAX_POINTS) {
            return next.slice(next.length - MAX_POINTS);
          }
          return next;
        });
      }
    } catch {
      /* ignore */
    }
  }, []);

  useReconnectingWebSocket({
    path: "/ws/system",
    onMessage: handleMessage,
    onOpen: () => setConnected(true),
    onClose: () => setConnected(false),
    enabled: backendConnected,
  });

  return { connected, latest, history };
}

interface ChartPoint {
  step: number;
  value: number;
}

interface StatCardProps {
  label: string;
  value: string;
  subtext?: string;
  color?: string;
}

function StatCard({ label, value, subtext, color }: StatCardProps) {
  return (
    <div className="bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] rounded-[var(--radius-sm)] p-4 flex flex-col gap-1">
      <span className="text-label font-medium text-[var(--color-on-surface-secondary)] uppercase tracking-wider">
        {label}
      </span>
      <span
        className="mono tabular-nums text-2xl font-bold leading-tight"
        style={{ color: color ?? "var(--color-on-surface)" }}
      >
        {value}
      </span>
      {subtext && (
        <span className="mono tabular-nums text-label text-[var(--color-on-surface-secondary)]">{subtext}</span>
      )}
    </div>
  );
}

function extractTimeSeries(
  history: TimestampedMetrics[],
  accessor: (m: TimestampedMetrics) => number | null | undefined,
): ChartPoint[] {
  if (history.length === 0) return [];

  const startTime = history[0].timestamp;
  const points: ChartPoint[] = [];

  for (const m of history) {
    const val = accessor(m);
    if (val !== null && val !== undefined) {
      points.push({ step: m.timestamp - startTime, value: val });
    }
  }

  return points;
}

export default function PerformancePage() {
  const { connected, latest, history } = useSystemWebSocket();

  const cpuPoints = useMemo(() => extractTimeSeries(history, (m) => m.cpu_percent), [history]);
  const ramPoints = useMemo(() => extractTimeSeries(history, (m) => m.ram_used_gb), [history]);

  const gpuCount = latest?.gpus.length ?? 0;

  // Extract all per-GPU series (VRAM/utilization/temperature) in a single pass over `history`
  // so that appending one new sample doesn't trigger 3xgpuCount independent O(history) rebuilds
  // on every render. This recomputes only when `history` or `gpuCount` actually changes (e.g. not
  // when `connected` toggles independently).
  const gpuSeries = useMemo(
    () =>
      Array.from({ length: gpuCount }, (_, gpuIdx) => ({
        vram: extractTimeSeries(history, (m) => m.gpus[gpuIdx]?.vram_used_mb ?? null),
        utilization: extractTimeSeries(history, (m) => m.gpus[gpuIdx]?.utilization ?? null),
        temperature: extractTimeSeries(history, (m) => m.gpus[gpuIdx]?.temperature ?? null),
      })),
    [history, gpuCount],
  );

  return (
    <div className="flex flex-col gap-6">
      <div className="card card-static p-4">
        <div className="flex items-center gap-4 flex-wrap">
          <span className="text-caption font-semibold text-[var(--color-on-surface)]">System Monitor</span>

          <span
            className={`text-label inline-flex items-center gap-1 ${
              connected ? "text-[var(--color-success-500)]" : "text-[var(--color-on-surface-secondary)]"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full inline-block ${
                connected ? "bg-[var(--color-success-500)]" : "bg-[var(--color-on-surface-secondary)]"
              }`}
              style={{
                animation: connected ? "pulseAlive 2s infinite" : "none",
              }}
            />
            {connected ? "Live (1s interval)" : "Connecting..."}
          </span>

          {history.length > 0 && (
            <span className="mono tabular-nums text-label text-[var(--color-on-surface-secondary)] ml-auto">
              {history.length}/{MAX_POINTS} samples
            </span>
          )}
        </div>
      </div>

      {latest && (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3">
          <StatCard
            label="CPU"
            value={`${latest.cpu_percent.toFixed(1)}%`}
            color={
              latest.cpu_percent > 90
                ? "var(--color-error-500)"
                : latest.cpu_percent > 70
                  ? "var(--color-warning-500)"
                  : "var(--color-success-500)"
            }
          />
          <StatCard
            label="RAM"
            value={`${latest.ram_used_gb.toFixed(1)} GB`}
            subtext={`/ ${latest.ram_total_gb.toFixed(1)} GB (${latest.ram_percent.toFixed(0)}%)`}
            color={
              latest.ram_percent > 90
                ? "var(--color-error-500)"
                : latest.ram_percent > 70
                  ? "var(--color-warning-500)"
                  : "var(--color-on-surface)"
            }
          />
          {latest.gpus.map((gpu) => (
            <StatCard
              key={gpu.index}
              label={`GPU ${gpu.index} VRAM`}
              value={`${(gpu.vram_used_mb / 1024).toFixed(1)} GB`}
              subtext={`/ ${(gpu.vram_total_mb / 1024).toFixed(1)} GB (${gpu.vram_percent.toFixed(0)}%) - ${gpu.name}`}
              color={
                gpu.vram_percent > 90
                  ? "var(--color-error-500)"
                  : gpu.vram_percent > 70
                    ? "var(--color-warning-500)"
                    : "var(--color-cobalt-600)"
              }
            />
          ))}
          {latest.gpus.map(
            (gpu) =>
              gpu.temperature !== null && (
                <StatCard
                  key={`temp-${gpu.index}`}
                  label={`GPU ${gpu.index} Temp`}
                  value={`${gpu.temperature.toFixed(0)}\u00B0C`}
                  color={
                    gpu.temperature > 85
                      ? "var(--color-error-500)"
                      : gpu.temperature > 70
                        ? "var(--color-warning-500)"
                        : "var(--color-success-500)"
                  }
                />
              ),
          )}
          {latest.gpus.map(
            (gpu) =>
              gpu.utilization !== null && (
                <StatCard
                  key={`util-${gpu.index}`}
                  label={`GPU ${gpu.index} Util`}
                  value={`${gpu.utilization.toFixed(0)}%`}
                  color={gpu.utilization > 90 ? "var(--color-cobalt-600)" : "var(--color-on-surface)"}
                />
              ),
          )}
        </div>
      )}

      {latest && gpuCount === 0 && (
        <div className="card card-static p-4 text-center text-[var(--color-on-surface-secondary)] text-small">
          No GPU detected. GPU metrics are unavailable. CPU and RAM monitoring remain active.
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <ScalarChart
          tag="CPU Usage"
          points={cpuPoints}
          unit="%"
          lineColor="var(--color-info-500)"
          area={false}
          yMax={100}
          yMin={0}
          xFormat={formatTime}
          xAxisLabel="time"
        />

        <ScalarChart
          tag="RAM Usage"
          points={ramPoints}
          unit="GB"
          lineColor="var(--color-azure-500)"
          yMin={0}
          yMax={latest?.ram_total_gb}
          xFormat={formatTime}
          xAxisLabel="time"
        />

        {Array.from({ length: gpuCount }).flatMap((_, gpuIdx) => {
          const gpu = latest?.gpus[gpuIdx];
          const gpuName = gpu?.name ?? `GPU ${gpuIdx}`;
          const color = GPU_COLORS[gpuIdx % GPU_COLORS.length];
          const series = gpuSeries[gpuIdx];
          const charts: React.ReactNode[] = [];

          charts.push(
            <ScalarChart
              key={`vram-${gpuIdx}`}
              tag={`VRAM - ${gpuName}`}
              points={series.vram}
              unit="MB"
              lineColor={color}
              yMin={0}
              yMax={gpu?.vram_total_mb}
              xFormat={formatTime}
              xAxisLabel="time"
            />,
          );

          if (gpu?.utilization !== null) {
            charts.push(
              <ScalarChart
                key={`util-${gpuIdx}`}
                tag={`GPU Utilization - ${gpuName}`}
                points={series.utilization}
                unit="%"
                lineColor={color}
                area={false}
                yMax={100}
                yMin={0}
                xFormat={formatTime}
                xAxisLabel="time"
              />,
            );
          }

          if (gpu?.temperature !== null) {
            charts.push(
              <ScalarChart
                key={`temp-${gpuIdx}`}
                tag={`Temperature - ${gpuName}`}
                points={series.temperature}
                unit={"\u00B0C"}
                lineColor="var(--color-error-500)"
                area={false}
                yMin={0}
                yMax={100}
                xFormat={formatTime}
                xAxisLabel="time"
              />,
            );
          }

          return charts;
        })}
      </div>

      {!latest && (
        <div className="card card-static p-8 text-center">
          <h3 className="m-0 mb-2 text-[var(--color-on-surface)] text-body font-semibold">Waiting for Metrics</h3>
          <p className="m-0 text-[var(--color-on-surface-secondary)] text-small">
            System metrics will appear here once the backend connection is established. The monitor streams CPU, RAM,
            and GPU data in real time.
          </p>
        </div>
      )}
    </div>
  );
}
