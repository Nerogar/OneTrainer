import { memo, useCallback, useMemo, useState } from "react";

import { useReconnectingWebSocket } from "@/hooks/useReconnectingWebSocket";
import { useUiStore } from "@/store/uiStore";
import { formatValue, generateTicks } from "@/utils/chartUtils";

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

const CHART_WIDTH = 520;
const CHART_HEIGHT = 260;
const CHART_PADDING = { top: 24, right: 16, bottom: 36, left: 64 };

const LINE_COLOR = "var(--color-cobalt-600)";
const GRID_COLOR = "var(--color-border-subtle)";
const TEXT_COLOR = "var(--color-on-surface-secondary)";
const AXIS_COLOR = "var(--color-on-surface-secondary)";

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
  time: number;
  value: number;
}

interface MetricChartProps {
  title: string;
  points: ChartPoint[];
  unit: string;
  color?: string;
  area?: boolean;
  yMax?: number;
  yMin?: number;
}

const MetricChart = memo(function MetricChart({
  title,
  points,
  unit,
  color = LINE_COLOR,
  area = false,
  yMax: fixedYMax,
  yMin: fixedYMin,
}: MetricChartProps) {
  if (points.length === 0) {
    return (
      <div className="card card-static p-4">
        <h4 className="m-0 mb-2 text-caption font-semibold text-[var(--color-on-surface)]">{title}</h4>
        <div
          className="flex items-center justify-center text-[var(--color-on-surface-secondary)] text-caption"
          style={{ height: CHART_HEIGHT }}
        >
          Waiting for data...
        </div>
      </div>
    );
  }

  const times = points.map((p) => p.time);
  const values = points.map((p) => p.value);
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);

  const rawMinVal = Math.min(...values);
  const rawMaxVal = Math.max(...values);

  const yMin = fixedYMin !== undefined ? fixedYMin : Math.max(0, rawMinVal - (rawMaxVal - rawMinVal) * 0.05);
  const yMax = fixedYMax !== undefined ? fixedYMax : rawMaxVal + (rawMaxVal - rawMinVal) * 0.05 || 1;

  const timeRange = maxTime - minTime || 1;

  const plotW = CHART_WIDTH - CHART_PADDING.left - CHART_PADDING.right;
  const plotH = CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom;

  const xScale = (time: number) => CHART_PADDING.left + ((time - minTime) / timeRange) * plotW;
  const yScale = (val: number) => CHART_PADDING.top + plotH - ((val - yMin) / (yMax - yMin)) * plotH;

  const pathParts = points.map((p, i) => {
    const x = xScale(p.time);
    const y = yScale(p.value);
    return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const linePath = pathParts.join(" ");

  const areaPath =
    `${linePath} L${xScale(points[points.length - 1].time).toFixed(2)},${(CHART_PADDING.top + plotH).toFixed(2)}` +
    ` L${xScale(points[0].time).toFixed(2)},${(CHART_PADDING.top + plotH).toFixed(2)} Z`;

  const xTicks = generateTicks(0, maxTime - minTime, 5);
  const yTicks = generateTicks(yMin, yMax, 5);

  const latestValue = points[points.length - 1].value;
  const gradId = `perf-area-${title.replace(/[^a-zA-Z0-9]/g, "-")}`;

  return (
    <div className="card card-static p-4">
      <div className="flex justify-between items-baseline mb-2">
        <h4 className="m-0 text-caption font-semibold text-[var(--color-on-surface)]">{title}</h4>
        <span className="mono tabular-nums text-micro font-semibold" style={{ color }}>
          {formatValue(latestValue)} {unit}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        width="100%"
        className="block"
        role="img"
        aria-label={`${title} chart`}
      >
        <title>{title}</title>
        {yTicks.map((tick) => {
          const y = yScale(tick);
          return (
            <line
              key={`yg-${tick}`}
              x1={CHART_PADDING.left}
              y1={y}
              x2={CHART_WIDTH - CHART_PADDING.right}
              y2={y}
              stroke={GRID_COLOR}
              strokeWidth="0.5"
              strokeDasharray="4,3"
            />
          );
        })}

        {area && (
          <>
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                <stop offset="100%" stopColor={color} stopOpacity="0.02" />
              </linearGradient>
            </defs>
            <path d={areaPath} fill={`url(#${gradId})`} />
          </>
        )}

        <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />

        {xTicks.map((tick) => {
          const x = xScale(minTime + tick);
          if (x < CHART_PADDING.left || x > CHART_WIDTH - CHART_PADDING.right) return null;
          return (
            <text
              key={`xl-${tick}`}
              x={x}
              y={CHART_HEIGHT - 6}
              textAnchor="middle"
              fill={TEXT_COLOR}
              fontSize="9"
              fontFamily="var(--font-mono)"
            >
              {formatTime(tick)}
            </text>
          );
        })}

        {yTicks.map((tick) => {
          const y = yScale(tick);
          if (y < CHART_PADDING.top || y > CHART_PADDING.top + plotH) return null;
          return (
            <text
              key={`yl-${tick}`}
              x={CHART_PADDING.left - 6}
              y={y + 3}
              textAnchor="end"
              fill={TEXT_COLOR}
              fontSize="9"
              fontFamily="var(--font-mono)"
            >
              {formatValue(tick)}
            </text>
          );
        })}

        <text
          x={CHART_PADDING.left + plotW / 2}
          y={CHART_HEIGHT - 0}
          textAnchor="middle"
          fill={AXIS_COLOR}
          fontSize="9"
          fontFamily="var(--font-sans)"
        >
          time
        </text>

        <line
          x1={CHART_PADDING.left}
          y1={CHART_PADDING.top}
          x2={CHART_PADDING.left}
          y2={CHART_PADDING.top + plotH}
          stroke={AXIS_COLOR}
          strokeWidth="0.5"
          opacity="0.5"
        />
        <line
          x1={CHART_PADDING.left}
          y1={CHART_PADDING.top + plotH}
          x2={CHART_WIDTH - CHART_PADDING.right}
          y2={CHART_PADDING.top + plotH}
          stroke={AXIS_COLOR}
          strokeWidth="0.5"
          opacity="0.5"
        />

        <text
          x={CHART_WIDTH - CHART_PADDING.right}
          y={CHART_PADDING.top - 8}
          textAnchor="end"
          fill={TEXT_COLOR}
          fontSize="8"
          fontFamily="var(--font-mono)"
          opacity="0.6"
        >
          {points.length} pts
        </text>
      </svg>
    </div>
  );
});

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
      points.push({ time: m.timestamp - startTime, value: val });
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
        <MetricChart title="CPU Usage" points={cpuPoints} unit="%" color="var(--color-info-500)" yMax={100} yMin={0} />

        <MetricChart
          title="RAM Usage"
          points={ramPoints}
          unit="GB"
          color="var(--color-azure-500)"
          area
          yMin={0}
          yMax={latest?.ram_total_gb}
        />

        {Array.from({ length: gpuCount }).flatMap((_, gpuIdx) => {
          const gpu = latest?.gpus[gpuIdx];
          const gpuName = gpu?.name ?? `GPU ${gpuIdx}`;
          const color = GPU_COLORS[gpuIdx % GPU_COLORS.length];
          const series = gpuSeries[gpuIdx];
          const charts: React.ReactNode[] = [];

          charts.push(
            <MetricChart
              key={`vram-${gpuIdx}`}
              title={`VRAM - ${gpuName}`}
              points={series.vram}
              unit="MB"
              color={color}
              area
              yMin={0}
              yMax={gpu?.vram_total_mb}
            />,
          );

          if (gpu?.utilization !== null) {
            charts.push(
              <MetricChart
                key={`util-${gpuIdx}`}
                title={`GPU Utilization - ${gpuName}`}
                points={series.utilization}
                unit="%"
                color={color}
                yMax={100}
                yMin={0}
              />,
            );
          }

          if (gpu?.temperature !== null) {
            charts.push(
              <MetricChart
                key={`temp-${gpuIdx}`}
                title={`Temperature - ${gpuName}`}
                points={series.temperature}
                unit={"\u00B0C"}
                color="var(--color-error-500)"
                yMin={0}
                yMax={100}
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
