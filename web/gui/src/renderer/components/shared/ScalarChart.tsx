import { formatStep, formatValue, generateTicks } from "@/utils/chartUtils";

const DEFAULT_WIDTH = 520;
const DEFAULT_HEIGHT = 260;
const CHART_PADDING = { top: 24, right: 16, bottom: 36, left: 64 };

const DEFAULT_LINE_COLOR = "var(--color-cobalt-600)";
const GRID_COLOR = "var(--color-border-subtle)";
const TEXT_COLOR = "var(--color-on-surface-secondary)";
const AXIS_COLOR = "var(--color-on-surface-secondary)";

export interface ScalarChartProps {
  tag: string;
  points: Array<{ step: number; value: number }>;
  lineColor?: string;
  width?: number;
  height?: number;
  xFormat?: (n: number) => string;
  xAxisLabel?: string;
  area?: boolean;
  yMin?: number;
  yMax?: number;
  unit?: string;
}

export function ScalarChart({
  tag,
  points,
  lineColor = DEFAULT_LINE_COLOR,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  xFormat = formatStep,
  xAxisLabel = "step",
  area = true,
  yMin: yMinProp,
  yMax: yMaxProp,
  unit,
}: ScalarChartProps) {
  const gradientId = `area-grad-${tag.replace(/[^a-zA-Z0-9]/g, "-")}`;

  if (points.length === 0) {
    return (
      <div className="card card-static p-4">
        <h4 className="m-0 mb-2 font-semibold text-caption text-[var(--color-on-surface)]">{tag}</h4>
        <div
          className="flex items-center justify-center text-caption text-[var(--color-on-surface-secondary)]"
          style={{ height }}
        >
          No data yet
        </div>
      </div>
    );
  }

  const steps = points.map((p) => p.step);
  const values = points.map((p) => p.value);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);

  const valRange = maxVal - minVal || 1;
  const yMin = yMinProp !== undefined ? yMinProp : minVal - valRange * 0.05;
  const yMax = yMaxProp !== undefined ? yMaxProp : maxVal + valRange * 0.05;

  const stepRange = maxStep - minStep || 1;

  const plotW = width - CHART_PADDING.left - CHART_PADDING.right;
  const plotH = height - CHART_PADDING.top - CHART_PADDING.bottom;

  const xScale = (step: number) => CHART_PADDING.left + ((step - minStep) / stepRange) * plotW;
  const yScale = (val: number) => CHART_PADDING.top + plotH - ((val - yMin) / (yMax - yMin)) * plotH;

  const pathParts = points.map((p, i) => {
    const x = xScale(p.step);
    const y = yScale(p.value);
    return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const linePath = pathParts.join(" ");

  const areaPath =
    `${linePath} L${xScale(points[points.length - 1].step).toFixed(2)},${(CHART_PADDING.top + plotH).toFixed(2)}` +
    ` L${xScale(points[0].step).toFixed(2)},${(CHART_PADDING.top + plotH).toFixed(2)} Z`;

  const xTicks = generateTicks(minStep, maxStep, 5);
  const yTicks = generateTicks(yMin, yMax, 5);

  const latest = points[points.length - 1];

  return (
    <div className="card card-static p-4">
      <div className="flex items-baseline justify-between mb-2">
        <h4 className="m-0 font-semibold text-caption text-[var(--color-on-surface)]">{tag}</h4>
        <span className="mono tabular-nums text-xs font-semibold" style={{ color: lineColor }}>
          {formatValue(latest.value)}
          {unit ? ` ${unit}` : ""}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" className="block">
        {yTicks.map((tick) => {
          const y = yScale(tick);
          return (
            <line
              key={`yg-${tick}`}
              x1={CHART_PADDING.left}
              y1={y}
              x2={width - CHART_PADDING.right}
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
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={lineColor} stopOpacity="0.2" />
                <stop offset="100%" stopColor={lineColor} stopOpacity="0.02" />
              </linearGradient>
            </defs>
            <path d={areaPath} fill={`url(#${gradientId})`} />
          </>
        )}

        <path
          d={linePath}
          fill="none"
          stroke={lineColor}
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {xTicks.map((tick) => {
          const x = xScale(tick);
          if (x < CHART_PADDING.left || x > width - CHART_PADDING.right) return null;
          return (
            <text
              key={`xl-${tick}`}
              x={x}
              y={height - 6}
              textAnchor="middle"
              fill={TEXT_COLOR}
              fontSize="9"
              fontFamily="var(--font-mono)"
            >
              {xFormat(tick)}
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
          y={height - 0}
          textAnchor="middle"
          fill={AXIS_COLOR}
          fontSize="9"
          fontFamily="var(--font-sans)"
        >
          {xAxisLabel}
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
          x2={width - CHART_PADDING.right}
          y2={CHART_PADDING.top + plotH}
          stroke={AXIS_COLOR}
          strokeWidth="0.5"
          opacity="0.5"
        />

        <text
          x={width - CHART_PADDING.right}
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
}
