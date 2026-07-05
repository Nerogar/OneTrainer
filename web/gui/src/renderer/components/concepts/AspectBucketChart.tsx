export interface AspectBucketChartProps {
  buckets: Record<string, number>;
}

function decimalToAspectRatio(value: number): string {
  // Find closest fraction with limited denominator (matching Python's Fraction.limit_denominator(16))
  let bestNum = 1;
  let bestDen = 1;
  let bestError = Math.abs(value - 1);

  for (let den = 1; den <= 16; den++) {
    const num = Math.round(value * den);
    if (num < 1) continue;
    const error = Math.abs(value - num / den);
    if (error < bestError) {
      bestError = error;
      bestNum = num;
      bestDen = den;
    }
  }

  // Format as W:H (denominator:numerator since aspect = height/width)
  return `${bestDen}:${bestNum}`;
}

export function AspectBucketChart({ buckets }: AspectBucketChartProps) {
  const entries = Object.entries(buckets)
    .map(([k, v]) => ({ ratio: parseFloat(k), count: v as number }))
    .sort((a, b) => a.ratio - b.ratio);

  if (entries.length === 0) {
    return (
      <p className="text-xs text-[var(--color-on-surface-secondary)] text-center py-4">
        No aspect bucket data available.
      </p>
    );
  }

  const maxCount = Math.max(...entries.map((e) => e.count), 1);
  const nonZeroEntries = entries.filter((e) => e.count > 0);
  const minNonZero = nonZeroEntries.length > 0 ? Math.min(...nonZeroEntries.map((e) => e.count)) : 0;

  const barWidth = 18;
  const barGap = 2;
  const chartHeight = 160;
  const labelHeight = 50;
  const topPadding = 20;
  const leftPadding = 5;
  const totalWidth = entries.length * (barWidth + barGap) + leftPadding * 2;
  const totalHeight = chartHeight + labelHeight + topPadding;

  return (
    <div className="overflow-x-auto">
      <svg
        width={Math.max(totalWidth, 200)}
        height={totalHeight}
        viewBox={`0 0 ${Math.max(totalWidth, 200)} ${totalHeight}`}
        className="block"
      >
        <text
          x={leftPadding + 2}
          y={totalHeight - 2}
          className="fill-[var(--color-on-surface-secondary)]"
          fontSize="9"
          textAnchor="start"
        >
          Wide
        </text>
        <text
          x={totalWidth / 2}
          y={totalHeight - 2}
          className="fill-[var(--color-on-surface-secondary)]"
          fontSize="9"
          textAnchor="middle"
        >
          Square
        </text>
        <text
          x={totalWidth - leftPadding - 2}
          y={totalHeight - 2}
          className="fill-[var(--color-on-surface-secondary)]"
          fontSize="9"
          textAnchor="end"
        >
          Tall
        </text>

        {entries.map((entry, i) => {
          const x = leftPadding + i * (barWidth + barGap);
          const barH = maxCount > 0 ? (entry.count / maxCount) * chartHeight : 0;
          const y = topPadding + chartHeight - barH;
          const label = decimalToAspectRatio(entry.ratio);

          const isSmallest = entry.count > 0 && entry.count === minNonZero && nonZeroEntries.length > 1;
          const barColor = isSmallest ? "var(--color-warning-500)" : "var(--color-cobalt-600)";

          return (
            <g key={entry.ratio}>
              {entry.count > 0 && (
                <rect x={x} y={y} width={barWidth} height={barH} fill={barColor} opacity={0.8} rx={2} />
              )}
              {entry.count > 0 && (
                <text
                  x={x + barWidth / 2}
                  y={y - 3}
                  className="fill-[var(--color-on-surface-secondary)]"
                  fontSize="8"
                  textAnchor="middle"
                >
                  {entry.count}
                </text>
              )}
              <text
                x={x + barWidth / 2}
                y={topPadding + chartHeight + 12}
                className="fill-[var(--color-on-surface-secondary)]"
                fontSize="7"
                textAnchor="middle"
                transform={`rotate(45, ${x + barWidth / 2}, ${topPadding + chartHeight + 12})`}
              >
                {label}
              </text>
            </g>
          );
        })}

        <line
          x1={leftPadding}
          y1={topPadding + chartHeight}
          x2={totalWidth - leftPadding}
          y2={topPadding + chartHeight}
          stroke="var(--color-border-subtle)"
          strokeWidth="1"
        />
      </svg>
    </div>
  );
}
