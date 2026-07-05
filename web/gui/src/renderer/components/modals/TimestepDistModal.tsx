import { RefreshCw } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { Button, FormEntry, Select, Toggle } from "@/components/shared";
import { useConfigField } from "@/hooks/useConfigField";
import type { TimestepDistribution } from "@/types/generated/enums";
import { TimestepDistributionValues } from "@/types/generated/enums";
import { formatValue, generateTicks } from "@/utils/chartUtils";
import { computeTimestepHistogram } from "@/utils/timestepDistribution";

import { ModalBase } from "./ModalBase";

/* ── Chart constants ──────────────────────────────────────── */

const CHART_W = 500;
const CHART_H = 280;
const PAD = { top: 16, right: 16, bottom: 32, left: 56 };
const X_TICKS = [0, 200, 400, 600, 800, 999];

/* ── SVG histogram ────────────────────────────────────────── */

function DistributionChart({ bins }: { bins: Float64Array }) {
  const plotW = CHART_W - PAD.left - PAD.right;
  const plotH = CHART_H - PAD.top - PAD.bottom;

  let maxVal = 0;
  for (const v of bins) {
    if (v > maxVal) maxVal = v;
  }
  const yMax = maxVal || 1;

  const xScale = (bin: number) => PAD.left + (bin / (bins.length - 1)) * plotW;
  const yScale = (val: number) => PAD.top + plotH - (val / yMax) * plotH;

  // Build line + area paths
  const parts: string[] = [];
  for (let i = 0; i < bins.length; i++) {
    parts.push(`${i === 0 ? "M" : "L"}${xScale(i).toFixed(2)},${yScale(bins[i]).toFixed(2)}`);
  }
  const linePath = parts.join(" ");
  const areaPath =
    `${linePath} L${xScale(bins.length - 1).toFixed(2)},${(PAD.top + plotH).toFixed(2)}` +
    ` L${xScale(0).toFixed(2)},${(PAD.top + plotH).toFixed(2)} Z`;

  const yTicks = generateTicks(0, yMax, 4);

  return (
    <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} width="100%" className="block">
      {/* Horizontal grid lines */}
      {yTicks.map((tick) => {
        const y = yScale(tick);
        if (y < PAD.top || y > PAD.top + plotH) return null;
        return (
          <line
            key={`yg-${tick}`}
            x1={PAD.left}
            y1={y}
            x2={CHART_W - PAD.right}
            y2={y}
            stroke="var(--color-border-subtle)"
            strokeWidth="0.5"
            strokeDasharray="4,3"
          />
        );
      })}

      {/* Area gradient */}
      <defs>
        <linearGradient id="timestep-dist-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--color-cobalt-600)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="var(--color-cobalt-600)" stopOpacity="0.03" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill="url(#timestep-dist-grad)" />

      {/* Distribution line */}
      <path
        d={linePath}
        fill="none"
        stroke="var(--color-cobalt-600)"
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* Axes */}
      <line
        x1={PAD.left}
        y1={PAD.top}
        x2={PAD.left}
        y2={PAD.top + plotH}
        stroke="var(--color-on-surface-secondary)"
        strokeWidth="0.5"
        opacity="0.5"
      />
      <line
        x1={PAD.left}
        y1={PAD.top + plotH}
        x2={CHART_W - PAD.right}
        y2={PAD.top + plotH}
        stroke="var(--color-on-surface-secondary)"
        strokeWidth="0.5"
        opacity="0.5"
      />

      {/* X labels */}
      {X_TICKS.map((tick) => (
        <text
          key={`xl-${tick}`}
          x={xScale(tick)}
          y={CHART_H - 6}
          textAnchor="middle"
          fill="var(--color-on-surface-secondary)"
          fontSize="9"
          fontFamily="var(--font-mono)"
        >
          {tick}
        </text>
      ))}

      {/* Y labels */}
      {yTicks.map((tick) => {
        const y = yScale(tick);
        if (y < PAD.top || y > PAD.top + plotH) return null;
        return (
          <text
            key={`yl-${tick}`}
            x={PAD.left - 6}
            y={y + 3}
            textAnchor="end"
            fill="var(--color-on-surface-secondary)"
            fontSize="9"
            fontFamily="var(--font-mono)"
          >
            {formatValue(tick)}
          </text>
        );
      })}

      {/* Axis title */}
      <text
        x={PAD.left + plotW / 2}
        y={CHART_H}
        textAnchor="middle"
        fill="var(--color-on-surface-secondary)"
        fontSize="9"
        fontFamily="var(--font-sans)"
      >
        timestep
      </text>
    </svg>
  );
}

/* ── Modal ─────────────────────────────────────────────────── */

export interface TimestepDistModalProps {
  open: boolean;
  onClose: () => void;
}

export function TimestepDistModal({ open, onClose }: TimestepDistModalProps) {
  const [distribution] = useConfigField<TimestepDistribution>("timestep_distribution");
  const [minNoisingStrength] = useConfigField<number>("min_noising_strength");
  const [maxNoisingStrength] = useConfigField<number>("max_noising_strength");
  const [noisingWeight] = useConfigField<number>("noising_weight");
  const [noisingBias] = useConfigField<number>("noising_bias");
  const [timestepShift] = useConfigField<number>("timestep_shift");

  // Increment to force re-sampling (continuous distributions use Math.random)
  const [seed, setSeed] = useState(0);

  const bins = useMemo(() => {
    if (
      distribution === undefined ||
      minNoisingStrength === undefined ||
      maxNoisingStrength === undefined ||
      noisingWeight === undefined ||
      noisingBias === undefined ||
      timestepShift === undefined
    ) {
      return new Float64Array(1000);
    }
    return computeTimestepHistogram({
      distribution,
      minNoisingStrength,
      maxNoisingStrength,
      noisingWeight,
      noisingBias,
      timestepShift,
    });
  }, [distribution, minNoisingStrength, maxNoisingStrength, noisingWeight, noisingBias, timestepShift, seed]);

  const handleUpdatePreview = useCallback(() => setSeed((s) => s + 1), []);

  return (
    <ModalBase open={open} onClose={onClose} title="Timestep Distribution" size="xl">
      <div className="flex gap-6">
        {/* Controls */}
        <div className="flex flex-col gap-4 min-w-[240px]">
          <Select
            label="Timestep Distribution"
            configPath="timestep_distribution"
            options={[...TimestepDistributionValues]}
            tooltip="Selects the function to sample timesteps during training"
          />
          <FormEntry
            label="Min Noising Strength"
            configPath="min_noising_strength"
            type="number"
            tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained"
          />
          <FormEntry
            label="Max Noising Strength"
            configPath="max_noising_strength"
            type="number"
            tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition"
          />
          <FormEntry
            label="Noising Weight"
            configPath="noising_weight"
            type="number"
            tooltip="Controls the weight parameter of the timestep distribution function. Use the preview to see more details."
          />
          <FormEntry
            label="Noising Bias"
            configPath="noising_bias"
            type="number"
            tooltip="Controls the bias parameter of the timestep distribution function. Use the preview to see more details."
          />
          <FormEntry
            label="Timestep Shift"
            configPath="timestep_shift"
            type="number"
            tooltip="Shift the timestep distribution. Use the preview to see more details."
          />
          <Toggle
            configPath="dynamic_timestep_shifting"
            label="Dynamic Timestep Shifting"
            tooltip="Dynamically shift the timestep distribution based on resolution. If enabled, the shifting parameters are taken from the model's scheduler configuration and Timestep Shift is ignored. Dynamic Timestep Shifting is not shown in the preview."
          />

          <Button variant="secondary" size="sm" onClick={handleUpdatePreview}>
            <RefreshCw className="w-3.5 h-3.5" />
            Update Preview
          </Button>
        </div>

        {/* Chart */}
        <div className="flex-1 min-w-0 rounded-[var(--radius-sm)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] p-3 flex items-center">
          <DistributionChart bins={bins} />
        </div>
      </div>

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
