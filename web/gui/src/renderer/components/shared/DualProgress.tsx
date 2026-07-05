import { ProgressBar } from "./ProgressBar";

export interface DualProgressProps {
  epochProgress: number;
  stepProgress: number;
  epochLabel?: string;
  stepLabel?: string;
}

export function DualProgress({
  epochProgress,
  stepProgress,
  epochLabel = "Epoch",
  stepLabel = "Step",
}: DualProgressProps) {
  return (
    <div className="flex flex-col gap-2 min-w-[200px]">
      <div className="flex flex-col gap-0.5">
        <span className="text-xs font-medium text-[var(--color-on-surface)] tabular-nums">{epochLabel}</span>
        <ProgressBar value={epochProgress} />
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-xs font-medium text-[var(--color-on-surface)] tabular-nums">{stepLabel}</span>
        <ProgressBar value={stepProgress} />
      </div>
    </div>
  );
}
