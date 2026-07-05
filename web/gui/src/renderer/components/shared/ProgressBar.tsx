export interface ProgressBarProps {
  value: number;
  label?: string;
  indeterminate?: boolean;
}

export function ProgressBar({ value, label, indeterminate }: ProgressBarProps) {
  const clamped = Math.min(100, Math.max(0, value));
  const isAnimating = !indeterminate && clamped > 0 && clamped < 100;

  return (
    <div className="flex flex-col gap-1">
      {label && <span className="text-xs text-[var(--color-on-surface-secondary)]">{label}</span>}
      <div
        className="h-2.5 w-full rounded-full bg-[var(--color-border-subtle)] overflow-hidden"
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label || "Progress"}
      >
        <div
          className={`h-full rounded-full transition-all duration-300 ease-out ${indeterminate ? "skeleton" : ""}`}
          style={{
            width: indeterminate ? "100%" : `${clamped}%`,
            background: indeterminate
              ? undefined
              : isAnimating
                ? `linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent), linear-gradient(90deg, var(--color-cobalt-600), var(--color-azure-500))`
                : `linear-gradient(90deg, var(--color-cobalt-600), var(--color-azure-500))`,
            backgroundSize: isAnimating ? "200% 100%, 100% 100%" : undefined,
            animation: isAnimating ? "progressShimmer 2s ease-in-out infinite" : undefined,
            boxShadow: clamped > 0 && !indeterminate ? "0 0 8px var(--color-cobalt-600-alpha-20)" : "none",
          }}
        />
      </div>
      {!indeterminate && (
        <span className="text-xs text-[var(--color-on-surface-secondary)] tabular-nums">{clamped.toFixed(1)}%</span>
      )}
    </div>
  );
}
