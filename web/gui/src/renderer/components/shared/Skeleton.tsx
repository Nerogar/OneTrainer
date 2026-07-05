export interface SkeletonProps {
  width?: string;
  height?: string;
  variant?: "text" | "rectangular" | "circular";
  count?: number;
}

export function Skeleton({ width = "100%", height = "1rem", variant = "text", count = 1 }: SkeletonProps) {
  const radiusClass = variant === "circular" ? "rounded-full" : variant === "text" ? "rounded-[var(--radius-sm)]" : "";
  return (
    <div className="flex flex-col gap-2" aria-busy="true" aria-label="Loading">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className={`skeleton ${radiusClass}`} style={{ width, height }} />
      ))}
    </div>
  );
}
