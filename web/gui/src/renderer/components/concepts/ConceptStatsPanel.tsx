import { useCallback, useState } from "react";

import { configApi } from "@/api/configApi";
import { Button, Tooltip } from "@/components/shared";

import { AspectBucketChart } from "./AspectBucketChart";

export interface ConceptStatsPanelProps {
  conceptPath: string;
  includeSubdirectories: boolean;
}

interface StatValue {
  label: string;
  value: string;
  tooltip?: string;
}

/* ── Format helpers ────────────────────────────────────────── */

function formatPixelStat(raw: unknown): string {
  if (typeof raw === "string") return "-";
  if (Array.isArray(raw)) {
    const [pixels, file, resolution] = raw;
    const mp = (pixels / 1_000_000).toFixed(2);
    return `${mp} MP, ${resolution}\n${file}`;
  }
  if (typeof raw === "number") {
    const mp = (raw / 1_000_000).toFixed(2);
    const side = Math.round(Math.sqrt(raw));
    return `${mp} MP, ~${side}w x ${side}h`;
  }
  return "-";
}

function formatLengthStat(raw: unknown): string {
  if (typeof raw === "string") return "-";
  if (Array.isArray(raw)) return `${Math.round(raw[0])} frames\n${raw[1]}`;
  if (typeof raw === "number") return `${Math.round(raw)} frames`;
  return "-";
}

function formatFpsStat(raw: unknown): string {
  if (typeof raw === "string") return "-";
  if (Array.isArray(raw)) return `${Math.round(raw[0])} fps\n${raw[1]}`;
  if (typeof raw === "number") return `${Math.round(raw)} fps`;
  return "-";
}

function formatCaptionStat(raw: unknown): string {
  if (typeof raw === "string") return "-";
  if (Array.isArray(raw)) {
    if (raw.length >= 3) return `${raw[0]} chars, ${raw[2]} words\n${raw[1]}`;
    if (raw.length >= 2) return `${Math.round(raw[0])} chars, ${Math.round(raw[1])} words`;
  }
  return "-";
}

function decimalToAspectRatio(value: number): string {
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
  return `${bestDen}:${bestNum}`;
}

function computeSmallestBuckets(buckets: Record<string, number>): string {
  const nonZero = Object.entries(buckets)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({ ratio: parseFloat(k), count: v }));

  if (nonZero.length === 0) return "-";

  nonZero.sort((a, b) => a.count - b.count);
  const minVal = nonZero[0].count;
  const minVal2 = nonZero.length > 1 && nonZero[1].count > minVal ? nonZero[1].count : minVal;

  return nonZero
    .filter((e) => e.count === minVal || e.count === minVal2)
    .map((e) => `aspect ${decimalToAspectRatio(e.ratio)} : ${e.count} img`)
    .join("\n");
}

/* ── Stat label with optional tooltip ──────────────────────── */

function StatLabel({ label, tooltip }: { label: string; tooltip?: string }) {
  const el = (
    <span className="text-xs font-semibold text-[var(--color-on-surface-secondary)] underline decoration-dotted underline-offset-2">
      {label}
    </span>
  );
  if (tooltip) {
    return <Tooltip content={tooltip}>{el}</Tooltip>;
  }
  return el;
}

/* ── Main component ────────────────────────────────────────── */

export function ConceptStatsPanel({ conceptPath, includeSubdirectories }: ConceptStatsPanelProps) {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanType, setScanType] = useState<"basic" | "advanced" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runScan = useCallback(
    async (advanced: boolean) => {
      if (!conceptPath) return;
      setScanning(true);
      setScanType(advanced ? "advanced" : "basic");
      setError(null);
      try {
        const result = await configApi.conceptStats(conceptPath, includeSubdirectories, advanced);
        setStats(result);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("Stats scan failed:", err);
        setError(msg);
      } finally {
        setScanning(false);
        setScanType(null);
      }
    },
    [conceptPath, includeSubdirectories],
  );

  const handleCancel = useCallback(async () => {
    try {
      await configApi.cancelConceptStats();
    } catch (err) {
      console.error("Cancel failed:", err);
    }
  }, []);

  const s = stats ?? {};

  const fileSize = typeof s.file_size === "number" ? `${Math.round(s.file_size / 1_048_576)} MB` : "-";
  const processingTime = typeof s.processing_time === "number" ? `${(s.processing_time as number).toFixed(2)} s` : "-";

  /* ── Stat definitions ────────────────────────────────────── */

  const basicStats: StatValue[][] = [
    [
      { label: "Total Size", value: fileSize, tooltip: "Total size of all image, mask, and caption files in MB" },
      {
        label: "Directories",
        value: String(s.directory_count ?? "-"),
        tooltip:
          "Total number of directories including and under (if 'include subdirectories' is enabled) the main concept directory",
      },
    ],
    [
      {
        label: "Total Images",
        value: String(s.image_count ?? "-"),
        tooltip: "Total number of image files, excluding mask and condition label files",
      },
      { label: "Total Videos", value: String(s.video_count ?? "-"), tooltip: "Total number of video files" },
      {
        label: "Total Masks",
        value: String(s.mask_count ?? "-"),
        tooltip: "Total number of mask files (ending in '-masklabel.png')",
      },
      {
        label: "Total Captions",
        value:
          s.subcaption_count && typeof s.subcaption_count === "number" && s.subcaption_count > 0
            ? `${s.caption_count} (${s.subcaption_count})`
            : String(s.caption_count ?? "-"),
        tooltip:
          "Total number of caption files (.txt). With advanced scan, includes the total number of captions on separate lines across all files in parentheses.",
      },
    ],
    [
      {
        label: "Images with Masks",
        value: String(s.image_with_mask_count ?? "-"),
        tooltip: "Total number of image files with an associated mask",
      },
      {
        label: "Unpaired Masks",
        value: String(s.unpaired_masks ?? "-"),
        tooltip: "Total number of mask files which lack a corresponding image file — if > 0, check your dataset!",
      },
    ],
    [
      {
        label: "Images with Captions",
        value: String(s.image_with_caption_count ?? "-"),
        tooltip: "Total number of image files with an associated caption",
      },
      {
        label: "Videos with Captions",
        value: String(s.video_with_caption_count ?? "-"),
        tooltip: "Total number of video files with an associated caption",
      },
      {
        label: "Unpaired Captions",
        value: String(s.unpaired_captions ?? "-"),
        tooltip:
          "Total number of caption files which lack a corresponding image file — if > 0, check your dataset! If using 'from file name' or 'from single text file' this can be ignored.",
      },
    ],
  ];

  const advancedStats: StatValue[][] = [
    [
      {
        label: "Max Pixels",
        value: formatPixelStat(s.max_pixels),
        tooltip: "Largest image in the concept by number of pixels (width x height)",
      },
      {
        label: "Avg Pixels",
        value: formatPixelStat(s.avg_pixels),
        tooltip: "Average size of images in the concept by number of pixels (width x height)",
      },
      {
        label: "Min Pixels",
        value: formatPixelStat(s.min_pixels),
        tooltip: "Smallest image in the concept by number of pixels (width x height)",
      },
    ],
    [
      {
        label: "Max Length",
        value: formatLengthStat(s.max_length),
        tooltip: "Longest video in the concept by number of frames",
      },
      {
        label: "Avg Length",
        value: formatLengthStat(s.avg_length),
        tooltip: "Average length of videos in the concept by number of frames",
      },
      {
        label: "Min Length",
        value: formatLengthStat(s.min_length),
        tooltip: "Shortest video in the concept by number of frames",
      },
    ],
    [
      { label: "Max FPS", value: formatFpsStat(s.max_fps), tooltip: "Video in concept with highest fps" },
      { label: "Avg FPS", value: formatFpsStat(s.avg_fps), tooltip: "Average fps of videos in the concept" },
      { label: "Min FPS", value: formatFpsStat(s.min_fps), tooltip: "Video in concept with the lowest fps" },
    ],
    [
      {
        label: "Max Caption Length",
        value: formatCaptionStat(s.max_caption_length),
        tooltip: "Largest caption in concept by character count. For token count, assume ~2 tokens/word",
      },
      {
        label: "Avg Caption Length",
        value: formatCaptionStat(s.avg_caption_length),
        tooltip: "Average length of caption in concept by character count. For token count, assume ~2 tokens/word",
      },
      {
        label: "Min Caption Length",
        value: formatCaptionStat(s.min_caption_length),
        tooltip: "Smallest caption in concept by character count. For token count, assume ~2 tokens/word",
      },
    ],
  ];

  const aspectBuckets = (s.aspect_buckets ?? {}) as Record<string, number>;
  const hasAspectData = Object.values(aspectBuckets).some((v) => v > 0);
  const smallestBuckets = hasAspectData ? computeSmallestBuckets(aspectBuckets) : "-";

  if (!conceptPath) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-[var(--color-on-surface-secondary)]">
        <p className="text-sm">Set a concept path first to scan for statistics.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Toolbar */}
      <div className="flex items-center gap-3">
        <Button variant="primary" size="sm" onClick={() => runScan(false)} disabled={scanning}>
          {scanning && scanType === "basic" ? "Scanning..." : "Refresh Basic"}
        </Button>
        <Button variant="primary" size="sm" onClick={() => runScan(true)} disabled={scanning}>
          {scanning && scanType === "advanced" ? "Scanning..." : "Refresh Advanced"}
        </Button>
        <Button variant="secondary" size="sm" onClick={handleCancel} disabled={!scanning}>
          Abort Scan
        </Button>
        <span className="ml-auto text-xs text-[var(--color-on-surface-secondary)]">
          {processingTime !== "-" ? `Processed in ${processingTime}` : ""}
        </span>
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-3 py-2 rounded-[var(--radius-sm)] bg-[var(--color-error-500)]/10 border border-[var(--color-error-500)]/30 text-sm text-[var(--color-error-500)]">
          Scan failed: {error}
        </div>
      )}

      {stats === null ? (
        <div className="flex flex-col items-center justify-center py-12 text-[var(--color-on-surface-secondary)]">
          <p className="text-sm">
            Click {'"'}Refresh Basic{'"'} or {'"'}Refresh Advanced{'"'} to scan concept statistics.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {/* Basic stats */}
          {basicStats.map((row, ri) => (
            <div key={ri} className="grid grid-cols-4 gap-3">
              {row.map((stat) => (
                <div key={stat.label} className="flex flex-col gap-0.5">
                  <StatLabel label={stat.label} tooltip={stat.tooltip} />
                  <span className="text-sm text-[var(--color-on-surface)] whitespace-pre-line">{stat.value}</span>
                </div>
              ))}
            </div>
          ))}

          {/* Advanced stats */}
          {advancedStats.some((row) => row.some((stat) => stat.value !== "-")) && (
            <>
              <div className="border-t border-[var(--color-border-subtle)]" />
              {advancedStats.map((row, ri) => (
                <div key={`adv-${ri}`} className="grid grid-cols-3 gap-3">
                  {row.map((stat) => (
                    <div key={stat.label} className="flex flex-col gap-0.5">
                      <StatLabel label={stat.label} tooltip={stat.tooltip} />
                      <span className="text-xs text-[var(--color-on-surface)] whitespace-pre-line break-all">
                        {stat.value}
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </>
          )}

          {/* Aspect bucketing */}
          {hasAspectData && (
            <>
              <div className="border-t border-[var(--color-border-subtle)]" />
              <div className="grid grid-cols-[1fr_auto] gap-4 items-start">
                <div className="flex flex-col gap-1">
                  <StatLabel
                    label="Aspect Bucketing"
                    tooltip="Distribution of images across aspect ratio buckets (height/width). Buckets range from 0.25 (4:1, extremely wide) to 4 (1:4, extremely tall). Images that don't match a bucket exactly are cropped to the nearest one."
                  />
                  <AspectBucketChart buckets={aspectBuckets} />
                </div>
                <div className="flex flex-col gap-0.5 min-w-[160px]">
                  <StatLabel
                    label="Smallest Buckets"
                    tooltip="Aspect ratio buckets with the fewest images. If your batch size is larger than the smallest bucket count, those images will be ignored during training!"
                  />
                  <span className="text-xs text-[var(--color-on-surface)] whitespace-pre-line">{smallestBuckets}</span>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
