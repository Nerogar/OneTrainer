import { useCallback, useEffect, useState } from "react";

import { videoToolsApi, type VideoToolStatusResponse } from "@/api/videoToolsApi";
import { Button, DirPicker, FilePicker, FormEntry, FormFieldWrapper, Toggle } from "@/components/shared";
import { INPUT_FULL } from "@/utils/inputStyles";

import { ModalBase } from "./ModalBase";

export interface VideoToolModalProps {
  open: boolean;
  onClose: () => void;
}

type TabId = "clips" | "images" | "download";

const TAB_BASE =
  "px-4 py-2 text-sm font-medium rounded-t-[var(--radius-sm)] border border-b-0 transition-colors duration-200 cursor-pointer";
const TAB_ACTIVE = "bg-[var(--color-surface)] text-[var(--color-on-surface)] border-[var(--color-border-subtle)]";
const TAB_INACTIVE =
  "bg-transparent text-[var(--color-on-surface-secondary)] border-transparent hover:text-[var(--color-on-surface)] hover:border-[var(--color-border-subtle)]";

function StatusBar({ status }: { status: VideoToolStatusResponse | null }) {
  if (!status || status.status === "idle") return null;

  const colorClass =
    status.status === "error"
      ? "text-[var(--color-error-500)]"
      : status.status === "completed"
        ? "text-[var(--color-success-500)]"
        : "text-[var(--color-cobalt-600)]";

  return (
    <div className="mt-4 pt-3 border-t border-[var(--color-border-subtle)]">
      <div className="flex items-center gap-2">
        <span className={`text-sm font-medium ${colorClass}`}>
          {status.status === "running" && "Running..."}
          {status.status === "completed" && "Completed"}
          {status.status === "error" && "Error"}
        </span>
        {status.message && (
          <span className="text-sm text-[var(--color-on-surface-secondary)] truncate">{status.message}</span>
        )}
        {status.error && <span className="text-sm text-[var(--color-error-500)] truncate">{status.error}</span>}
      </div>
    </div>
  );
}

function ExtractClipsTab({ onStatusChange }: { onStatusChange: () => void }) {
  const [videoPath, setVideoPath] = useState("");
  const [directory, setDirectory] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [timeStart, setTimeStart] = useState("00:00:00");
  const [timeEnd, setTimeEnd] = useState("99:99:99");
  const [outputSubdirectories, setOutputSubdirectories] = useState(false);
  const [splitAtCuts, setSplitAtCuts] = useState(false);
  const [maxLength, setMaxLength] = useState<number>(3);
  const [fps, setFps] = useState<number>(24);
  const [removeBorders, setRemoveBorders] = useState(false);
  const [cropVariation, setCropVariation] = useState<number>(0.2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExtract = async (batchMode: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const result = await videoToolsApi.extractClips({
        video_path: videoPath,
        directory,
        batch_mode: batchMode,
        output_dir: outputDir,
        time_start: timeStart,
        time_end: timeEnd,
        output_subdirectories: outputSubdirectories,
        split_at_cuts: splitAtCuts,
        max_length: maxLength,
        fps,
        remove_borders: removeBorders,
        crop_variation: cropVariation,
      });
      if (!result.ok) {
        setError(result.error ?? "Unknown error");
      }
      onStatusChange();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <FilePicker
            label="Single Video"
            value={videoPath}
            onChange={setVideoPath}
            tooltip="Link to single video file to process."
            filters={[
              { name: "Video files", extensions: ["webm", "mkv", "flv", "avi", "mov", "wmv", "mp4", "mpeg", "m4v"] },
            ]}
          />
        </div>
        <Button variant="secondary" size="sm" onClick={() => handleExtract(false)} loading={loading} disabled={loading}>
          Extract Single
        </Button>
      </div>

      <FormFieldWrapper
        label="Time Range"
        tooltip="Time range to limit selection for single video. Format: HH:MM:SS, MM:SS, or seconds."
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={timeStart}
            onChange={(e) => setTimeStart(e.target.value)}
            className={INPUT_FULL}
            placeholder="00:00:00"
          />
          <input
            type="text"
            value={timeEnd}
            onChange={(e) => setTimeEnd(e.target.value)}
            className={INPUT_FULL}
            placeholder="99:99:99"
          />
        </div>
      </FormFieldWrapper>

      <div className="flex items-end gap-2">
        <div className="flex-1">
          <DirPicker
            label="Directory"
            value={directory}
            onChange={setDirectory}
            tooltip="Path to directory with multiple videos to process, including in subdirectories."
          />
        </div>
        <Button variant="secondary" size="sm" onClick={() => handleExtract(true)} loading={loading} disabled={loading}>
          Extract Directory
        </Button>
      </div>

      <DirPicker
        label="Output"
        value={outputDir}
        onChange={setOutputDir}
        tooltip="Path to folder where extracted clips will be saved."
      />

      <div className="grid grid-cols-2 gap-4">
        <Toggle
          label="Output to Subdirectories"
          value={outputSubdirectories}
          onChange={setOutputSubdirectories}
          tooltip="If enabled, files are saved to subfolders based on filename and input directory."
        />
        <Toggle
          label="Split at Cuts"
          value={splitAtCuts}
          onChange={setSplitAtCuts}
          tooltip="If enabled, detect cuts in the input video and split at those points."
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <FormEntry
          label="Max Length (s)"
          value={maxLength}
          onChange={(v) => setMaxLength(Number(v))}
          type="number"
          tooltip="Maximum length in seconds for saved clips. Larger clips will be broken into multiple small clips."
        />
        <FormEntry
          label="Set FPS"
          value={fps}
          onChange={(v) => setFps(Number(v))}
          type="number"
          tooltip="FPS to convert output videos to. Set to 0 to keep original rate."
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Toggle
          label="Remove Borders"
          value={removeBorders}
          onChange={setRemoveBorders}
          tooltip="Remove black borders from output clip."
        />
        <FormEntry
          label="Crop Variation"
          value={cropVariation}
          onChange={(v) => setCropVariation(Number(v))}
          type="number"
          tooltip="Output clips will be randomly cropped to +/- the base aspect ratio. Set to 0 to use only base aspect."
        />
      </div>

      {error && <p className="text-sm text-[var(--color-error-500)]">{error}</p>}
    </div>
  );
}

function ExtractImagesTab({ onStatusChange }: { onStatusChange: () => void }) {
  const [videoPath, setVideoPath] = useState("");
  const [directory, setDirectory] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [timeStart, setTimeStart] = useState("00:00:00");
  const [timeEnd, setTimeEnd] = useState("99:99:99");
  const [outputSubdirectories, setOutputSubdirectories] = useState(false);
  const [imagesPerSecond, setImagesPerSecond] = useState<number>(0.5);
  const [blurRemoval, setBlurRemoval] = useState<number>(0.2);
  const [removeBorders, setRemoveBorders] = useState(false);
  const [cropVariation, setCropVariation] = useState<number>(0.2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExtract = async (batchMode: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const result = await videoToolsApi.extractImages({
        video_path: videoPath,
        directory,
        batch_mode: batchMode,
        output_dir: outputDir,
        time_start: timeStart,
        time_end: timeEnd,
        output_subdirectories: outputSubdirectories,
        images_per_second: imagesPerSecond,
        blur_removal: blurRemoval,
        remove_borders: removeBorders,
        crop_variation: cropVariation,
      });
      if (!result.ok) {
        setError(result.error ?? "Unknown error");
      }
      onStatusChange();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <FilePicker
            label="Single Video"
            value={videoPath}
            onChange={setVideoPath}
            tooltip="Link to single video file to process."
            filters={[
              { name: "Video files", extensions: ["webm", "mkv", "flv", "avi", "mov", "wmv", "mp4", "mpeg", "m4v"] },
            ]}
          />
        </div>
        <Button variant="secondary" size="sm" onClick={() => handleExtract(false)} loading={loading} disabled={loading}>
          Extract Single
        </Button>
      </div>

      <FormFieldWrapper
        label="Time Range"
        tooltip="Time range to limit selection for single video. Format: HH:MM:SS, MM:SS, or seconds."
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={timeStart}
            onChange={(e) => setTimeStart(e.target.value)}
            className={INPUT_FULL}
            placeholder="00:00:00"
          />
          <input
            type="text"
            value={timeEnd}
            onChange={(e) => setTimeEnd(e.target.value)}
            className={INPUT_FULL}
            placeholder="99:99:99"
          />
        </div>
      </FormFieldWrapper>

      <div className="flex items-end gap-2">
        <div className="flex-1">
          <DirPicker
            label="Directory"
            value={directory}
            onChange={setDirectory}
            tooltip="Path to directory with multiple videos to process, including in subdirectories."
          />
        </div>
        <Button variant="secondary" size="sm" onClick={() => handleExtract(true)} loading={loading} disabled={loading}>
          Extract Directory
        </Button>
      </div>

      <DirPicker
        label="Output"
        value={outputDir}
        onChange={setOutputDir}
        tooltip="Path to folder where extracted images will be saved."
      />

      <Toggle
        label="Output to Subdirectories"
        value={outputSubdirectories}
        onChange={setOutputSubdirectories}
        tooltip="If enabled, files are saved to subfolders based on filename and input directory."
      />

      <div className="grid grid-cols-2 gap-4">
        <FormEntry
          label="Images/sec"
          value={imagesPerSecond}
          onChange={(v) => setImagesPerSecond(Number(v))}
          type="number"
          tooltip="Number of images to capture per second of video. Images will be taken at semi-random frames around the specified frequency."
        />
        <FormEntry
          label="Blur Removal"
          value={blurRemoval}
          onChange={(v) => setBlurRemoval(Number(v))}
          type="number"
          tooltip="Threshold for removal of blurry images, relative to all others. At 0.2, the blurriest 20% of frames will not be saved."
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Toggle
          label="Remove Borders"
          value={removeBorders}
          onChange={setRemoveBorders}
          tooltip="Remove black borders from output image."
        />
        <FormEntry
          label="Crop Variation"
          value={cropVariation}
          onChange={(v) => setCropVariation(Number(v))}
          type="number"
          tooltip="Output images will be randomly cropped to +/- the base aspect ratio. Set to 0 to use only base aspect."
        />
      </div>

      {error && <p className="text-sm text-[var(--color-error-500)]">{error}</p>}
    </div>
  );
}

function DownloadTab({ onStatusChange }: { onStatusChange: () => void }) {
  const [url, setUrl] = useState("");
  const [linkListPath, setLinkListPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [additionalArgs, setAdditionalArgs] = useState("--quiet --no-warnings --progress");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async (batchMode: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const result = await videoToolsApi.download({
        url,
        link_list_path: linkListPath,
        batch_mode: batchMode,
        output_dir: outputDir,
        additional_args: additionalArgs,
      });
      if (!result.ok) {
        setError(result.error ?? "Unknown error");
      }
      onStatusChange();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <FormEntry
            label="Single Link"
            value={url}
            onChange={(v) => setUrl(String(v))}
            type="text"
            tooltip="Link to video/playlist to download. Uses yt-dlp; supports YouTube, Twitch, Instagram, and many other sites."
            placeholder="https://..."
          />
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => handleDownload(false)}
          loading={loading}
          disabled={loading}
        >
          Download Link
        </Button>
      </div>

      <div className="flex items-end gap-2">
        <div className="flex-1">
          <FilePicker
            label="Link List"
            value={linkListPath}
            onChange={setLinkListPath}
            tooltip="Path to .txt file with list of links separated by newlines."
            filters={[{ name: "Text file", extensions: ["txt"] }]}
          />
        </div>
        <Button variant="secondary" size="sm" onClick={() => handleDownload(true)} loading={loading} disabled={loading}>
          Download List
        </Button>
      </div>

      <DirPicker
        label="Output"
        value={outputDir}
        onChange={setOutputDir}
        tooltip="Path to folder where downloaded videos will be saved."
      />

      <FormFieldWrapper
        label="Additional Args"
        tooltip="Any additional arguments to pass to yt-dlp, for example '--restrict-filenames --force-overwrite'. Default args will hide most terminal outputs."
      >
        <textarea
          value={additionalArgs}
          onChange={(e) => setAdditionalArgs(e.target.value)}
          rows={3}
          className="px-3 py-2 rounded-[var(--radius-sm)] text-sm bg-[var(--color-input-bg)] border border-[var(--color-border-subtle)] text-[var(--color-on-surface)] focus:outline-none focus:border-[var(--color-cobalt-600)] focus:shadow-[var(--shadow-inset-focus)] resize-y w-full transition-[border-color,box-shadow,background-color] duration-200 ease-out"
        />
      </FormFieldWrapper>

      <div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => window.open("https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#usage-and-options", "_blank")}
        >
          yt-dlp info
        </Button>
      </div>

      {error && <p className="text-sm text-[var(--color-error-500)]">{error}</p>}
    </div>
  );
}

export function VideoToolModal({ open, onClose }: VideoToolModalProps) {
  const [activeTab, setActiveTab] = useState<TabId>("clips");
  const [status, setStatus] = useState<VideoToolStatusResponse | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await videoToolsApi.getStatus();
      setStatus(s);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    fetchStatus();
    const interval = setInterval(() => {
      fetchStatus();
    }, 2000);
    return () => clearInterval(interval);
  }, [open, fetchStatus]);

  const tabs: Array<{ id: TabId; label: string }> = [
    { id: "clips", label: "Extract Clips" },
    { id: "images", label: "Extract Images" },
    { id: "download", label: "Download" },
  ];

  return (
    <ModalBase open={open} onClose={onClose} title="Video Tools" size="lg">
      <div className="flex gap-1 mb-4 border-b border-[var(--color-border-subtle)]">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`${TAB_BASE} ${activeTab === tab.id ? TAB_ACTIVE : TAB_INACTIVE}`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "clips" && <ExtractClipsTab onStatusChange={fetchStatus} />}
      {activeTab === "images" && <ExtractImagesTab onStatusChange={fetchStatus} />}
      {activeTab === "download" && <DownloadTab onStatusChange={fetchStatus} />}

      <StatusBar status={status} />

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
