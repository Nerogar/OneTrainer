import {
  Check,
  ChevronRight,
  Eraser,
  FileText,
  Loader2,
  MousePointerClick,
  Paintbrush,
  Pentagon,
  Redo2,
  RotateCcw,
  Save,
  Search,
  Undo2,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { configApi } from "@/api/configApi";
import { type CapabilitiesResponse, toolsApi } from "@/api/toolsApi";
import { MaskEditorCanvas, type MaskEditorCanvasHandle } from "@/components/mask-editor/MaskEditorCanvas";
import { Button, DirPicker, FilePicker, SliderEntry, Toggle } from "@/components/shared";
import { type MaskTool, useMaskEditor } from "@/hooks/useMaskEditor";

interface MaskEditorPageProps {
  initialFolder?: string;
}

export default function MaskEditorPage({ initialFolder }: MaskEditorPageProps) {
  const { state, update, loadImages, selectImage, maskCanvasRef, pushUndo, undo, redo, runYolo, saveMask } =
    useMaskEditor(initialFolder);

  const [hoveredDetection, setHoveredDetection] = useState<number | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [yoloPanelOpen, setYoloPanelOpen] = useState(true);
  const canvasHandleRef = useRef<MaskEditorCanvasHandle | null>(null);

  // Caption editor state
  const [caption, setCaption] = useState<string>("");
  const [captionDirty, setCaptionDirty] = useState(false);
  const [captionSaving, setCaptionSaving] = useState(false);
  const captionRef = useRef<HTMLTextAreaElement | null>(null);

  // Load caption when current image changes
  useEffect(() => {
    setCaptionDirty(false);
    if (!state.currentImagePath) {
      setCaption("");
      return;
    }
    const idx = state.currentImagePath.lastIndexOf(".");
    const txtPath = idx >= 0 ? `${state.currentImagePath.slice(0, idx)}.txt` : `${state.currentImagePath}.txt`;
    let cancelled = false;
    configApi
      .conceptTextFile(txtPath)
      .then((res) => {
        if (!cancelled) setCaption(res.content ?? "");
      })
      .catch(() => {
        if (!cancelled) setCaption("");
      });
    return () => {
      cancelled = true;
    };
  }, [state.currentImagePath]);

  const saveCaption = useCallback(async () => {
    if (!state.currentImagePath || !captionDirty) return;
    setCaptionSaving(true);
    try {
      await configApi.saveCaption(state.currentImagePath, caption);
      setCaptionDirty(false);
    } catch (e) {
      alert(`Failed to save caption: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setCaptionSaving(false);
    }
  }, [state.currentImagePath, caption, captionDirty]);

  const navigate = useCallback(
    (delta: number) => {
      if (!state.currentImagePath) return;
      const idx = state.images.findIndex((img) => img.path === state.currentImagePath);
      if (idx < 0) return;
      const next = state.images[Math.max(0, Math.min(state.images.length - 1, idx + delta))];
      if (!next || next.path === state.currentImagePath) return;
      if ((state.maskModified || captionDirty) && !confirm("Unsaved changes. Continue?")) return;
      selectImage(next.path);
    },
    [state.currentImagePath, state.images, state.maskModified, captionDirty, selectImage],
  );

  useEffect(() => {
    toolsApi
      .getCapabilities()
      .then(setCapabilities)
      .catch(() => {
        /* ignore */
      });
  }, []);

  useEffect(() => {
    if (state.folder) loadImages(state.folder, state.includeSubdirs);
  }, [state.folder, state.includeSubdirs, loadImages]);

  useEffect(() => {
    if (initialFolder) {
      update("folder", initialFolder);
    }
  }, [initialFolder, update]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const inEditor =
        target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT" || target.isContentEditable);

      if (e.ctrlKey && e.key === "z") {
        e.preventDefault();
        undo();
      } else if (e.ctrlKey && e.key === "y") {
        e.preventDefault();
        redo();
      } else if (e.ctrlKey && e.key === "s") {
        e.preventDefault();
        saveMask();
        if (captionDirty) void saveCaption();
      } else if (!inEditor && (e.key === "ArrowDown" || e.key === "ArrowRight")) {
        e.preventDefault();
        navigate(1);
      } else if (!inEditor && (e.key === "ArrowUp" || e.key === "ArrowLeft")) {
        e.preventDefault();
        navigate(-1);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [undo, redo, saveMask, navigate, saveCaption, captionDirty]);

  const handleCanvasReady = useCallback((handle: MaskEditorCanvasHandle) => {
    canvasHandleRef.current = handle;
  }, []);

  const handleAcceptDetection = (index: number) => {
    const det = state.yoloDetections[index];
    if (det && canvasHandleRef.current) {
      canvasHandleRef.current.acceptDetection(det);
    }
  };

  const toolButtons: Array<{ tool: MaskTool; icon: typeof Paintbrush; label: string }> = [
    { tool: "brush", icon: Paintbrush, label: "Brush" },
    { tool: "eraser", icon: Eraser, label: "Eraser" },
    { tool: "polygon", icon: Pentagon, label: "Polygon" },
  ];

  return (
    <div className="flex h-screen bg-[var(--color-surface)] text-[var(--color-on-surface)]">
      <div className="w-64 flex-shrink-0 border-r border-[var(--color-border-subtle)] flex flex-col bg-[var(--color-surface-elevated)]">
        <div className="p-3 border-b border-[var(--color-border-subtle)] flex flex-col gap-2">
          <DirPicker label="Folder" value={state.folder} onChange={(v) => update("folder", v)} />
          <Toggle
            label="Include subfolders"
            value={state.includeSubdirs}
            onChange={(v) => update("includeSubdirs", v)}
          />
        </div>
        <div className="flex-1 overflow-y-auto">
          {state.images.length === 0 && state.folder && (
            <p className="p-3 text-xs text-[var(--color-on-surface-secondary)]">No images found</p>
          )}
          {state.images.map((img) => (
            <button
              key={img.path}
              type="button"
              onClick={() => {
                if (state.maskModified && !confirm("Unsaved changes. Continue?")) return;
                selectImage(img.path);
              }}
              className={`w-full text-left px-3 py-2 text-sm flex items-center gap-2 hover:bg-[var(--color-border-subtle)]/50 transition-colors ${
                state.currentImagePath === img.path
                  ? "bg-[var(--color-cobalt-600)]/10 border-l-2 border-[var(--color-cobalt-600)]"
                  : ""
              }`}
            >
              {img.has_mask && <span className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" />}
              <span className="truncate">{img.filename}</span>
            </button>
          ))}
        </div>
        <div className="p-2 border-t border-[var(--color-border-subtle)] text-xs text-[var(--color-on-surface-secondary)]">
          {state.images.length} images
        </div>
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center gap-2 p-2 border-b border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] flex-shrink-0">
          {toolButtons.map(({ tool: t, icon: Icon, label }) => (
            <button
              key={t}
              type="button"
              onClick={() => update("tool", t)}
              title={label}
              className={`p-2 rounded-[var(--radius-sm)] transition-colors ${
                state.tool === t ? "bg-[var(--color-cobalt-600)] text-white" : "hover:bg-[var(--color-border-subtle)]"
              }`}
            >
              <Icon className="w-4 h-4" />
            </button>
          ))}

          <div className="w-px h-6 bg-[var(--color-border-subtle)] mx-1" />

          {(state.tool === "brush" || state.tool === "eraser") && (
            <div className="w-40">
              <SliderEntry
                label="Size"
                value={state.brushSize}
                onChange={(v) => update("brushSize", v)}
                min={1}
                max={100}
                step={1}
              />
            </div>
          )}

          <div className="w-px h-6 bg-[var(--color-border-subtle)] mx-1" />

          <button
            type="button"
            onClick={undo}
            title="Undo (Ctrl+Z)"
            className="p-2 rounded-[var(--radius-sm)] hover:bg-[var(--color-border-subtle)] disabled:opacity-30"
          >
            <Undo2 className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={redo}
            title="Redo (Ctrl+Y)"
            className="p-2 rounded-[var(--radius-sm)] hover:bg-[var(--color-border-subtle)] disabled:opacity-30"
          >
            <Redo2 className="w-4 h-4" />
          </button>

          <div className="w-px h-6 bg-[var(--color-border-subtle)] mx-1" />

          <span className="text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
            {Math.round(state.zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => {
              update("zoom", 1);
              update("panOffset", { x: 0, y: 0 });
            }}
            title="Reset zoom"
            className="p-2 rounded-[var(--radius-sm)] hover:bg-[var(--color-border-subtle)]"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <div className="w-28">
              <SliderEntry
                label="Smooth"
                value={state.smooth}
                onChange={(v) => update("smooth", v)}
                min={0}
                max={20}
                step={1}
              />
            </div>
            <div className="w-28">
              <SliderEntry
                label="Expand"
                value={state.expand}
                onChange={(v) => update("expand", v)}
                min={0}
                max={20}
                step={1}
              />
            </div>
          </div>

          <div className="w-px h-6 bg-[var(--color-border-subtle)] mx-1" />

          <Button variant="primary" onClick={saveMask} disabled={!state.maskModified || state.saving}>
            {state.saving ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : <Save className="w-4 h-4 mr-1" />}
            Save
          </Button>
        </div>

        <MaskEditorCanvas
          imagePath={state.currentImagePath}
          tool={state.tool}
          brushSize={state.brushSize}
          zoom={state.zoom}
          panOffset={state.panOffset}
          onZoomChange={(z) => update("zoom", z)}
          onPanChange={(p) => update("panOffset", p)}
          maskCanvasRef={maskCanvasRef}
          onBeforeStroke={pushUndo}
          yoloDetections={state.yoloDetections}
          hoveredDetection={hoveredDetection}
          onReady={handleCanvasReady}
        />

        {state.currentImagePath && (
          <div className="border-t border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] p-3 flex flex-col gap-2 max-h-[40vh]">
            <div className="flex items-center gap-2 text-xs text-[var(--color-on-surface-secondary)]">
              <FileText className="w-3.5 h-3.5" />
              <span className="truncate flex-1">
                Caption:{" "}
                {state.currentImagePath
                  .split(/[/\\]/)
                  .pop()
                  ?.replace(/\.[^.]+$/, "")}
                .txt
              </span>
              {captionDirty && <span className="text-[var(--color-cobalt-600)]">(unsaved)</span>}
              <Button variant="primary" size="sm" onClick={saveCaption} disabled={!captionDirty || captionSaving}>
                {captionSaving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                <span className="ml-1">Save Caption</span>
              </Button>
            </div>
            <textarea
              ref={captionRef}
              value={caption}
              onChange={(e) => {
                setCaption(e.target.value);
                setCaptionDirty(true);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  void saveCaption();
                }
              }}
              placeholder="Enter caption for this image. Ctrl+Enter to save."
              className="w-full min-h-[80px] flex-1 resize-y rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] p-2 text-sm font-mono text-[var(--color-on-surface)]"
              spellCheck={false}
            />
          </div>
        )}
      </div>

      <div
        className={`flex-shrink-0 border-l border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] transition-all ${
          yoloPanelOpen ? "w-72" : "w-10"
        }`}
      >
        <button
          type="button"
          onClick={() => setYoloPanelOpen(!yoloPanelOpen)}
          className="w-full p-2 flex items-center justify-center hover:bg-[var(--color-border-subtle)]"
          title={yoloPanelOpen ? "Collapse YOLO panel" : "Expand YOLO panel"}
        >
          <ChevronRight className={`w-4 h-4 transition-transform ${yoloPanelOpen ? "rotate-180" : ""}`} />
        </button>

        {yoloPanelOpen && (
          <div className="p-3 flex flex-col gap-3">
            <h3 className="text-sm font-semibold flex items-center gap-1">
              <MousePointerClick className="w-4 h-4" /> YOLO Detection
            </h3>

            {capabilities && !capabilities.ultralytics_available && (
              <div className="p-2 rounded-[var(--radius-sm)] bg-[var(--color-error-500)]/10 border border-[var(--color-error-500)]/30">
                <p className="text-xs text-[var(--color-error-500)]">
                  Ultralytics not installed.
                  <br />
                  <code className="text-[10px]">pip install ultralytics</code>
                </p>
              </div>
            )}

            <FilePicker
              label="Model File"
              value={state.yoloModelPath}
              onChange={(v) => update("yoloModelPath", v)}
              filters={[{ name: "YOLO Models", extensions: ["pt"] }]}
              tooltip="Select a YOLO .pt model for object detection"
            />

            <Button
              variant="secondary"
              onClick={runYolo}
              disabled={
                !state.currentImagePath ||
                !state.yoloModelPath ||
                state.yoloLoading ||
                (capabilities !== null && !capabilities.ultralytics_available)
              }
            >
              {state.yoloLoading ? (
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
              ) : (
                <Search className="w-4 h-4 mr-1" />
              )}
              Find Objects
            </Button>

            {state.yoloDetections.length > 0 && (
              <div className="flex flex-col gap-1 max-h-80 overflow-y-auto">
                <p className="text-xs text-[var(--color-on-surface-secondary)]">
                  {state.yoloDetections.length} detections
                </p>
                {state.yoloDetections.map((det, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 p-2 rounded-[var(--radius-sm)] border border-[var(--color-border-subtle)] hover:bg-[var(--color-border-subtle)]/50 cursor-pointer"
                    onMouseEnter={() => setHoveredDetection(i)}
                    onMouseLeave={() => setHoveredDetection(null)}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{det.class_name}</p>
                      <p className="text-xs text-[var(--color-on-surface-secondary)]">
                        {(det.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleAcceptDetection(i)}
                      title="Accept detection as mask"
                      className="p-1 rounded-[var(--radius-sm)] text-green-500 hover:bg-green-500/10"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
