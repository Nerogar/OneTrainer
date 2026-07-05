import { useCallback, useRef, useState } from "react";

import { type ImageListItem, toolsApi, type YoloDetection } from "@/api/toolsApi";

export type MaskTool = "brush" | "eraser" | "polygon";

export interface MaskEditorState {
  tool: MaskTool;
  brushSize: number;
  zoom: number;
  panOffset: { x: number; y: number };
  currentImagePath: string | null;
  maskModified: boolean;
  yoloModelPath: string;
  yoloDetections: YoloDetection[];
  yoloLoading: boolean;
  images: ImageListItem[];
  folder: string;
  includeSubdirs: boolean;
  saving: boolean;
  smooth: number;
  expand: number;
}

const MAX_UNDO = 20;

export function useMaskEditor(initialFolder?: string) {
  const [state, setState] = useState<MaskEditorState>({
    tool: "brush",
    brushSize: 20,
    zoom: 1,
    panOffset: { x: 0, y: 0 },
    currentImagePath: null,
    maskModified: false,
    yoloModelPath: "",
    yoloDetections: [],
    yoloLoading: false,
    images: [],
    folder: initialFolder ?? "",
    includeSubdirs: false,
    saving: false,
    smooth: 0,
    expand: 0,
  });

  const undoStackRef = useRef<ImageData[]>([]);
  const redoStackRef = useRef<ImageData[]>([]);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const update = useCallback(<K extends keyof MaskEditorState>(field: K, value: MaskEditorState[K]) => {
    setState((prev) => ({ ...prev, [field]: value }));
  }, []);

  const loadImages = useCallback(async (folder: string, includeSubdirs: boolean) => {
    try {
      const images = await toolsApi.listMaskEditorImages(folder, includeSubdirs);
      setState((prev) => ({ ...prev, images, folder, includeSubdirs }));
    } catch {
      setState((prev) => ({ ...prev, images: [] }));
    }
  }, []);

  const selectImage = useCallback((path: string) => {
    undoStackRef.current = [];
    redoStackRef.current = [];
    setState((prev) => ({
      ...prev,
      currentImagePath: path,
      maskModified: false,
      yoloDetections: [],
      zoom: 1,
      panOffset: { x: 0, y: 0 },
    }));
  }, []);

  const pushUndo = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    undoStackRef.current.push(data);
    if (undoStackRef.current.length > MAX_UNDO) undoStackRef.current.shift();
    redoStackRef.current = [];
    setState((prev) => ({ ...prev, maskModified: true }));
  }, []);

  const undo = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas || undoStackRef.current.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    redoStackRef.current.push(current);
    const prev = undoStackRef.current.pop();
    if (prev) ctx.putImageData(prev, 0, 0);
  }, []);

  const redo = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas || redoStackRef.current.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    undoStackRef.current.push(current);
    const next = redoStackRef.current.pop();
    if (next) ctx.putImageData(next, 0, 0);
  }, []);

  const runYolo = useCallback(async () => {
    if (!state.currentImagePath || !state.yoloModelPath) return;
    setState((prev) => ({ ...prev, yoloLoading: true }));
    try {
      const result = await toolsApi.yoloPredict(state.currentImagePath, state.yoloModelPath);
      setState((prev) => ({
        ...prev,
        yoloDetections: result.ok ? result.detections : [],
        yoloLoading: false,
      }));
    } catch {
      setState((prev) => ({ ...prev, yoloDetections: [], yoloLoading: false }));
    }
  }, [state.currentImagePath, state.yoloModelPath]);

  const saveMask = useCallback(async () => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !state.currentImagePath) return;
    setState((prev) => ({ ...prev, saving: true }));

    try {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;
      const exportCtx = exportCanvas.getContext("2d");
      if (!exportCtx) return;
      const exportData = exportCtx.createImageData(canvas.width, canvas.height);

      for (let i = 0; i < imageData.data.length; i += 4) {
        const alpha = imageData.data[i + 3];
        const value = alpha >= 128 ? 255 : 0;
        exportData.data[i] = value;
        exportData.data[i + 1] = value;
        exportData.data[i + 2] = value;
        exportData.data[i + 3] = 255;
      }
      exportCtx.putImageData(exportData, 0, 0);

      const blob = await new Promise<Blob>((resolve, reject) => {
        exportCanvas.toBlob((b) => {
          if (b) resolve(b);
          else reject(new Error("Failed to create blob"));
        }, "image/png");
      });

      await toolsApi.saveMask(state.currentImagePath, blob, state.smooth, state.expand);
      setState((prev) => ({ ...prev, maskModified: false, saving: false }));

      if (state.folder) {
        const images = await toolsApi.listMaskEditorImages(state.folder, state.includeSubdirs);
        setState((prev) => ({ ...prev, images }));
      }
    } catch {
      setState((prev) => ({ ...prev, saving: false }));
    }
  }, [state.currentImagePath, state.folder, state.includeSubdirs, state.smooth, state.expand]);

  return {
    state,
    update,
    loadImages,
    selectImage,
    maskCanvasRef,
    pushUndo,
    undo,
    redo,
    runYolo,
    saveMask,
    canUndo: undoStackRef.current.length > 0,
    canRedo: redoStackRef.current.length > 0,
  };
}
