import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

import { toolsApi, type YoloDetection } from "@/api/toolsApi";
import type { MaskTool } from "@/hooks/useMaskEditor";

export interface MaskEditorCanvasHandle {
  acceptDetection: (detection: YoloDetection) => void;
}

interface MaskEditorCanvasProps {
  imagePath: string | null;
  tool: MaskTool;
  brushSize: number;
  zoom: number;
  panOffset: { x: number; y: number };
  onZoomChange: (zoom: number) => void;
  onPanChange: (offset: { x: number; y: number }) => void;
  maskCanvasRef: RefObject<HTMLCanvasElement | null>;
  onBeforeStroke: () => void;
  yoloDetections: YoloDetection[];
  hoveredDetection: number | null;
  onReady?: (handle: MaskEditorCanvasHandle) => void;
}

export function MaskEditorCanvas({
  imagePath,
  tool,
  brushSize,
  zoom,
  panOffset,
  onZoomChange,
  onPanChange,
  maskCanvasRef,
  onBeforeStroke,
  yoloDetections,
  hoveredDetection,
  onReady,
}: MaskEditorCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgCanvasRef = useRef<HTMLCanvasElement>(null);
  const cursorCanvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  const isDrawingRef = useRef(false);
  const isPanningRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const panStartRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);
  const polygonPointsRef = useRef<Array<{ x: number; y: number }>>([]);

  const getTransform = useCallback(() => {
    if (imageSize.width === 0 || imageSize.height === 0) return { scale: 1, ox: 0, oy: 0 };
    const baseScale = Math.min(canvasSize.width / imageSize.width, canvasSize.height / imageSize.height);
    const scale = baseScale * zoom;
    const ox = (canvasSize.width - imageSize.width * scale) / 2 + panOffset.x;
    const oy = (canvasSize.height - imageSize.height * scale) / 2 + panOffset.y;
    return { scale, ox, oy };
  }, [canvasSize, imageSize, zoom, panOffset]);

  const screenToImage = useCallback(
    (sx: number, sy: number) => {
      const { scale, ox, oy } = getTransform();
      return { x: (sx - ox) / scale, y: (sy - oy) / scale };
    },
    [getTransform],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    obs.observe(container);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!imagePath) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imageRef.current = img;
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });

      const maskCanvas = maskCanvasRef.current;
      if (maskCanvas) {
        maskCanvas.width = img.naturalWidth;
        maskCanvas.height = img.naturalHeight;
        const ctx = maskCanvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      }

      const maskUrl = toolsApi.getMaskEditorImageUrl(imagePath.replace(/(\.[^.]+)$/, "-masklabel.png"));
      const maskImg = new Image();
      maskImg.crossOrigin = "anonymous";
      maskImg.onload = () => {
        if (maskCanvas) {
          const ctx = maskCanvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            ctx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
          }
        }
      };
      maskImg.onerror = () => {
        /* no existing mask, canvas stays clear */
      };
      maskImg.src = maskUrl;
    };
    img.src = toolsApi.getMaskEditorImageUrl(imagePath);
  }, [imagePath, maskCanvasRef]);

  const renderImage = useCallback(() => {
    const canvas = imgCanvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { scale, ox, oy } = getTransform();
    ctx.drawImage(img, ox, oy, imageSize.width * scale, imageSize.height * scale);

    const maskCanvas = maskCanvasRef.current;
    if (maskCanvas) {
      ctx.globalAlpha = 0.4;
      ctx.drawImage(maskCanvas, ox, oy, imageSize.width * scale, imageSize.height * scale);
      ctx.globalAlpha = 1.0;
    }
  }, [canvasSize, imageSize, getTransform, maskCanvasRef]);

  const renderCursor = useCallback(
    (mouseX?: number, mouseY?: number) => {
      const canvas = cursorCanvasRef.current;
      if (!canvas) return;
      canvas.width = canvasSize.width;
      canvas.height = canvasSize.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const { scale, ox, oy } = getTransform();

      if ((tool === "brush" || tool === "eraser") && mouseX !== undefined && mouseY !== undefined) {
        ctx.beginPath();
        ctx.arc(mouseX, mouseY, (brushSize / 2) * scale, 0, Math.PI * 2);
        ctx.strokeStyle = tool === "brush" ? "rgba(255,255,255,0.8)" : "rgba(255,0,0,0.8)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      if (tool === "polygon" && polygonPointsRef.current.length > 0) {
        ctx.beginPath();
        const pts = polygonPointsRef.current;
        for (let i = 0; i < pts.length; i++) {
          const sx = pts[i].x * scale + ox;
          const sy = pts[i].y * scale + oy;
          if (i === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        if (mouseX !== undefined && mouseY !== undefined) {
          ctx.lineTo(mouseX, mouseY);
        }
        ctx.strokeStyle = "rgba(255,255,0,0.8)";
        ctx.lineWidth = 2;
        ctx.stroke();

        for (const pt of pts) {
          ctx.beginPath();
          ctx.arc(pt.x * scale + ox, pt.y * scale + oy, 4, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(255,255,0,1)";
          ctx.fill();
        }
      }

      if (hoveredDetection !== null && hoveredDetection < yoloDetections.length) {
        const det = yoloDetections[hoveredDetection];
        if (det.polygon.length > 0) {
          ctx.beginPath();
          for (let i = 0; i < det.polygon.length; i++) {
            const px = det.polygon[i][0] * imageSize.width * scale + ox;
            const py = det.polygon[i][1] * imageSize.height * scale + oy;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.fillStyle = "rgba(37, 99, 235, 0.3)";
          ctx.fill();
          ctx.strokeStyle = "rgba(37, 99, 235, 0.8)";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    },
    [canvasSize, getTransform, tool, brushSize, yoloDetections, hoveredDetection, imageSize],
  );

  useEffect(() => {
    renderImage();
    renderCursor();
  }, [renderImage, renderCursor]);

  const paintAt = useCallback(
    (imgX: number, imgY: number, erase: boolean) => {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) return;
      const ctx = maskCanvas.getContext("2d");
      if (!ctx) return;

      if (erase) {
        ctx.globalCompositeOperation = "destination-out";
      } else {
        ctx.globalCompositeOperation = "source-over";
      }

      ctx.beginPath();
      ctx.arc(imgX, imgY, brushSize / 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 1)";
      ctx.fill();
      ctx.globalCompositeOperation = "source-over";
    },
    [brushSize, maskCanvasRef],
  );

  const interpolateStroke = useCallback(
    (from: { x: number; y: number }, to: { x: number; y: number }, erase: boolean) => {
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const steps = Math.max(1, Math.ceil(dist / (brushSize / 4)));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        paintAt(from.x + dx * t, from.y + dy * t, erase);
      }
    },
    [brushSize, paintAt],
  );

  const getCanvasCoords = (e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const { x, y } = getCanvasCoords(e);

    if (e.altKey || e.button === 1) {
      isPanningRef.current = true;
      panStartRef.current = { x, y, ox: panOffset.x, oy: panOffset.y };
      return;
    }

    if (e.button !== 0) return;

    if (tool === "brush" || tool === "eraser") {
      onBeforeStroke();
      isDrawingRef.current = true;
      const imgPt = screenToImage(x, y);
      lastPointRef.current = imgPt;
      paintAt(imgPt.x, imgPt.y, tool === "eraser");
      renderImage();
    } else if (tool === "polygon") {
      const imgPt = screenToImage(x, y);
      polygonPointsRef.current.push(imgPt);
      renderCursor(x, y);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const { x, y } = getCanvasCoords(e);

    if (isPanningRef.current && panStartRef.current) {
      const dx = x - panStartRef.current.x;
      const dy = y - panStartRef.current.y;
      onPanChange({ x: panStartRef.current.ox + dx, y: panStartRef.current.oy + dy });
      return;
    }

    if (isDrawingRef.current && (tool === "brush" || tool === "eraser")) {
      const imgPt = screenToImage(x, y);
      if (lastPointRef.current) {
        interpolateStroke(lastPointRef.current, imgPt, tool === "eraser");
      }
      lastPointRef.current = imgPt;
      renderImage();
    }

    renderCursor(x, y);
  };

  const handleMouseUp = () => {
    isDrawingRef.current = false;
    isPanningRef.current = false;
    lastPointRef.current = null;
    panStartRef.current = null;
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const newZoom = Math.max(0.1, Math.min(10, zoom * factor));
    onZoomChange(newZoom);
  };

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (tool === "polygon" && e.key === "Enter" && polygonPointsRef.current.length >= 3) {
        onBeforeStroke();
        const maskCanvas = maskCanvasRef.current;
        if (maskCanvas) {
          const ctx = maskCanvas.getContext("2d");
          if (ctx) {
            ctx.beginPath();
            const pts = polygonPointsRef.current;
            ctx.moveTo(pts[0].x, pts[0].y);
            for (let i = 1; i < pts.length; i++) {
              ctx.lineTo(pts[i].x, pts[i].y);
            }
            ctx.closePath();
            ctx.fillStyle = "rgba(255, 255, 255, 1)";
            ctx.fill();
          }
        }
        polygonPointsRef.current = [];
        renderImage();
        renderCursor();
      }
      if (e.key === "Escape") {
        polygonPointsRef.current = [];
        renderCursor();
      }
    },
    [tool, onBeforeStroke, maskCanvasRef, renderImage, renderCursor],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const acceptDetection = useCallback(
    (detection: YoloDetection) => {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas || detection.polygon.length === 0) return;
      onBeforeStroke();
      const ctx = maskCanvas.getContext("2d");
      if (!ctx) return;
      ctx.beginPath();
      for (let i = 0; i < detection.polygon.length; i++) {
        const px = detection.polygon[i][0] * imageSize.width;
        const py = detection.polygon[i][1] * imageSize.height;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 255, 255, 1)";
      ctx.fill();
      renderImage();
    },
    [maskCanvasRef, imageSize, onBeforeStroke, renderImage],
  );

  useEffect(() => {
    if (onReady) onReady({ acceptDetection });
  }, [onReady, acceptDetection]);

  return (
    <div
      ref={containerRef}
      className="relative flex-1 overflow-hidden bg-[var(--color-surface-dark)] cursor-crosshair"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      onContextMenu={(e) => {
        e.preventDefault();
        if (tool === "polygon" && polygonPointsRef.current.length > 0) {
          polygonPointsRef.current.pop();
          renderCursor();
        }
      }}
    >
      <canvas ref={imgCanvasRef} width={canvasSize.width} height={canvasSize.height} className="absolute inset-0" />
      <canvas ref={maskCanvasRef} className="hidden" />
      <canvas
        ref={cursorCanvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        className="absolute inset-0 pointer-events-none"
      />
      {!imagePath && (
        <div className="absolute inset-0 flex items-center justify-center text-[var(--color-on-surface-secondary)]">
          Select an image to begin editing
        </div>
      )}
    </div>
  );
}
