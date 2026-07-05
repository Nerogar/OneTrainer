import { ChevronLeft, ChevronRight, Dice5, ImageIcon, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { configApi } from "@/api/configApi";
import { Button } from "@/components/shared";
import type { ConceptImageConfig, ConceptTextConfig } from "@/types/generated/config";

export interface ImagePreviewPanelProps {
  conceptPath: string;
  includeSubdirectories: boolean;
  textConfig: ConceptTextConfig;
  imageConfig?: ConceptImageConfig;
}

interface ImageEntry {
  filename: string;
  path: string;
  caption: string | null;
}

export function ImagePreviewPanel({
  conceptPath,
  includeSubdirectories,
  textConfig,
  imageConfig,
}: ImagePreviewPanelProps) {
  const [images, setImages] = useState<ImageEntry[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [conceptCaption, setConceptCaption] = useState<string | null>(null);

  const [showAugmented, setShowAugmented] = useState(false);
  const [augSeed, setAugSeed] = useState<number>(() => Math.floor(Math.random() * 2 ** 30));
  const [augBase64, setAugBase64] = useState<string | null>(null);
  const [augLoading, setAugLoading] = useState(false);
  const augTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    if (!conceptPath) {
      setImages([]);
      setCurrentIndex(0);
      return;
    }

    let cancelled = false;
    setLoading(true);

    configApi
      .conceptImages(conceptPath, includeSubdirectories)
      .then((result) => {
        if (cancelled) return;
        setImages(result.images);
        setCurrentIndex(0);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setImages([]);
        setCurrentIndex(0);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [conceptPath, includeSubdirectories]);

  useEffect(() => {
    if (textConfig.prompt_source !== "concept" || !textConfig.prompt_path) {
      setConceptCaption(null);
      return;
    }

    let cancelled = false;
    configApi
      .conceptTextFile(textConfig.prompt_path)
      .then((result) => {
        if (!cancelled) setConceptCaption(result.content);
      })
      .catch(() => {
        if (!cancelled) setConceptCaption(null);
      });

    return () => {
      cancelled = true;
    };
  }, [textConfig.prompt_source, textConfig.prompt_path]);

  const handlePrev = useCallback(() => {
    setCurrentIndex((i) => Math.max(0, i - 1));
  }, []);

  const handleNext = useCallback(() => {
    setCurrentIndex((i) => Math.min(images.length - 1, i + 1));
  }, [images.length]);

  const currentImage = images[currentIndex] ?? null;

  let displayCaption = "";
  if (currentImage) {
    if (textConfig.prompt_source === "filename") {
      const stem = currentImage.filename.replace(/\.[^.]+$/, "");
      displayCaption = stem || "[Empty prompt]";
    } else if (textConfig.prompt_source === "concept") {
      displayCaption = conceptCaption ?? "[No concept file loaded]";
    } else {
      displayCaption = currentImage.caption ?? "[No caption file]";
    }
  }

  const imageUrl = currentImage ? configApi.conceptImageUrl(currentImage.path) : null;

  useEffect(() => {
    if (!showAugmented || !currentImage || !imageConfig) {
      setAugBase64(null);
      return;
    }
    setAugLoading(true);
    if (augTimer.current) clearTimeout(augTimer.current);
    augTimer.current = setTimeout(async () => {
      try {
        const res = await configApi.augmentationPreview({
          image_path: currentImage.path,
          image: imageConfig as unknown as Record<string, unknown>,
          seed: augSeed,
        });
        if (res.ok) setAugBase64(res.image_base64);
      } catch {
        setAugBase64(null);
      } finally {
        setAugLoading(false);
      }
    }, 250);
    return () => {
      if (augTimer.current) clearTimeout(augTimer.current);
    };
  }, [showAugmented, currentImage, imageConfig, augSeed]);

  const rerollAugSeed = () => setAugSeed(Math.floor(Math.random() * 2 ** 30));

  return (
    <div className="w-[300px] flex-shrink-0 flex flex-col gap-3">
      <div className="w-[300px] h-[300px] rounded-[var(--radius-sm)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] flex items-center justify-center overflow-hidden relative">
        {loading ? (
          <span className="text-xs text-[var(--color-on-surface-secondary)]">Loading...</span>
        ) : showAugmented && augBase64 ? (
          <img
            src={`data:image/png;base64,${augBase64}`}
            alt="Augmented preview"
            className="max-w-full max-h-full object-contain"
          />
        ) : imageUrl ? (
          <img src={imageUrl} alt={currentImage?.filename ?? ""} className="max-w-full max-h-full object-contain" />
        ) : (
          <ImageIcon className="w-16 h-16 text-[var(--color-on-surface-secondary)]" strokeWidth={1} />
        )}
        {showAugmented && augLoading && (
          <span className="absolute bottom-1 right-1 text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-surface-container)] text-[var(--color-on-surface-secondary)]">
            updating…
          </span>
        )}
      </div>

      {imageConfig && (
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-xs cursor-pointer text-[var(--color-on-surface)]">
            <input
              type="checkbox"
              checked={showAugmented}
              onChange={(e) => setShowAugmented(e.target.checked)}
              className="accent-[var(--color-cobalt-600)]"
            />
            <Sparkles className="w-3 h-3" /> Augmentations
          </label>
          {showAugmented && (
            <Button variant="secondary" size="sm" onClick={rerollAugSeed} title="Reroll augmentation seed">
              <Dice5 className="w-3.5 h-3.5" />
            </Button>
          )}
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button variant="secondary" size="sm" onClick={handlePrev} disabled={currentIndex <= 0 || images.length === 0}>
          <ChevronLeft className="w-4 h-4" />
        </Button>
        <span className="flex-1 text-center text-xs text-[var(--color-on-surface-secondary)]">
          {images.length > 0 ? `${currentIndex + 1} / ${images.length}` : "No images"}
        </span>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleNext}
          disabled={currentIndex >= images.length - 1 || images.length === 0}
        >
          <ChevronRight className="w-4 h-4" />
        </Button>
      </div>

      {currentImage && (
        <p className="text-xs text-[var(--color-on-surface-secondary)] break-all leading-snug px-1">
          {currentImage.filename}
        </p>
      )}

      <textarea
        readOnly
        value={displayCaption}
        className="w-full h-[150px] text-xs text-[var(--color-on-surface)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] rounded-[var(--radius-sm)] p-2 resize-none"
        style={{ wordWrap: "break-word" }}
      />
    </div>
  );
}
