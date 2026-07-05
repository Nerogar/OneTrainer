import { X } from "lucide-react";
import { type ReactNode, useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";

import { IconButton } from "@/components/shared";

export interface ModalBaseProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  size?: "sm" | "md" | "lg" | "xl" | "2xl" | "full";
  closeOnBackdrop?: boolean;
  closeOnEscape?: boolean;
}

const sizeClasses: Record<string, string> = {
  sm: "max-w-[400px]",
  md: "max-w-[600px]",
  lg: "max-w-[800px]",
  xl: "max-w-[1000px]",
  "2xl": "max-w-[1200px]",
  full: "max-w-none w-[calc(100vw-32px)]",
};

export function ModalBase({
  open,
  onClose,
  title,
  children,
  size = "md",
  closeOnBackdrop = true,
  closeOnEscape = true,
}: ModalBaseProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useEffect(() => {
    if (!open || !closeOnEscape) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [open, onClose, closeOnEscape]);

  useEffect(() => {
    if (!open) return;

    const modal = dialogRef.current;
    if (!modal) return;

    const previouslyFocused = document.activeElement as HTMLElement | null;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const focusable = modal.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    modal.addEventListener("keydown", handleKeyDown);

    setTimeout(() => {
      const firstFocusable = modal.querySelector<HTMLElement>("button, input, select, textarea");
      firstFocusable?.focus();
    }, 50);

    return () => {
      modal.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [open]);

  if (!open) return null;

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (closeOnBackdrop && e.target === e.currentTarget) onClose();
  };

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={handleBackdropClick}
      style={{ backdropFilter: "blur(8px)", background: "rgba(0, 0, 0, 0.5)" }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`w-full ${sizeClasses[size]} ${size === "full" ? "h-[calc(100vh-32px)]" : "max-h-[90vh]"} flex flex-col rounded-[var(--radius-md)]
          bg-[var(--color-surface)] border border-[var(--color-border-subtle)]
          shadow-[var(--shadow-3)] animate-[rowFade_200ms_ease-out]`}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--color-border-subtle)]">
          <h2 id={titleId} className="text-lg font-semibold text-[var(--color-on-surface)]">
            {title}
          </h2>
          <IconButton
            icon={<X className="w-full h-full" />}
            label="Close"
            variant="ghost"
            size="sm"
            onClick={onClose}
          />
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-4">{children}</div>
      </div>
    </div>,
    document.body,
  );
}
