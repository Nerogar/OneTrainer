import { type ReactNode, useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface TooltipProps {
  children: ReactNode;
  content: ReactNode;
  position?: "top" | "right" | "bottom" | "left";
  delay?: number;
}

interface Coords {
  top: number;
  left: number;
  resolvedPosition: "top" | "right" | "bottom" | "left";
}

const ARROW_SIZE = 6;
const OFFSET = 8;

function computeCoords(
  triggerRect: DOMRect,
  tooltipRect: DOMRect,
  preferred: "top" | "right" | "bottom" | "left",
): Coords {
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  const fits = {
    top: triggerRect.top - tooltipRect.height - OFFSET - ARROW_SIZE > 0,
    bottom: triggerRect.bottom + tooltipRect.height + OFFSET + ARROW_SIZE < vh,
    left: triggerRect.left - tooltipRect.width - OFFSET - ARROW_SIZE > 0,
    right: triggerRect.right + tooltipRect.width + OFFSET + ARROW_SIZE < vw,
  };

  const opposites: Record<string, "top" | "right" | "bottom" | "left"> = {
    top: "bottom",
    bottom: "top",
    left: "right",
    right: "left",
  };
  const fallbackOrder: Array<"top" | "right" | "bottom" | "left"> = [
    preferred,
    opposites[preferred],
    ...(["top", "right", "bottom", "left"] as const).filter((d) => d !== preferred && d !== opposites[preferred]),
  ];

  const resolved = fallbackOrder.find((d) => fits[d]) ?? preferred;

  let top = 0;
  let left = 0;

  switch (resolved) {
    case "top":
      top = triggerRect.top - tooltipRect.height - OFFSET - ARROW_SIZE;
      left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
      break;
    case "bottom":
      top = triggerRect.bottom + OFFSET + ARROW_SIZE;
      left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
      break;
    case "left":
      top = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
      left = triggerRect.left - tooltipRect.width - OFFSET - ARROW_SIZE;
      break;
    case "right":
      top = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
      left = triggerRect.right + OFFSET + ARROW_SIZE;
      break;
  }

  // Clamp within viewport
  left = Math.max(4, Math.min(left, vw - tooltipRect.width - 4));
  top = Math.max(4, Math.min(top, vh - tooltipRect.height - 4));

  return { top, left, resolvedPosition: resolved };
}

function arrowStyle(pos: "top" | "right" | "bottom" | "left"): React.CSSProperties {
  const base: React.CSSProperties = {
    position: "absolute",
    width: 0,
    height: 0,
    borderStyle: "solid",
    borderColor: "transparent",
  };

  const color = "var(--color-slate-950)";
  const size = `${ARROW_SIZE}px`;

  switch (pos) {
    case "top":
      return {
        ...base,
        bottom: `-${ARROW_SIZE}px`,
        left: "50%",
        transform: "translateX(-50%)",
        borderWidth: `${size} ${size} 0 ${size}`,
        borderTopColor: color,
      };
    case "bottom":
      return {
        ...base,
        top: `-${ARROW_SIZE}px`,
        left: "50%",
        transform: "translateX(-50%)",
        borderWidth: `0 ${size} ${size} ${size}`,
        borderBottomColor: color,
      };
    case "left":
      return {
        ...base,
        right: `-${ARROW_SIZE}px`,
        top: "50%",
        transform: "translateY(-50%)",
        borderWidth: `${size} 0 ${size} ${size}`,
        borderLeftColor: color,
      };
    case "right":
      return {
        ...base,
        left: `-${ARROW_SIZE}px`,
        top: "50%",
        transform: "translateY(-50%)",
        borderWidth: `${size} ${size} ${size} 0`,
        borderRightColor: color,
      };
  }
}

export function Tooltip({ children, content, position = "top", delay = 200 }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const [coords, setCoords] = useState<Coords | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const id = useId();

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    const tooltip = tooltipRef.current;
    if (!trigger || !tooltip) return;

    const triggerRect = trigger.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    setCoords(computeCoords(triggerRect, tooltipRect, position));
  }, [position]);

  const show = () => {
    timeoutRef.current = setTimeout(() => setVisible(true), delay);
  };

  const hide = () => {
    clearTimeout(timeoutRef.current);
    setVisible(false);
    setCoords(null);
  };

  useEffect(() => () => clearTimeout(timeoutRef.current), []);

  useEffect(() => {
    if (!visible) return;

    const rafId = requestAnimationFrame(() => {
      updatePosition();
    });

    const handleScrollOrResize = () => updatePosition();

    window.addEventListener("scroll", handleScrollOrResize, { passive: true });
    window.addEventListener("resize", handleScrollOrResize, { passive: true });

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("scroll", handleScrollOrResize);
      window.removeEventListener("resize", handleScrollOrResize);
    };
  }, [visible, updatePosition]);

  return (
    <>
      <span
        ref={triggerRef}
        className="inline-flex"
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
      >
        <span aria-describedby={visible ? id : undefined}>{children}</span>
      </span>

      {visible &&
        createPortal(
          <span
            ref={tooltipRef}
            id={id}
            role="tooltip"
            className="pointer-events-none z-[9999] px-3 py-2 text-sm rounded-[var(--radius-sm)]
              bg-[var(--color-slate-950)] text-[var(--color-frost-50)] shadow-[var(--shadow-2)]"
            style={{
              position: "fixed",
              top: coords ? coords.top : -9999,
              left: coords ? coords.left : -9999,
              maxWidth: 300,
              whiteSpace: "normal",
              opacity: coords ? 1 : 0,
              transition: "opacity 150ms ease-out",
            }}
          >
            {content}
            {coords && <span style={arrowStyle(coords.resolvedPosition)} />}
          </span>,
          document.body,
        )}
    </>
  );
}
