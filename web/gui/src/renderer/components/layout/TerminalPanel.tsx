import type { FitAddon } from "@xterm/addon-fit";
import type { Terminal } from "@xterm/xterm";
import { ChevronDown, Eraser, Terminal as TerminalIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { useTerminalWebSocket } from "@/hooks/useTerminalWebSocket";
import { useUiStore } from "@/store/uiStore";

import { AZURE_500, COBALT_600, ON_SURFACE_DARK, SLATE_950, SURFACE_SUNKEN_DARK } from "../../../shared/brandColors";

const MIN_HEIGHT = 120;
const MAX_HEIGHT = 600;
const DEFAULT_HEIGHT = 240;
const STORAGE_KEY = "terminalPanelHeight";

interface TerminalPanelProps {
  isOpen: boolean;
  backendConnected: boolean;
}

export default function TerminalPanel({ isOpen, backendConnected }: TerminalPanelProps) {
  const setTerminalOpen = useUiStore((s) => s.setTerminalOpen);
  const containerRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);

  const [panelHeight, setPanelHeight] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, parseInt(saved, 10))) : DEFAULT_HEIGHT;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, String(panelHeight));
  }, [panelHeight]);

  useEffect(() => {
    if (!isOpen || !containerRef.current) return;

    let term: Terminal | null = null;
    let disposed = false;

    async function init() {
      const { Terminal } = await import("@xterm/xterm");
      const { FitAddon } = await import("@xterm/addon-fit");
      const { WebLinksAddon } = await import("@xterm/addon-web-links");
      const { SearchAddon } = await import("@xterm/addon-search");
      await import("@xterm/xterm/css/xterm.css");

      if (disposed || !containerRef.current) return;

      term = new Terminal({
        theme: {
          background: SURFACE_SUNKEN_DARK,
          foreground: "#D4DCE8",
          cursor: COBALT_600,
          selectionBackground: `${AZURE_500}4D`, // 30% opacity
          black: SLATE_950,
          red: "#FF6B6B",
          green: "#4ADE80",
          yellow: "#FACC15",
          blue: "#60A5FA",
          magenta: COBALT_600,
          cyan: "#22D3EE",
          white: ON_SURFACE_DARK,
          brightBlack: "#4B5563",
          brightRed: "#FCA5A5",
          brightGreen: "#86EFAC",
          brightYellow: "#FDE68A",
          brightBlue: "#93C5FD",
          brightMagenta: "#60A5FA",
          brightCyan: "#67E8F9",
          brightWhite: "#FFFFFF",
        },
        fontFamily: '"JetBrains Mono", "SF Mono", "Cascadia Code", Consolas, monospace',
        fontSize: 12,
        lineHeight: 1.4,
        scrollback: 5000,
        convertEol: true,
        cursorBlink: false,
        disableStdin: true,
        allowTransparency: true,
      });

      const fitAddon = new FitAddon();
      term.loadAddon(fitAddon);
      term.loadAddon(new WebLinksAddon());
      term.loadAddon(new SearchAddon());
      term.open(containerRef.current);

      requestAnimationFrame(() => {
        if (!disposed) {
          fitAddon.fit();
        }
      });

      termRef.current = term;
      fitAddonRef.current = fitAddon;
    }

    init();

    return () => {
      disposed = true;
      if (term) {
        term.dispose();
      }
      termRef.current = null;
      fitAddonRef.current = null;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !containerRef.current) return;

    const ro = new ResizeObserver(() => {
      fitAddonRef.current?.fit();
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [isOpen]);

  const onData = useCallback((text: string) => {
    termRef.current?.write(text);
  }, []);

  useTerminalWebSocket(onData, isOpen && backendConnected);

  const panelHeightRef = useRef(panelHeight);
  panelHeightRef.current = panelHeight;

  const dragStartY = useRef(0);
  const dragStartH = useRef(0);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragStartY.current = e.clientY;
    dragStartH.current = panelHeightRef.current;

    const onMouseMove = (e: MouseEvent) => {
      const delta = dragStartY.current - e.clientY;
      const newHeight = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, dragStartH.current + delta));
      setPanelHeight(newHeight);
    };

    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };

    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, []);

  const clearTerminal = useCallback(() => {
    termRef.current?.clear();
  }, []);

  return (
    <div
      className="terminal-panel"
      style={{ height: panelHeight }}
      role="log"
      aria-label="Backend log output"
      aria-live="off"
    >
      <div className="terminal-panel-resize-handle" onMouseDown={handleResizeStart} aria-hidden="true" />
      <div className="terminal-panel-header">
        <span className="terminal-panel-title">
          <TerminalIcon className="w-3.5 h-3.5" />
          Backend Log
        </span>
        <div className="terminal-panel-actions">
          <button onClick={clearTerminal} className="theme-toggle" aria-label="Clear terminal" title="Clear">
            <Eraser className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setTerminalOpen(false)}
            className="theme-toggle"
            aria-label="Close terminal panel"
            title="Close"
          >
            <ChevronDown className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      <div ref={containerRef} className="terminal-panel-xterm" />
    </div>
  );
}
