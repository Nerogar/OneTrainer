import { type ChildProcess, execSync, spawn } from "child_process";
import { randomBytes, randomUUID } from "crypto";
import { app, BrowserWindow, dialog, type FileFilter, ipcMain, Menu, shell } from "electron";
import * as fs from "fs";
import * as http from "http";
import * as path from "path";

import { IPC_CHANNELS } from "../shared/ipc-channels";
import { closeSplash, createSplashWindow, showSplashError, updateSplash } from "./splash";

const isWindows = process.platform === "win32";
let BACKEND_PORT = 8000;
let BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
let HEALTH_URL = `${BACKEND_URL}/api/health`;

function setBackendPort(port: number): void {
  BACKEND_PORT = port;
  BACKEND_URL = `http://127.0.0.1:${port}`;
  HEALTH_URL = `${BACKEND_URL}/api/health`;
}

function findFreePort(start: number): Promise<number> {
  return new Promise((resolve) => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const net = require("net") as typeof import("net");
    const server = net.createServer();
    server.listen(start, "127.0.0.1", () => {
      const port = (server.address() as import("net").AddressInfo).port;
      server.close(() => resolve(port));
    });
    server.on("error", () => {
      resolve(findFreePort(start + 1));
    });
  });
}

const HEALTH_POLL_INTERVAL = 500;
const MAX_HEALTH_RETRIES = 120;
const DEFAULT_DEV_SERVER_PORT = 5173;

// vite.config.ts's writeResolvedPortPlugin writes the port Vite actually bound
// to here once it starts listening -- Vite auto-increments past 5173 when
// that port is taken, and this is how a separate process (us) finds out.
const VITE_PORT_FILE = path.resolve(__dirname, "..", "..", "..", ".vite-port");

async function resolveDevServerPort(timeoutMs = 10000): Promise<number> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const contents = fs.readFileSync(VITE_PORT_FILE, "utf-8").trim();
      const port = Number.parseInt(contents, 10);
      if (Number.isInteger(port) && port > 0) return port;
    } catch {
      // File not written yet -- keep polling until the dev server starts.
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  return DEFAULT_DEV_SERVER_PORT;
}

let DEV_SERVER_URL = `http://127.0.0.1:${DEFAULT_DEV_SERVER_PORT}`;

// When run_web.bat / run_web_dev.bat start the backend externally,
// they set this env var so Electron doesn't spawn a duplicate.
const externalBackend = process.env.OT_EXTERNAL_BACKEND === "1";
const devMode = process.env.OT_DEV === "1" || process.argv.includes("--dev");

// Disable Chromium GPU compositing — the UI is form-based and PyTorch needs the VRAM.
app.disableHardwareAcceleration();
Menu.setApplicationMenu(null);

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
const shutdownToken = randomBytes(32).toString("hex");

function getProjectRoot(): string {
  let dir = __dirname;
  for (let i = 0; i < 10; i++) {
    if (fs.existsSync(path.join(dir, "pyproject.toml"))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error(`Could not locate project root (no pyproject.toml found above ${__dirname})`);
}

function getAppIcon(): string {
  const root = getProjectRoot();
  return isWindows
    ? path.join(root, "resources", "icons", "icon.ico")
    : path.join(root, "resources", "icons", "icon.png");
}

function findPython(): string {
  const projectRoot = getProjectRoot();
  const venvPaths = isWindows
    ? [
        path.join(projectRoot, "venv", "Scripts", "python.exe"),
        path.join(projectRoot, ".venv", "Scripts", "python.exe"),
      ]
    : [path.join(projectRoot, "venv", "bin", "python"), path.join(projectRoot, ".venv", "bin", "python")];

  for (const p of venvPaths) {
    if (fs.existsSync(p)) {
      console.log(`[Electron] Found Python at: ${p}`);
      return p;
    }
  }

  const fallback = isWindows ? "python" : "python3";
  console.log(`[Electron] No venv found, falling back to: ${fallback}`);
  return fallback;
}

function killProcessTree(proc: ChildProcess | null): void {
  if (!proc || proc.killed) return;
  try {
    if (isWindows) {
      execSync(`taskkill /PID ${proc.pid} /T /F`, {
        windowsHide: true,
        stdio: "ignore",
      });
    } else {
      const pid = proc.pid;
      if (pid === undefined) return;
      // Try to kill the process group first (negative PID)
      try {
        process.kill(-pid, "SIGTERM");
      } catch {
        process.kill(pid, "SIGTERM");
      }
      setTimeout(() => {
        try {
          process.kill(pid, "SIGKILL");
        } catch {
          /* already dead */
        }
      }, 2000);
    }
  } catch {
    try {
      proc.kill();
    } catch {
      /* ignore */
    }
  }
}

function killStaleBackend(): void {
  try {
    if (isWindows) {
      const output = execSync(`netstat -ano | findstr ":${BACKEND_PORT} " | findstr "LISTENING"`, {
        windowsHide: true,
        encoding: "utf8",
        timeout: 5000,
      });
      const pids = new Set<string>();
      for (const line of output.trim().split("\n")) {
        const pid = line.trim().split(/\s+/).pop();
        if (pid && /^\d+$/.test(pid) && pid !== "0") pids.add(pid);
      }
      for (const pid of pids) {
        console.log(`[Electron] Killing stale process on port ${BACKEND_PORT} (PID ${pid})`);
        try {
          execSync(`taskkill /PID ${pid} /T /F`, {
            windowsHide: true,
            stdio: "ignore",
          });
        } catch {
          /* already dead */
        }
      }
    } else {
      execSync(`fuser -k ${BACKEND_PORT}/tcp 2>/dev/null || true`, {
        encoding: "utf8",
        timeout: 5000,
        stdio: "ignore",
      });
    }
  } catch {
    // No process on port -- expected for clean starts
  }
}

function startBackend(): ChildProcess | null {
  if (backendProcess && !backendProcess.killed) {
    killProcessTree(backendProcess);
  }

  killStaleBackend();

  const projectRoot = getProjectRoot();
  const python = findPython();

  // Only pass --reload for an explicit `--dev` launch (devMode). Plain
  // start-web-ui.bat/.sh runs are unpackaged too (app.isPackaged is false
  // whenever running via `electron .`), so gating on devMode instead of
  // isPackaged keeps normal launches from paying for file-watching.
  const reloadArgs = devMode ? ["--reload", "--reload-dir", path.join(projectRoot, "web", "backend")] : [];

  const proc = spawn(
    python,
    [
      "-m",
      "uvicorn",
      "web.backend.main:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(BACKEND_PORT),
      "--log-level",
      "info",
      ...reloadArgs,
    ],
    {
      cwd: projectRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        OT_SHUTDOWN_TOKEN: shutdownToken,
        OT_ELECTRON: "1",
      },
      ...(isWindows ? { windowsHide: true } : { detached: true }),
    },
  );

  attachBackendHandlers(proc);
  return proc;
}

const NOISY_LOG_RE =
  /GET \/api\/health|GET \/api\/presets|GET \/api\/config|GET \/api\/training\/status|WebSocket \/ws\//;

function attachBackendHandlers(proc: ChildProcess): void {
  proc.stdout?.on("data", (data: Buffer) => {
    const line = data.toString();
    if (!devMode && NOISY_LOG_RE.test(line)) return;
    process.stdout.write(`[Backend] ${line}`);
  });

  proc.stderr?.on("data", (data: Buffer) => {
    const line = data.toString();
    if (!devMode && NOISY_LOG_RE.test(line)) return;
    process.stderr.write(`[Backend] ${line}`);
  });

  proc.on("close", (code) => {
    console.log(`[Backend] Process exited with code ${code}`);
    backendProcess = null;
  });
}

function checkHealth(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, (res) => {
      resolve(res.statusCode !== undefined && res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(2000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(onProgress?: (attempt: number, maxAttempts: number) => void): Promise<boolean> {
  for (let i = 0; i < MAX_HEALTH_RETRIES; i++) {
    const healthy = await checkHealth();
    if (healthy) {
      console.log(`[Electron] Backend is ready (attempt ${i + 1})`);
      return true;
    }
    if (devMode) console.log(`[Electron] Waiting for backend... (${i + 1}/${MAX_HEALTH_RETRIES})`);
    if (onProgress) onProgress(i + 1, MAX_HEALTH_RETRIES);
    await new Promise((r) => setTimeout(r, HEALTH_POLL_INTERVAL));
  }
  console.error("[Electron] Backend failed to start within timeout");
  return false;
}

function flushWindow(win: BrowserWindow, timeoutMs = 3000): Promise<void> {
  if (win.isDestroyed() || win.webContents.isDestroyed()) return Promise.resolve();

  return new Promise<void>((resolve) => {
    const requestId = randomUUID();

    const timer = setTimeout(() => {
      ipcMain.removeListener(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
      console.warn(`[Electron] Flush timed out after ${timeoutMs}ms for window ${win.id}`);
      resolve();
    }, timeoutMs);

    const onComplete = (_event: Electron.IpcMainEvent, id: string) => {
      if (id !== requestId) return;
      clearTimeout(timer);
      ipcMain.removeListener(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
      resolve();
    };

    ipcMain.on(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
    try {
      win.webContents.send(IPC_CHANNELS.FLUSH_REQUEST, requestId);
    } catch (err) {
      console.warn(`[Electron] Failed to send flush request: ${String(err)}`);
      clearTimeout(timer);
      ipcMain.removeListener(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
      resolve();
    }
  });
}

function createWindow(): Promise<BrowserWindow> {
  return new Promise((resolve) => {
    const win = new BrowserWindow({
      width: 1400,
      height: 1000,
      minWidth: 1024,
      minHeight: 700,
      title: "OneTrainer",
      icon: getAppIcon(),
      show: false,
      webPreferences: {
        preload: path.join(__dirname, "preload.cjs"),
        contextIsolation: true,
        nodeIntegration: false,
        // Electron's default sandboxed preload loader can't require() relative
        // files — it'd break our shared/ipc-channels import. contextIsolation
        // plus nodeIntegration:false is still the real renderer security
        // boundary; disabling sandbox just lets the preload use full Node.
        sandbox: false,
      },
    });
    mainWindow = win;

    const rendererPath = path.join(__dirname, "..", "..", "renderer", "index.html");
    // Only probe the Vite dev server on an explicit `--dev` launch -- a plain
    // start-web-ui.bat/.sh run is unpackaged too, but should always load the
    // built renderer rather than attempt (and hot-reload from) a dev server.
    const isDev = devMode && !app.isPackaged;
    if (isDev) {
      new Promise<boolean>((probeResolve) => {
        const req = http.get(DEV_SERVER_URL, () => probeResolve(true));
        req.on("error", () => probeResolve(false));
        req.setTimeout(500, () => {
          req.destroy();
          probeResolve(false);
        });
      }).then((devServerUp) => {
        if (devServerUp) {
          win.loadURL(`${DEV_SERVER_URL}?backendPort=${BACKEND_PORT}${devMode ? "&dev=1" : ""}`);
        } else {
          win.loadFile(rendererPath, {
            query: { backendPort: String(BACKEND_PORT), ...(devMode ? { dev: "1" } : {}) },
          });
        }
      });
      if (process.env.OT_DEVTOOLS === "1") {
        win.webContents.openDevTools();
      }
    } else {
      win.loadFile(rendererPath, {
        query: { backendPort: String(BACKEND_PORT), ...(devMode ? { dev: "1" } : {}) },
      });
    }

    win.webContents.on("will-navigate", (event, url) => {
      const parsed = new URL(url);
      if (parsed.hostname === "localhost") return;
      event.preventDefault();
      shell.openExternal(url);
    });

    win.webContents.setWindowOpenHandler(({ url }) => {
      shell.openExternal(url);
      return { action: "deny" };
    });

    const loadTimeout = setTimeout(() => {
      console.warn("[Electron] Main window ready-to-show timeout, resolving anyway");
      resolve(win);
    }, 30000);

    win.once("ready-to-show", () => {
      clearTimeout(loadTimeout);
      resolve(win);
    });

    win.on("closed", () => {
      mainWindow = null;
    });

    let didFlush = false;
    win.on("close", (event) => {
      if (didFlush) return;
      event.preventDefault();
      void (async () => {
        try {
          await flushWindow(win);
        } finally {
          didFlush = true;
          if (!win.isDestroyed()) win.destroy();
        }
      })();
    });
  });
}

function registerIpcHandlers(): void {
  ipcMain.handle(IPC_CHANNELS.OPEN_FILE, async (_event, filters?: FileFilter[]) => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile"],
      filters: filters ?? [],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle(IPC_CHANNELS.OPEN_DIRECTORY, async () => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openDirectory"],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle(IPC_CHANNELS.SAVE_FILE, async (_event, defaultPath?: string, filters?: FileFilter[]) => {
    if (!mainWindow) return null;
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath,
      filters: filters ?? [],
    });
    return result.canceled ? null : result.filePath;
  });

  ipcMain.handle(IPC_CHANNELS.GET_APP_PATH, () => {
    return app.getAppPath();
  });

  ipcMain.handle(IPC_CHANNELS.RESTART_BACKEND, async () => {
    if (externalBackend) {
      console.log("[Electron] Backend is externally managed; waiting for it to come back...");
      return waitForBackend();
    }

    console.log("[Electron] Restarting backend...");
    killProcessTree(backendProcess);
    backendProcess = startBackend();
    if (!backendProcess) return false;
    return waitForBackend();
  });

  ipcMain.handle(IPC_CHANNELS.GET_PLATFORM_INFO, () => {
    return {
      platform: process.platform,
      isPackaged: app.isPackaged,
      version: app.getVersion(),
      projectRoot: getProjectRoot(),
    };
  });

  ipcMain.handle(IPC_CHANNELS.GET_BACKEND_PORT, () => {
    return BACKEND_PORT;
  });

  ipcMain.handle(IPC_CHANNELS.OPEN_MASK_EDITOR, async (_event, folder?: string) => {
    const rendererPath = path.join(__dirname, "..", "..", "renderer", "index.html");
    const isDev = devMode && !app.isPackaged;
    const folderParam = folder ? `&folder=${encodeURIComponent(folder)}` : "";

    const editorWindow = new BrowserWindow({
      width: 1600,
      height: 1000,
      minWidth: 1200,
      minHeight: 800,
      title: "Mask Editor - OneTrainer",
      icon: getAppIcon(),
      webPreferences: {
        preload: path.join(__dirname, "preload.cjs"),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: false,
      },
    });

    if (isDev) {
      const devServerUp = await new Promise<boolean>((probeResolve) => {
        const req = http.get(DEV_SERVER_URL, () => probeResolve(true));
        req.on("error", () => probeResolve(false));
        req.setTimeout(500, () => {
          req.destroy();
          probeResolve(false);
        });
      });
      if (devServerUp) {
        editorWindow.loadURL(`${DEV_SERVER_URL}?backendPort=${BACKEND_PORT}&route=mask-editor${folderParam}`);
      } else {
        editorWindow.loadFile(rendererPath, {
          query: {
            backendPort: String(BACKEND_PORT),
            route: "mask-editor",
            ...(folder ? { folder } : {}),
          },
        });
      }
    } else {
      editorWindow.loadFile(rendererPath, {
        query: {
          backendPort: String(BACKEND_PORT),
          route: "mask-editor",
          ...(folder ? { folder } : {}),
        },
      });
    }

    editorWindow.webContents.on("will-navigate", (event, url) => {
      const parsed = new URL(url);
      if (parsed.hostname === "localhost") return;
      event.preventDefault();
      shell.openExternal(url);
    });

    editorWindow.webContents.setWindowOpenHandler(({ url }) => {
      shell.openExternal(url);
      return { action: "deny" };
    });

    return true;
  });
}

function isProcessAlive(proc: ChildProcess | null): boolean {
  if (!proc || proc.killed || proc.pid === undefined) return false;
  try {
    process.kill(proc.pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function gracefulShutdownBackend(timeoutMs = 8000): Promise<void> {
  if (!backendProcess || backendProcess.killed) return;

  console.log("[Electron] Requesting graceful backend shutdown...");
  try {
    await new Promise<void>((resolve) => {
      const req = http.request(
        {
          hostname: "localhost",
          port: BACKEND_PORT,
          path: "/api/shutdown",
          method: "POST",
          headers: {
            "Content-Length": 0,
            "x-shutdown-token": shutdownToken,
          },
        },
        (res) => {
          res.resume();
          resolve();
        },
      );
      req.on("error", () => resolve());
      req.setTimeout(3000, () => {
        req.destroy();
        resolve();
      });
      req.end("");
    });
  } catch {
    console.log("[Electron] Backend unreachable, skipping graceful shutdown");
    killProcessTree(backendProcess);
    return;
  }

  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    if (!isProcessAlive(backendProcess)) {
      console.log("[Electron] Backend exited gracefully");
      return;
    }
    await new Promise((r) => setTimeout(r, 250));
  }

  console.log("[Electron] Backend did not exit within grace period, force-killing");
  killProcessTree(backendProcess);
}

let cleanupPromise: Promise<void> | null = null;

function cleanupAndQuit(): Promise<void> {
  if (cleanupPromise) return cleanupPromise;

  cleanupPromise = (async () => {
    if (!externalBackend) {
      await gracefulShutdownBackend(8000);
    }
  })();

  return cleanupPromise;
}

async function main(): Promise<void> {
  await app.whenReady();
  registerIpcHandlers();
  createSplashWindow();

  if (!externalBackend) {
    updateSplash(2, 8, "Finding available port...");
    const port = await findFreePort(8000);
    setBackendPort(port);
    if (port !== 8000) {
      console.log(`[Electron] Port 8000 busy, using port ${port}`);
    }
    updateSplash(2, 8, `Starting backend server on port ${port}...`);
    console.log("[Electron] Starting backend...");
    backendProcess = startBackend();
    if (!backendProcess) {
      console.error("[Electron] Failed to spawn backend process");
      showSplashError("Failed to start backend process");
      setTimeout(() => app.quit(), 3000);
      return;
    }
    updateSplash(2, 12, "Backend process started");
  } else {
    updateSplash(2, 12, "Using external backend...");
    console.log("[Electron] OT_EXTERNAL_BACKEND=1 -- skipping backend spawn");
  }

  updateSplash(3, 15, "Waiting for backend...");
  console.log("[Electron] Waiting for backend to start...");

  const healthy = await waitForBackend((attempt, maxAttempts) => {
    const progress = Math.min(15 + (attempt / maxAttempts) * 60, 75);
    updateSplash(3, progress, `Loading... ${Math.round(progress)}%`);
  });

  if (!healthy) {
    console.error("[Electron] Backend failed to start within timeout");
    showSplashError("Backend failed to start (60s timeout)");
    if (!externalBackend) {
      killProcessTree(backendProcess);
    }
    setTimeout(() => app.quit(), 5000);
    return;
  }

  console.log("[Electron] Backend is healthy");
  updateSplash(4, 80, "Loading interface...");

  if (devMode && !app.isPackaged) {
    const port = await resolveDevServerPort();
    DEV_SERVER_URL = `http://127.0.0.1:${port}`;
    if (port !== DEFAULT_DEV_SERVER_PORT) {
      console.log(`[Electron] Vite dev server on port ${port} (default ${DEFAULT_DEV_SERVER_PORT} was busy)`);
    }
  }

  const window = await createWindow();
  updateSplash(5, 95, "Almost ready...");

  await closeSplash(window);

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
}

app.on("window-all-closed", () => {
  cleanupAndQuit().finally(() => {
    if (process.platform !== "darwin") {
      app.quit();
    }
  });
});

app.on("will-quit", (event) => {
  if (cleanupPromise) {
    event.preventDefault();
    cleanupPromise.finally(() => {
      cleanupPromise = null;
      app.quit();
    });
  }
});

process.on("uncaughtException", (err) => {
  console.error("[Electron] Uncaught exception:", err);
  if (!externalBackend) {
    killProcessTree(backendProcess);
  }
  app.quit();
});

main().catch(console.error);
