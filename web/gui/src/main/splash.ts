import { app, BrowserWindow, screen } from "electron";
import * as fs from "fs";
import * as path from "path";

import { SURFACE_DARK } from "../shared/brandColors";

const isWindows = process.platform === "win32";

function getSplashIcon(): string {
  const root = app.isPackaged
    ? path.resolve(process.resourcesPath, "..")
    : path.resolve(__dirname, "..", "..", "..", "..", "..");
  return isWindows
    ? path.join(root, "resources", "icons", "icon.ico")
    : path.join(root, "resources", "icons", "icon.png");
}

let splashWindow: BrowserWindow | null = null;

export function createSplashWindow(): BrowserWindow | null {
  const splashPath = path.join(__dirname, "splash.html");
  if (!fs.existsSync(splashPath)) {
    console.warn("[Electron] splash.html not found, skipping splash screen");
    return null;
  }

  const primaryDisplay = screen.getPrimaryDisplay();
  const splashWidth = Math.max(720, Math.round(primaryDisplay.workAreaSize.width * 0.4));
  const splashHeight = Math.round(splashWidth * (9 / 16));

  splashWindow = new BrowserWindow({
    width: splashWidth,
    height: splashHeight,
    frame: false,
    resizable: false,
    backgroundColor: SURFACE_DARK,
    show: false,
    center: true,
    skipTaskbar: false,
    title: "OneTrainer",
    icon: getSplashIcon(),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  splashWindow.loadFile(splashPath);

  splashWindow.once("ready-to-show", () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.show();
    }
  });

  splashWindow.on("closed", () => {
    splashWindow = null;
  });

  return splashWindow;
}

export function updateSplash(step: number, percent: number, message: string): void {
  if (!splashWindow || splashWindow.isDestroyed()) return;
  splashWindow.webContents
    .executeJavaScript(`updateProgress(${step}, ${percent}, ${JSON.stringify(message)})`)
    .catch(() => {
      /* splash may have been destroyed */
    });
}

export function showSplashError(message: string): void {
  if (!splashWindow || splashWindow.isDestroyed()) return;
  splashWindow.webContents.executeJavaScript(`showError(${JSON.stringify(message)})`).catch(() => {
    /* splash may have been destroyed */
  });
}

export async function closeSplash(mainWindow: BrowserWindow, delayMs = 400): Promise<void> {
  if (!splashWindow || splashWindow.isDestroyed()) {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.show();
    }
    return;
  }

  updateSplash(5, 100, "Ready!");

  await new Promise((r) => setTimeout(r, delayMs));

  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.show();
  }

  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.close();
  }
  splashWindow = null;
}
