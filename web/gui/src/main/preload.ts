import { contextBridge, ipcRenderer } from "electron";

import type { ElectronAPI } from "../shared/electron-api";
import { IPC_CHANNELS } from "../shared/ipc-channels";

export type { ElectronAPI, PlatformInfo } from "../shared/electron-api";

contextBridge.exposeInMainWorld("electronAPI", {
  isElectron: true,
  platform: process.platform,
  openFile: (filters?) => ipcRenderer.invoke(IPC_CHANNELS.OPEN_FILE, filters),
  openDirectory: () => ipcRenderer.invoke(IPC_CHANNELS.OPEN_DIRECTORY),
  saveFile: (defaultPath?, filters?) => ipcRenderer.invoke(IPC_CHANNELS.SAVE_FILE, defaultPath, filters),
  getAppPath: () => ipcRenderer.invoke(IPC_CHANNELS.GET_APP_PATH),
  restartBackend: () => ipcRenderer.invoke(IPC_CHANNELS.RESTART_BACKEND),
  getPlatformInfo: () => ipcRenderer.invoke(IPC_CHANNELS.GET_PLATFORM_INFO),
  getBackendPort: () => ipcRenderer.invoke(IPC_CHANNELS.GET_BACKEND_PORT),
  openMaskEditor: (folder?) => ipcRenderer.invoke(IPC_CHANNELS.OPEN_MASK_EDITOR, folder),
  onFlushRequest: (handler) => {
    const listener = (_event: Electron.IpcRendererEvent, requestId: string) => {
      void Promise.resolve(handler(requestId));
    };
    ipcRenderer.on(IPC_CHANNELS.FLUSH_REQUEST, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.FLUSH_REQUEST, listener);
    };
  },
  signalFlushComplete: (requestId) => {
    ipcRenderer.send(IPC_CHANNELS.FLUSH_COMPLETE, requestId);
  },
} satisfies ElectronAPI);
