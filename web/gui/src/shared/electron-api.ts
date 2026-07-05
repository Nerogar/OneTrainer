export interface PlatformInfo {
  platform: string;
  isPackaged: boolean;
  version: string;
  projectRoot: string;
}

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface ElectronAPI {
  isElectron: true;
  platform: string;
  openFile: (filters?: FileFilter[]) => Promise<string | null>;
  openDirectory: () => Promise<string | null>;
  saveFile: (defaultPath?: string, filters?: FileFilter[]) => Promise<string | null>;
  getAppPath: () => Promise<string>;
  restartBackend: () => Promise<boolean>;
  getPlatformInfo: () => Promise<PlatformInfo>;
  getBackendPort: () => Promise<number>;
  openMaskEditor: (folder?: string) => Promise<boolean>;
  onFlushRequest: (handler: (requestId: string) => Promise<void> | void) => () => void;
  signalFlushComplete: (requestId: string) => void;
}
