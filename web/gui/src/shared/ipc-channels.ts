export const IPC_CHANNELS = {
  OPEN_FILE: "dialog:openFile",
  OPEN_DIRECTORY: "dialog:openDirectory",
  SAVE_FILE: "dialog:saveFile",
  GET_APP_PATH: "app:getPath",
  RESTART_BACKEND: "backend:restart",
  GET_PLATFORM_INFO: "app:getPlatformInfo",
  GET_BACKEND_PORT: "backend:getPort",
  OPEN_MASK_EDITOR: "tools:openMaskEditor",
  FLUSH_REQUEST: "lifecycle:flushRequest",
  FLUSH_COMPLETE: "lifecycle:flushComplete",
} as const;

export type IpcChannel = (typeof IPC_CHANNELS)[keyof typeof IPC_CHANNELS];
