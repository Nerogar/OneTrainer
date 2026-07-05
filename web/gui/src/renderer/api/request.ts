const isFileProtocol = typeof window !== "undefined" && window.location.protocol === "file:";

function resolveBackendPort(): number {
  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    const portStr = params.get("backendPort");
    if (portStr) {
      const port = parseInt(portStr, 10);
      if (!isNaN(port)) return port;
    }
  }
  return 8000;
}

const BACKEND_PORT = resolveBackendPort();

export const API_BASE = isFileProtocol ? `http://127.0.0.1:${BACKEND_PORT}/api` : "/api";
export const WS_BASE = isFileProtocol ? `ws://127.0.0.1:${BACKEND_PORT}` : `ws://${window.location.host}`;

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class ClientError extends ApiError {
  constructor(status: number, body: string) {
    super(`Client error ${status}: ${body}`, status, body);
    this.name = "ClientError";
  }
}

export class ServerError extends ApiError {
  constructor(status: number, body: string) {
    super(`Server error ${status}: ${body}`, status, body);
    this.name = "ServerError";
  }
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options?.body ? { "Content-Type": "application/json" } : {}),
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const text = await res.text();
    if (res.status >= 400 && res.status < 500) {
      throw new ClientError(res.status, text);
    }
    if (res.status >= 500) {
      throw new ServerError(res.status, text);
    }
    throw new ApiError(`API error ${res.status}: ${text}`, res.status, text);
  }

  return res.json() as Promise<T>;
}
