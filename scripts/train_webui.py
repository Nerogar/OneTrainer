from util.import_util import script_imports

script_imports()

import argparse
import json
import os
from pathlib import Path
from typing import Any

from modules.webui.WebUIService import WebUIService

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import uvicorn


class StartTrainingRequest(BaseModel):
    config_path: str
    secrets_path: str | None = None


class StopTrainingResponse(BaseModel):
    ok: bool
    message: str


def _normalize_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _validate_json_file(path: str) -> str:
    normalized = _normalize_path(path)
    if not os.path.isfile(normalized):
        raise HTTPException(status_code=400, detail=f"File not found: {normalized}")

    try:
        with open(normalized, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file at {normalized}: {e}") from e

    return normalized


def create_app() -> FastAPI:
    app = FastAPI(title="OneTrainer WebUI", version="0.1.0")
    service = WebUIService()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return service.get_state()

    @app.post("/api/train/start")
    def start_training(request: StartTrainingRequest) -> dict[str, Any]:
        config_path = _validate_json_file(request.config_path)
        secrets_path = None

        if request.secrets_path:
            secrets_path = _validate_json_file(str(request.secrets_path))

        ok, message = service.start_training(config_path=config_path, secrets_path=secrets_path)
        if not ok:
            raise HTTPException(status_code=409, detail=message)

        return {"ok": ok, "message": message}

    @app.post("/api/train/stop", response_model=StopTrainingResponse)
    def stop_training() -> dict[str, Any]:
        ok, message = service.stop_training()
        if not ok:
            raise HTTPException(status_code=409, detail=message)
        return {"ok": ok, "message": message}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>OneTrainer WebUI</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1000px; margin: 20px auto; padding: 0 12px; }
    .row { display: flex; gap: 8px; margin: 8px 0; }
    input { flex: 1; padding: 8px; }
    button { padding: 8px 12px; cursor: pointer; }
    pre { background: #111; color: #ddd; padding: 12px; border-radius: 6px; max-height: 340px; overflow: auto; }
    .status { padding: 8px; background: #f2f2f2; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>OneTrainer WebUI (Experimental)</h1>
  <div class=\"status\" id=\"status\">Loading state...</div>

  <div class=\"row\">
    <input id=\"configPath\" placeholder=\"Path to config JSON (required)\" />
  </div>
  <div class=\"row\">
    <input id=\"secretsPath\" placeholder=\"Path to secrets JSON (optional; defaults to secrets.json)\" />
  </div>

  <div class=\"row\">
    <button onclick=\"startTraining()\">Start Training</button>
    <button onclick=\"stopTraining()\">Stop Training</button>
    <button onclick=\"refreshState()\">Refresh</button>
  </div>

  <h3>Progress / Runtime State</h3>
  <pre id=\"stateBox\"></pre>

  <h3>Logs</h3>
  <pre id=\"logsBox\"></pre>

<script>
async function refreshState() {
  const r = await fetch('/api/state');
  const data = await r.json();
  document.getElementById('status').innerText = `running=${data.running} | status=${data.status} | error=${data.error ?? 'none'}`;

  const progress = {
    running: data.running,
    status: data.status,
    error: data.error,
    started_at: data.started_at,
    ended_at: data.ended_at,
    last_config_path: data.last_config_path,
    progress: data.progress,
  };

  document.getElementById('stateBox').innerText = JSON.stringify(progress, null, 2);
  document.getElementById('logsBox').innerText = (data.logs || []).join('\n');
}

async function startTraining() {
  const configPath = document.getElementById('configPath').value.trim();
  const secretsPath = document.getElementById('secretsPath').value.trim();
  if (!configPath) {
    alert('config_path is required');
    return;
  }

  const payload = { config_path: configPath, secrets_path: secretsPath || null };
  const r = await fetch('/api/train/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!r.ok) {
    const err = await r.json();
    alert(`Failed to start training: ${err.detail}`);
  }
  await refreshState();
}

async function stopTraining() {
  const r = await fetch('/api/train/stop', { method: 'POST' });
  if (!r.ok) {
    const err = await r.json();
    alert(`Failed to stop training: ${err.detail}`);
  }
  await refreshState();
}

setInterval(refreshState, 2000);
refreshState();
</script>
</body>
</html>
"""

    return app


def main():
    parser = argparse.ArgumentParser(description="Start OneTrainer WebUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind WebUI server")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind WebUI server")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

