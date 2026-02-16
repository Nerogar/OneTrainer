import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

from modules.web.training_session import SessionStore


app = FastAPI(title="OneTrainer WebUI", version="0.1.0")
sessions = SessionStore()


def _index_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>OneTrainer WebUI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #111827; color: #e5e7eb; }
    .container { max-width: 1180px; margin: 0 auto; padding: 16px; }
    h1 { margin: 0 0 8px 0; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 12px; flex: 1; min-width: 320px; }
    button { background: #2563eb; border: 0; color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; margin-right: 6px; margin-bottom: 6px; }
    button.secondary { background: #374151; }
    button.warn { background: #b91c1c; }
    textarea, input, select { width: 100%; background: #0b1220; color: #e5e7eb; border: 1px solid #374151; border-radius: 8px; padding: 8px; box-sizing: border-box; }
    textarea { min-height: 360px; font-family: Consolas, monospace; }
    .mono { font-family: Consolas, monospace; white-space: pre-wrap; }
    #samples img { max-width: 240px; border-radius: 8px; border: 1px solid #374151; margin: 6px; }
    .status { font-size: 14px; }
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>OneTrainer WebUI</h1>
    <p class=\"status\">Lightweight browser control panel for training sessions.</p>

    <div class=\"row\">
      <div class=\"card\">
        <h3>Session</h3>
        <div class=\"row\">
          <div style=\"flex:1\">
            <label>Session ID</label>
            <input id=\"sessionId\" readonly />
          </div>
        </div>
        <div style=\"margin-top:10px\">
          <button onclick=\"createSession()\">Create Session</button>
          <button class=\"secondary\" onclick=\"refreshSession()\">Refresh</button>
          <button class=\"warn\" onclick=\"deleteSession()\">Delete</button>
        </div>
        <div style=\"margin-top:10px\">
          <button onclick=\"startTraining()\">Start</button>
          <button class=\"warn\" onclick=\"stopTraining()\">Stop</button>
          <button class=\"secondary\" onclick=\"sampleNow()\">Sample</button>
          <button class=\"secondary\" onclick=\"backupNow()\">Backup</button>
          <button class=\"secondary\" onclick=\"saveNow()\">Save</button>
        </div>
        <div style=\"margin-top:10px\" class=\"mono\" id=\"statusBox\">No session</div>
      </div>

      <div class=\"card\">
        <h3>Config (JSON)</h3>
        <textarea id=\"configText\" placeholder=\"Create session to load config\"></textarea>
        <div style=\"margin-top:10px\">
          <button onclick=\"saveConfig()\">Save Config</button>
          <button class=\"secondary\" onclick=\"loadConfig()\">Reload Config</button>
        </div>
      </div>
    </div>

    <div class=\"row\" style=\"margin-top:12px\">
      <div class=\"card\">
        <h3>Events</h3>
        <div id=\"events\" class=\"mono\" style=\"max-height:220px;overflow:auto\"></div>
      </div>
      <div class=\"card\">
        <h3>Latest Samples</h3>
        <div id=\"samples\"></div>
      </div>
    </div>
  </div>

<script>
let ws = null;

function appendEvent(msg) {
  const events = document.getElementById('events');
  const line = document.createElement('div');
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  events.prepend(line);
}

function setSessionId(id) {
  document.getElementById('sessionId').value = id || '';
}

function getSessionId() {
  return document.getElementById('sessionId').value.trim();
}

function renderStatus(data) {
  document.getElementById('statusBox').textContent = JSON.stringify(data, null, 2);
}

function renderSamples(images) {
  const el = document.getElementById('samples');
  el.innerHTML = '';
  (images || []).slice(-8).reverse().forEach(src => {
    const img = document.createElement('img');
    img.src = src;
    el.appendChild(img);
  });
}

async function createSession() {
  const raw = document.getElementById('configText').value.trim();
  let payload = {};
  if (raw) {
    try { payload = JSON.parse(raw); } catch (e) { alert('Invalid JSON'); return; }
  }
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({config: payload})
  });
  const data = await res.json();
  setSessionId(data.id);
  renderStatus(data);
  appendEvent(`Session created: ${data.id}`);
  await loadConfig();
  connectWs();
}

async function refreshSession() {
  const id = getSessionId();
  if (!id) return;
  const res = await fetch(`/api/sessions/${id}`);
  if (!res.ok) { appendEvent('Failed to refresh session'); return; }
  const data = await res.json();
  renderStatus(data);
  renderSamples(data.sample_images || []);
}

async function deleteSession() {
  const id = getSessionId();
  if (!id) return;
  const res = await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
  if (!res.ok) { alert('Delete failed'); return; }
  appendEvent(`Session deleted: ${id}`);
  setSessionId('');
  renderStatus({deleted: true});
  document.getElementById('samples').innerHTML = '';
  if (ws) { ws.close(); ws = null; }
}

async function loadConfig() {
  const id = getSessionId();
  if (!id) return;
  const res = await fetch(`/api/sessions/${id}/config`);
  if (!res.ok) { appendEvent('Load config failed'); return; }
  const txt = await res.text();
  document.getElementById('configText').value = txt;
}

async function saveConfig() {
  const id = getSessionId();
  if (!id) return;
  const raw = document.getElementById('configText').value;
  let config;
  try { config = JSON.parse(raw); } catch (e) { alert('Invalid JSON'); return; }
  const res = await fetch(`/api/sessions/${id}/config`, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({config})
  });
  if (!res.ok) { alert('Save config failed'); return; }
  const data = await res.json();
  renderStatus(data);
  appendEvent('Config updated');
}

async function action(path) {
  const id = getSessionId();
  if (!id) return;
  const res = await fetch(`/api/sessions/${id}/${path}`, { method: 'POST' });
  if (!res.ok) {
    const msg = await res.text();
    appendEvent(`${path} failed: ${msg}`);
    return;
  }
  const data = await res.json();
  renderStatus(data);
  appendEvent(`${path} ok`);
}

function startTraining(){ action('start'); }
function stopTraining(){ action('stop'); }
function sampleNow(){ action('sample'); }
function backupNow(){ action('backup'); }
function saveNow(){ action('save'); }

function connectWs() {
  const id = getSessionId();
  if (!id) return;
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/sessions/${id}`);
  ws.onmessage = (evt) => {
    const event = JSON.parse(evt.data);
    appendEvent(`${event.type}: ${JSON.stringify(event.payload)}`);
    if (event.type === 'sample') {
      const img = event.payload.image;
      if (img) {
        const target = document.getElementById('samples');
        const el = document.createElement('img');
        el.src = img;
        target.prepend(el);
      }
    }
  };
  ws.onopen = () => appendEvent('WebSocket connected');
  ws.onclose = () => appendEvent('WebSocket disconnected');
}
</script>
</body>
</html>
"""


def _safe_get_session(session_id: str):
    try:
        return sessions.get_session(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _index_html()


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/api/sessions")
def list_sessions() -> list[dict[str, Any]]:
    return [s.to_public_dict() for s in sessions.list_sessions()]


@app.post("/api/sessions")
def create_session(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = body or {}
    config_payload = payload.get("config")
    secrets_payload = payload.get("secrets")
    session = sessions.create_session(config_payload=config_payload, secrets_payload=secrets_payload)
    return session.to_public_dict()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = _safe_get_session(session_id)
    return session.to_public_dict()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    try:
        sessions.delete_session(session_id)
        return {"deleted": True}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.get("/api/sessions/{session_id}/config", response_class=PlainTextResponse)
def get_session_config(session_id: str) -> str:
    _safe_get_session(session_id)
    return sessions.export_config_json(session_id)


@app.put("/api/sessions/{session_id}/config")
def put_session_config(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    session = _safe_get_session(session_id)
    if session.running:
        raise HTTPException(status_code=409, detail="Cannot edit config while training is running")

    config_payload = body.get("config")
    if not isinstance(config_payload, dict):
        raise HTTPException(status_code=400, detail="Body must include object field 'config'")

    session.train_config.from_dict(config_payload)
    return session.to_public_dict()


@app.post("/api/sessions/{session_id}/start")
def start_session(session_id: str) -> dict[str, Any]:
    session = _safe_get_session(session_id)
    try:
        session.start()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return session.to_public_dict()


@app.post("/api/sessions/{session_id}/stop")
def stop_session(session_id: str) -> dict[str, Any]:
    session = _safe_get_session(session_id)
    session.stop()
    return session.to_public_dict()


@app.post("/api/sessions/{session_id}/sample")
def sample_session(session_id: str) -> dict[str, Any]:
    session = _safe_get_session(session_id)
    session.sample_default()
    return session.to_public_dict()


@app.post("/api/sessions/{session_id}/backup")
def backup_session(session_id: str) -> dict[str, Any]:
    session = _safe_get_session(session_id)
    session.backup()
    return session.to_public_dict()


@app.post("/api/sessions/{session_id}/save")
def save_session(session_id: str) -> dict[str, Any]:
    session = _safe_get_session(session_id)
    session.save()
    return session.to_public_dict()


@app.websocket("/ws/sessions/{session_id}")
async def websocket_events(websocket: WebSocket, session_id: str):
    session = _safe_get_session(session_id)
    await websocket.accept()

    try:
        while True:
            events = session.consume_events()
            for event in events:
                await websocket.send_text(json.dumps(event))
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return


def run(host: str = "127.0.0.1", port: int = 7865):
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()

