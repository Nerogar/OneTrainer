# WebUI

OneTrainer supports WebUI as the primary interactive UI for browser-based operation.

## What this provides

- Browser-accessible UI for starting/stopping training
- Runtime training state endpoint (status + progress + logs)
- Reuse of the same training backend (`TrainConfig`, trainer factory, callbacks, commands)

## Legacy desktop UI status

- The legacy CustomTkinter desktop training UI in [`scripts/train_ui.py`](../scripts/train_ui.py) is deprecated for this workflow.
- For WebUI workflows, run [`scripts/train_webui.py`](../scripts/train_webui.py) directly.
- TensorBoard behavior remains unchanged.

## Start WebUI

### Windows

- Run directly: `python scripts\\train_webui.py --host 127.0.0.1 --port 7860`
- Optional wrapper: [`start-webui.bat`](../start-webui.bat)

Optional environment variables:

- `WEBUI_HOST` (default: `127.0.0.1`)
- `WEBUI_PORT` (default: `7860`)

### Linux/macOS

- Run directly: `python scripts/train_webui.py --host 127.0.0.1 --port 7860`
- Optional wrapper: [`start-webui.sh`](../start-webui.sh)

Optional environment variables:

- `WEBUI_HOST` (default: `127.0.0.1`)
- `WEBUI_PORT` (default: `7860`)

You can also pass CLI args directly:

```bash
./start-webui.sh --host 0.0.0.0 --port 7860
```

## Endpoints

Implemented in [`scripts/train_webui.py`](../scripts/train_webui.py).

- `GET /` - simple browser UI page
- `GET /api/health` - health check
- `GET /api/state` - current training runtime state
- `POST /api/train/start` - start training from config
- `POST /api/train/stop` - request graceful stop

### Start training request payload

```json
{
  "config_path": "training_presets/#.json",
  "secrets_path": "secrets.json"
}
```

- `config_path` is required and must be valid JSON
- `secrets_path` is optional (defaults to `secrets.json` fallback logic)

## Architecture notes

- Web mode is implemented primarily in new files:
  - [`scripts/train_webui.py`](../scripts/train_webui.py)
  - [`modules/webui/WebUIService.py`](../modules/webui/WebUIService.py)
  - [`start-webui.bat`](../start-webui.bat)
  - [`start-webui.sh`](../start-webui.sh)
- Training still runs through existing shared backend creation (`create.create_trainer`)

## Dependency additions

The WebUI requires:

- `fastapi`
- `uvicorn`

These are added to [`requirements-global.txt`](../requirements-global.txt).

## Current limitations

- Minimal initial web front-end intended as a safe baseline
- Not yet feature-parity with all legacy desktop tabs
- Single training job at a time per WebUI process
