from util.import_util import script_imports

script_imports(allow_zluda=False)

import argparse
import subprocess
from pathlib import Path

from modules.api.rest.RestApi import RestApi
from modules.api.rest.TrainingService import TrainingService

import uvicorn
from fastapi import FastAPI

# Not a flag. This server is loopback-only by design: it can start training and
# read config files, and an exposed HTTP port has no transport security under
# it. Anything that needs remote access must put its own authenticated server in
# front.
HOST = "127.0.0.1"

DEFAULT_PORT = 7800


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneTrainer REST API server")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        dest="port",
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    return parser.parse_args(args)


def resolve_version(root_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        return "unknown"
    return "unknown"


def create_app(root_dir: Path) -> FastAPI:
    app = FastAPI(title="OneTrainer REST API", redoc_url=None)
    RestApi(TrainingService(), version=resolve_version(root_dir)).install(app)
    return app


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent

    print(f"OneTrainer REST API  ->  http://{HOST}:{args.port}")
    print("  Only this machine can reach it.")

    uvicorn.run(create_app(root_dir), host=HOST, port=args.port)


if __name__ == "__main__":
    main()
