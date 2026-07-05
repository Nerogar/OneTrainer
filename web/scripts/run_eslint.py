from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    gui_dir = Path(__file__).resolve().parents[1] / "gui"
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx is None:
        print("error: npx not found on PATH", file=sys.stderr)
        return 1

    result = subprocess.run(
        [npx, "eslint", "--fix", "--max-warnings", "0", "src/"],
        cwd=gui_dir,
        shell=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
