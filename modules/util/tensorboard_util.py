import os
import re
import subprocess
import sys
import threading

from modules.util.tqdm_util import tqdm

# Match only known startup notices; unknown output must remain visible.
_QUIET_LINES = {
    "",
    "TensorFlow installation not found - running with reduced feature set.",
    "NOTE: Using experimental fast data loading logic. To disable, pass",
    '"--load_fast=false" and report issues on GitHub. More details:',
    "https://github.com/tensorflow/tensorboard/issues/4784",
    "Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all",
}
_PKG_RESOURCES_WARNING = re.compile(
    r".*[\\/]tensorboard[\\/]default\.py:\d+: UserWarning: pkg_resources is deprecated as an API\..*"
)


class TensorboardProcess(subprocess.Popen):
    """Drain TensorBoard output without hiding failures or blocking the UI."""

    def __init__(self, args):
        self._stopping = False
        self._debug = bool(os.environ.get("OT_DEBUG_WARNINGS"))
        super().__init__(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        self._reader = threading.Thread(target=self._read_output, name="tensorboard-output", daemon=True)
        self._reader.start()

    def _read_output(self):
        assert self.stdout is not None
        skip_import = False
        with self.stdout:
            for line in self.stdout:
                message = line.strip()
                if skip_import and message == "import pkg_resources":
                    skip_import = False
                    continue
                skip_import = not self._debug and bool(_PKG_RESOURCES_WARNING.fullmatch(message))
                if skip_import:
                    continue
                if self._debug or message not in _QUIET_LINES:
                    tqdm.write(line, end="", file=sys.stderr)
        returncode = self.wait()
        if returncode and not self._stopping:
            tqdm.write(f"TensorBoard exited with code {returncode}", file=sys.stderr)

    def terminate(self):
        if self.poll() is None:
            self._stopping = True
        super().terminate()

    def kill(self):
        if self.poll() is None:
            self._stopping = True
        super().kill()
