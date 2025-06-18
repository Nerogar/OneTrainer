import logging
import os
import platform
import subprocess

import customtkinter as ctk
from util.import_util import script_imports

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

script_imports()

from modules.ui.TrainUI import TrainUI


def get_linux_scaling_factor():
    """
    Tries to detect the UI scaling factor on Linux by checking common desktop environments.
    This is a best-effort attempt as there is no standard method.
    """
    try:
        # Check for GNOME/Cinnamon/MATE settings (text scaling is often the one users set for fractional)
        cmd = ["gsettings", "get", "org.gnome.desktop.interface", "text-scaling-factor"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout:
            return float(result.stdout.strip())

        # Check for XFCE settings
        cmd = ["xfconf-query", "-c", "xsettings", "-p", "/Gdk/WindowScalingFactor"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout:
            return int(result.stdout.strip())

        # Fallback to DPI check (good for integer scaling, less so for fractional)
        temp_root = ctk.CTk()
        dpi = temp_root.winfo_fpixels('1i')
        temp_root.destroy()
        scaling = dpi / 96.0
        if scaling >= 1.25:
            return scaling

    except (FileNotFoundError, ValueError, Exception):
        pass

    return 1.0


def main():
    # This logic is for Linux, where we may need to manually set the scaling factor.
    if platform.system() == "Linux":
        # Prioritize manual override via environment variable, as it's the most reliable
        scaling_env = os.environ.get("ONETRAINER_UI_SCALING")
        scaling_factor = 1.0

        if scaling_env:
            try:
                scaling_factor = float(scaling_env)
            except (ValueError, TypeError):
                logger.warning("Invalid value for ONETRAINER_UI_SCALING, using default.")
                scaling_factor = 1.0
        else:
            # If no manual override, attempt to auto-detect
            scaling_factor = get_linux_scaling_factor()

        if scaling_factor != 1.0:
            ctk.set_widget_scaling(scaling_factor)
            ctk.set_window_scaling(scaling_factor)

    ui = TrainUI()
    ui.mainloop()


if __name__ == '__main__':
    main()
