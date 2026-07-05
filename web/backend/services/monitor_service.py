import contextlib
import logging

from web.backend.services._singleton import SingletonMixin
from web.backend.utils.gpu_info import get_gpu_metrics, get_gpu_static_info

import psutil

logger = logging.getLogger(__name__)


class MonitorService(SingletonMixin):
    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)

    def get_metrics(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        return {
            "cpu_percent": cpu_percent,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_percent": mem.percent,
            "gpus": get_gpu_metrics(),
        }

    def get_system_info(self) -> dict:
        mem = psutil.virtual_memory()
        info: dict = {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "gpus": [],
        }

        with contextlib.suppress(Exception):
            info["gpus"] = get_gpu_static_info()

        return info
