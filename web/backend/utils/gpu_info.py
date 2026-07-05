from __future__ import annotations

import logging
import threading
from contextlib import suppress
from typing import TypedDict

logger = logging.getLogger(__name__)

_nvml_initialized: bool = False
_nvml_available: bool = False
_nvml_init_lock = threading.Lock()


class GpuStaticInfo(TypedDict):
    index: int
    name: str
    vram_total_mb: float


class GpuMetrics(TypedDict):
    index: int
    name: str
    vram_used_mb: float
    vram_total_mb: float
    vram_percent: float
    temperature: float | None
    utilization: float | None


def _ensure_nvml() -> bool:
    global _nvml_initialized, _nvml_available  # noqa: PLW0603

    if _nvml_initialized:
        return _nvml_available

    with _nvml_init_lock:
        if _nvml_initialized:
            return _nvml_available

        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_available = True
            logger.info("pynvml initialised successfully")
        except Exception:  # noqa: BLE001
            _nvml_available = False
            logger.debug("pynvml not available, will try torch.cuda fallback")
        finally:
            _nvml_initialized = True

    return _nvml_available


def get_gpu_static_info() -> list[GpuStaticInfo]:
    if _ensure_nvml():
        try:
            return _static_info_nvml()
        except Exception:  # noqa: BLE001
            pass

    try:
        return _static_info_torch()
    except Exception:  # noqa: BLE001
        return []


def _static_info_nvml() -> list[GpuStaticInfo]:
    import pynvml

    device_count = pynvml.nvmlDeviceGetCount()
    gpus: list[GpuStaticInfo] = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append(
            GpuStaticInfo(
                index=i,
                name=name,
                vram_total_mb=round(mem_info.total / (1024**2), 1),
            )
        )
    return gpus


def _static_info_torch() -> list[GpuStaticInfo]:
    import torch

    if not torch.cuda.is_available():
        return []

    gpus: list[GpuStaticInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            GpuStaticInfo(
                index=i,
                name=props.name,
                vram_total_mb=round(props.total_memory / (1024**2), 1),
            )
        )
    return gpus


def get_gpu_metrics() -> list[GpuMetrics]:
    if _ensure_nvml():
        try:
            return _metrics_nvml()
        except Exception:  # noqa: BLE001
            logger.debug("pynvml metrics failed, trying torch.cuda fallback")

    try:
        return _metrics_torch()
    except Exception:  # noqa: BLE001
        pass

    return []


def _metrics_nvml() -> list[GpuMetrics]:
    import pynvml

    device_count = pynvml.nvmlDeviceGetCount()
    gpus: list[GpuMetrics] = []

    for i in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_mb = round(mem_info.used / (1024**2), 1)
            vram_total_mb = round(mem_info.total / (1024**2), 1)
            vram_percent = round((mem_info.used / mem_info.total) * 100, 1) if mem_info.total > 0 else 0.0

            temperature: float | None = None
            with suppress(Exception):
                temperature = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

            utilization: float | None = None
            with suppress(Exception):
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = float(util_rates.gpu)

            gpus.append(
                GpuMetrics(
                    index=i,
                    name=name,
                    vram_used_mb=vram_used_mb,
                    vram_total_mb=vram_total_mb,
                    vram_percent=vram_percent,
                    temperature=temperature,
                    utilization=utilization,
                )
            )
        except Exception:  # noqa: BLE001, PERF203
            logger.debug("Failed to read GPU %d via pynvml", i)

    return gpus


def _metrics_torch() -> list[GpuMetrics]:
    import torch

    if not torch.cuda.is_available():
        return []

    device_count = torch.cuda.device_count()
    gpus: list[GpuMetrics] = []

    for i in range(device_count):
        try:
            name = torch.cuda.get_device_name(i)
            mem_allocated = torch.cuda.memory_allocated(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory
            vram_used_mb = round(mem_allocated / (1024**2), 1)
            vram_total_mb = round(mem_total / (1024**2), 1)
            vram_percent = round((mem_allocated / mem_total) * 100, 1) if mem_total > 0 else 0.0

            gpus.append(
                GpuMetrics(
                    index=i,
                    name=name,
                    vram_used_mb=vram_used_mb,
                    vram_total_mb=vram_total_mb,
                    vram_percent=vram_percent,
                    temperature=None,
                    utilization=None,
                )
            )
        except Exception:  # noqa: BLE001, PERF203
            logger.debug("Failed to read GPU %d via torch.cuda", i)

    return gpus
