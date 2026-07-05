from web.backend.services.monitor_service import MonitorService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/system", tags=["system"])


class GpuMetrics(BaseModel):
    index: int
    name: str
    vram_used_mb: float
    vram_total_mb: float
    vram_percent: float
    temperature: float | None = None
    utilization: float | None = None


class SystemMetricsResponse(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpus: list[GpuMetrics]


class GpuInfo(BaseModel):
    index: int
    name: str
    vram_total_mb: float


class SystemInfoResponse(BaseModel):
    cpu_count: int
    cpu_count_physical: int | None = None
    ram_total_gb: float
    gpus: list[GpuInfo]


@router.get("/metrics", response_model=SystemMetricsResponse)
def get_metrics():
    monitor = MonitorService.get_instance()
    return SystemMetricsResponse(**monitor.get_metrics())


@router.get("/info", response_model=SystemInfoResponse)
def get_info():
    monitor = MonitorService.get_instance()
    return SystemInfoResponse(**monitor.get_system_info())


class CacheStatusResponse(BaseModel):
    cache_dir: str
    exists: bool
    size_mb: float
    file_count: int


class CacheActionResponse(BaseModel):
    ok: bool
    error: str | None = None


@router.get("/cache/status", response_model=CacheStatusResponse)
def get_cache_status():
    import os

    from web.backend.services.config_service import ConfigService

    config_service = ConfigService.get_instance()
    config = config_service.get_config_for_training()
    cache_dir = getattr(config, "cache_dir", "") or "workspace-cache/run"

    if not os.path.isdir(cache_dir):
        return CacheStatusResponse(cache_dir=cache_dir, exists=False, size_mb=0, file_count=0)

    total_size = 0
    file_count = 0
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    return CacheStatusResponse(
        cache_dir=cache_dir,
        exists=True,
        size_mb=round(total_size / (1024 * 1024), 2),
        file_count=file_count,
    )


@router.post("/cache/clear", response_model=CacheActionResponse)
def clear_cache():
    import os
    import shutil

    from web.backend.services.config_service import ConfigService

    config_service = ConfigService.get_instance()
    config = config_service.get_config_for_training()
    cache_dir = getattr(config, "cache_dir", "") or "workspace-cache/run"

    if not os.path.isdir(cache_dir):
        return CacheActionResponse(ok=True)

    try:
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        return CacheActionResponse(ok=True)
    except Exception as e:
        return CacheActionResponse(ok=False, error=str(e))


# ---- Profiling ----


class ProfilingDumpResponse(BaseModel):
    ok: bool
    path: str | None = None
    error: str | None = None


class ProfilingToggleResponse(BaseModel):
    ok: bool
    active: bool = False
    error: str | None = None


_profiling_active = False


@router.post("/profiling/dump-stacks", response_model=ProfilingDumpResponse)
def profiling_dump_stacks():
    """Dump current Python thread stacks to <workspace>/profiling/stacks_<ts>.txt."""
    import datetime
    import pathlib
    import sys
    import traceback

    from web.backend.services.config_service import ConfigService

    try:
        config_service = ConfigService.get_instance()
        config = config_service.get_config_for_training()
        workspace = getattr(config, "workspace_dir", "") or "workspace"
        out_dir = pathlib.Path(workspace) / "profiling"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"stacks_{ts}.txt"

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(f"Stack dump captured at {datetime.datetime.now().isoformat()}\n")
            for tid, frame in sys._current_frames().items():
                fh.write(f"\n--- thread {tid} ---\n")
                traceback.print_stack(frame, file=fh)

        return ProfilingDumpResponse(ok=True, path=str(out_path))
    except Exception as exc:
        return ProfilingDumpResponse(ok=False, error=str(exc))


@router.post("/profiling/toggle", response_model=ProfilingToggleResponse)
def profiling_toggle():
    """Toggle Scalene profiling. Requires OneTrainer to be launched with --profile."""
    global _profiling_active
    try:
        import scalene  # noqa: F401
    except ImportError:
        return ProfilingToggleResponse(
            ok=False,
            active=False,
            error="Scalene not installed - launch OneTrainer with --profile to enable profiling",
        )
    _profiling_active = not _profiling_active
    return ProfilingToggleResponse(ok=True, active=_profiling_active)
