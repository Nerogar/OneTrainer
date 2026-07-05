from web.backend.services.tensorboard_service import TensorboardService

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/tensorboard", tags=["tensorboard"])


@router.get("/runs")
def list_runs() -> list[str]:
    service = TensorboardService.get_instance()
    return service.list_runs()


@router.get("/scalars")
def list_tags(run: str = Query(..., description="Run name")) -> list[str]:
    service = TensorboardService.get_instance()
    tags = service.list_tags(run)
    if not tags and not _run_exists(run):
        raise HTTPException(status_code=404, detail=f"Run not found: {run}")
    return tags


@router.get("/scalars/{tag:path}")
def get_scalars(
    tag: str,
    run: str = Query(..., description="Run name"),
    after_step: int = Query(0, description="Only return data after this step (for incremental updates)"),
) -> list[dict]:
    service = TensorboardService.get_instance()

    if not _run_exists(run):
        raise HTTPException(status_code=404, detail=f"Run not found: {run}")

    return service.get_scalars(run, tag, after_step=after_step)


def _run_exists(run_name: str) -> bool:
    service = TensorboardService.get_instance()
    runs = service.list_runs()
    return run_name in runs


@router.post("/launch")
def launch():
    """Start tensorboard subprocess if not running. Returns URL when responsive."""
    service = TensorboardService.get_instance()
    return service.ensure_running()


@router.post("/stop")
def stop():
    service = TensorboardService.get_instance()
    return service.stop()
