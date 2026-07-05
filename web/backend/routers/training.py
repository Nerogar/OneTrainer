from web.backend.services.trainer_service import TrainerService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["training"])


class TrainingActionResponse(BaseModel):
    ok: bool
    error: str | None = None


class TrainingStatusResponse(BaseModel):
    status: str  # "idle" | "running" | "stopping" | "error"
    error: str | None = None
    start_time: float | None = None


class StartTrainingRequest(BaseModel):
    reattach: bool = False


class CustomSampleRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    height: int = 512
    width: int = 512
    seed: int = 42
    random_seed: bool = False
    diffusion_steps: int = 20
    cfg_scale: float = 7.0


@router.post("/training/start", response_model=TrainingActionResponse)
def start_training(req: StartTrainingRequest | None = None):
    reattach = req.reattach if req is not None else False
    service = TrainerService.get_instance()
    result = service.start_training(reattach=reattach)
    return TrainingActionResponse(**result)


@router.post("/training/stop", response_model=TrainingActionResponse)
def stop_training():
    service = TrainerService.get_instance()
    result = service.stop_training()
    return TrainingActionResponse(**result)


@router.post("/training/sample", response_model=TrainingActionResponse)
def sample_now():
    service = TrainerService.get_instance()
    result = service.sample_now()
    return TrainingActionResponse(**result)


@router.post("/training/sample/custom", response_model=TrainingActionResponse)
def sample_custom(req: CustomSampleRequest):
    from modules.util.config.SampleConfig import SampleConfig

    sample_config = SampleConfig.default_values()
    sample_config.from_dict(req.model_dump())

    service = TrainerService.get_instance()
    result = service.sample_custom(sample_config)
    return TrainingActionResponse(**result)


@router.post("/training/backup", response_model=TrainingActionResponse)
def backup_now():
    service = TrainerService.get_instance()
    result = service.backup_now()
    return TrainingActionResponse(**result)


@router.post("/training/save", response_model=TrainingActionResponse)
def save_now():
    service = TrainerService.get_instance()
    result = service.save_now()
    return TrainingActionResponse(**result)


@router.get("/training/status", response_model=TrainingStatusResponse)
def get_status():
    service = TrainerService.get_instance()
    status = service.get_status()
    return TrainingStatusResponse(**status)
