from web.backend.services.sampler_service import SamplerService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["tools"])


class SamplerActionResponse(BaseModel):
    ok: bool
    error: str | None = None


class SamplerSampleResponse(BaseModel):
    ok: bool
    error: str | None = None
    sample: dict | None = None


class SamplerStatusResponse(BaseModel):
    status: str  # "idle" | "loading" | "ready" | "sampling" | "error"
    error: str | None = None
    model_loaded: bool
    sample_progress: dict


class StandaloneSampleRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    height: int = 512
    width: int = 512
    seed: int = 42
    random_seed: bool = False
    diffusion_steps: int = 20
    cfg_scale: float = 7.0
    noise_scheduler: str = "DDIM"


@router.post("/tools/sampling/load-model", response_model=SamplerActionResponse)
def load_sampling_model():
    service = SamplerService.get_instance()
    result = service.load_model()
    return SamplerActionResponse(**result)


@router.post("/tools/sampling/sample", response_model=SamplerSampleResponse)
def standalone_sample(req: StandaloneSampleRequest):
    service = SamplerService.get_instance()
    result = service.sample(req.model_dump())
    return SamplerSampleResponse(**result)


@router.post("/tools/sampling/unload", response_model=SamplerActionResponse)
def unload_sampling_model():
    service = SamplerService.get_instance()
    result = service.unload_model()
    return SamplerActionResponse(**result)


@router.get("/tools/sampling/status", response_model=SamplerStatusResponse)
def sampling_status():
    service = SamplerService.get_instance()
    status = service.get_status()
    return SamplerStatusResponse(**status)
