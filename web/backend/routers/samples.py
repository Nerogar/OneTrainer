from web.backend.services.concept_service import ConceptService
from web.backend.services.config_service import ConfigService

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("")
def get_samples() -> list[dict]:
    service = ConfigService.get_instance()
    sample_path = service.config.sample_definition_file_name

    if not sample_path:
        raise HTTPException(status_code=422, detail="No sample_definition_file_name configured")

    concept_service = ConceptService()
    try:
        return concept_service.load_samples(sample_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Sample definition file not found: {sample_path}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("")
def save_samples(samples: list[dict]) -> dict:
    service = ConfigService.get_instance()
    sample_path = service.config.sample_definition_file_name

    if not sample_path:
        raise HTTPException(status_code=422, detail="No sample_definition_file_name configured")

    concept_service = ConceptService()
    try:
        concept_service.save_samples(sample_path, samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"saved": len(samples), "path": sample_path}
