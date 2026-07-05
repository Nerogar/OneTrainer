import logging
import threading
from uuid import uuid4

from web.backend.utils.path_security import validate_path

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])

_convert_lock = threading.Lock()


class QuantizationParams(BaseModel):
    layer_filter: str = ""
    layer_filter_preset: str = "full"
    layer_filter_regex: bool = False
    svd_dtype: str = "NONE"  # DataType enum value
    svd_rank: int = 16
    cache_dir: str | None = None


class ConvertModelRequest(BaseModel):
    model_type: str  # ModelType enum value, e.g. "STABLE_DIFFUSION_15"
    training_method: str  # "FINE_TUNE", "LORA", "EMBEDDING"
    input_name: str  # file / directory path or HuggingFace repo id
    output_dtype: str  # "FLOAT_32", "FLOAT_16", "BFLOAT_16"
    output_model_format: str  # "SAFETENSORS", "DIFFUSERS"
    output_model_destination: str  # output file / directory path
    quantization: QuantizationParams | None = None


class ConvertModelResponse(BaseModel):
    ok: bool
    error: str | None = None


def _looks_like_local_path(name: str) -> bool:
    return name.startswith(("/", "\\", "./", ".\\")) or (len(name) > 1 and name[1] == ":")


@router.post("/tools/convert", response_model=ConvertModelResponse)
def convert_model(req: ConvertModelRequest) -> ConvertModelResponse:
    validate_path(req.output_model_destination, must_exist=False)
    if _looks_like_local_path(req.input_name):
        validate_path(req.input_name, must_exist=True)

    if not _convert_lock.acquire(blocking=False):
        return ConvertModelResponse(ok=False, error="A conversion is already in progress")
    try:
        from modules.util import create
        from modules.util.config.TrainConfig import QuantizationConfig
        from modules.util.enum.DataType import DataType
        from modules.util.enum.ModelFormat import ModelFormat
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.TrainingMethod import TrainingMethod
        from modules.util.ModelNames import EmbeddingName, ModelNames
        from modules.util.ModelWeightDtypes import ModelWeightDtypes
        from modules.util.torch_util import torch_gc

        model_type = ModelType(req.model_type)
        training_method = TrainingMethod(req.training_method)
        output_dtype = DataType(req.output_dtype)
        output_model_format = ModelFormat(req.output_model_format)

        weight_dtypes = ModelWeightDtypes.from_single_dtype(output_dtype)
        quantization = QuantizationConfig.default_values()
        if req.quantization is not None:
            q = req.quantization
            quantization.layer_filter = q.layer_filter
            quantization.layer_filter_preset = q.layer_filter_preset
            quantization.layer_filter_regex = q.layer_filter_regex
            quantization.svd_dtype = DataType(q.svd_dtype)
            quantization.svd_rank = q.svd_rank
            quantization.cache_dir = q.cache_dir

        model_loader = create.create_model_loader(
            model_type=model_type,
            training_method=training_method,
        )
        model_saver = create.create_model_saver(
            model_type=model_type,
            training_method=training_method,
        )

        logger.info("Loading model %s", req.input_name)
        if training_method == TrainingMethod.FINE_TUNE:
            model = model_loader.load(
                model_type=model_type,
                model_names=ModelNames(base_model=req.input_name),
                weight_dtypes=weight_dtypes,
                quantization=quantization,
            )
        elif training_method in (TrainingMethod.LORA, TrainingMethod.EMBEDDING):
            model = model_loader.load(
                model_type=model_type,
                model_names=ModelNames(
                    lora=req.input_name,
                    embedding=EmbeddingName(str(uuid4()), req.input_name),
                ),
                weight_dtypes=weight_dtypes,
                quantization=quantization,
            )
        else:
            return ConvertModelResponse(ok=False, error=f"Unsupported training method: {req.training_method}")

        logger.info("Saving model %s", req.output_model_destination)
        model_saver.save(
            model=model,
            model_type=model_type,
            output_model_format=output_model_format,
            output_model_destination=req.output_model_destination,
            dtype=output_dtype.torch_dtype(),
        )
        logger.info("Model converted")
        torch_gc()

        return ConvertModelResponse(ok=True)

    except Exception as exc:
        logger.exception("Model conversion failed")
        try:
            from modules.util.torch_util import torch_gc

            torch_gc()
        except Exception:
            pass
        return ConvertModelResponse(ok=False, error=str(exc))
    finally:
        _convert_lock.release()
