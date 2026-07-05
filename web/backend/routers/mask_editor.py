import importlib.util
import logging
import os
from pathlib import Path
from typing import Annotated

from web.backend.services.tool_service import scan_image_folder
from web.backend.utils.path_security import validate_path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mask-editor"])

_yolo_cache: dict[str, object] = {}


class ImageListItem(BaseModel):
    path: str
    filename: str
    has_mask: bool


class YoloPredictRequest(BaseModel):
    image_path: str
    model_path: str


class YoloDetection(BaseModel):
    class_name: str
    confidence: float
    polygon: list[list[float]]
    bbox: list[float]


class YoloPredictResponse(BaseModel):
    ok: bool
    detections: list[YoloDetection] = []
    error: str | None = None


@router.get("/tools/mask-editor/images", response_model=list[ImageListItem])
def list_images(folder: str, include_subdirectories: bool = False):
    canonical = validate_path(folder, must_exist=True, allow_file=False)
    images = scan_image_folder(canonical, include_subdirectories)
    result = []
    for img_path in images:
        mask_path = img_path.parent / f"{img_path.stem}-masklabel.png"
        result.append(
            ImageListItem(
                path=str(img_path),
                filename=img_path.name,
                has_mask=mask_path.exists(),
            )
        )
    return result


@router.get("/tools/mask-editor/image")
def get_image(path: str):
    canonical = validate_path(path, must_exist=True, allow_dir=False)
    ext = os.path.splitext(canonical)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    def stream():
        with open(canonical, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(stream(), media_type=media_type)


@router.post("/tools/mask-editor/save-mask")
async def save_mask(
    image_path: Annotated[str, Form()],
    smooth: Annotated[int, Form()] = 0,
    expand: Annotated[int, Form()] = 0,
    threshold: Annotated[int, Form()] = 128,
    mask: Annotated[UploadFile, File()] = None,  # type: ignore[assignment]
):
    canonical = validate_path(image_path, must_exist=True, allow_dir=False)
    img_path = Path(canonical)
    mask_path = img_path.parent / f"{img_path.stem}-masklabel.png"

    mask_data = await mask.read()

    if smooth > 0 or expand > 0:
        import cv2
        import numpy as np

        arr = np.frombuffer(mask_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid mask image data")

        if smooth > 0:
            ksize = smooth * 2 + 1
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        if expand > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
            img = cv2.dilate(img, kernel)

        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        _, buf = cv2.imencode(".png", img)
        mask_data = buf.tobytes()

    mask_path.write_bytes(mask_data)
    return {"ok": True, "mask_path": str(mask_path)}


@router.post("/tools/mask-editor/yolo-predict", response_model=YoloPredictResponse)
def yolo_predict(req: YoloPredictRequest):
    if not importlib.util.find_spec("ultralytics"):
        return YoloPredictResponse(
            ok=False,
            error="Ultralytics is not installed. Install with: pip install ultralytics",
        )

    image_canonical = validate_path(req.image_path, must_exist=True, allow_dir=False)
    model_canonical = validate_path(req.model_path, must_exist=True, allow_dir=False)

    try:
        from ultralytics import YOLO

        if model_canonical not in _yolo_cache:
            _yolo_cache[model_canonical] = YOLO(model_canonical)
        model = _yolo_cache[model_canonical]

        results = model.predict(image_canonical, verbose=False)
        result = results[0] if results else None

        detections: list[YoloDetection] = []
        if result and result.boxes is not None:
            names = result.names or {}
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxyn[0].tolist()  # normalized [x1, y1, x2, y2]

                polygon: list[list[float]] = []
                if result.masks is not None and i < len(result.masks.xyn):
                    polygon = result.masks.xyn[i].tolist()

                detections.append(
                    YoloDetection(
                        class_name=names.get(cls_id, str(cls_id)),
                        confidence=round(conf, 3),
                        polygon=polygon,
                        bbox=bbox,
                    )
                )

        return YoloPredictResponse(ok=True, detections=detections)

    except Exception as exc:
        logger.warning("YOLO prediction failed", exc_info=True)
        return YoloPredictResponse(ok=False, error=str(exc))
