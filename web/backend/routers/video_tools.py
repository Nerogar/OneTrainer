from web.backend.services.video_service import VideoService
from web.backend.utils.path_security import validate_path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/tools/video", tags=["tools"])


class ExtractClipsRequest(BaseModel):
    video_path: str = ""
    directory: str = ""
    batch_mode: bool = False
    output_dir: str
    time_start: str = "00:00:00"
    time_end: str = "99:99:99"
    output_subdirectories: bool = False
    split_at_cuts: bool = False
    max_length: float = 3.0
    fps: int = 24
    remove_borders: bool = False
    crop_variation: float = 0.2


class ExtractImagesRequest(BaseModel):
    video_path: str = ""
    directory: str = ""
    batch_mode: bool = False
    output_dir: str
    time_start: str = "00:00:00"
    time_end: str = "99:99:99"
    output_subdirectories: bool = False
    images_per_second: float = 0.5
    blur_removal: float = 0.2
    remove_borders: bool = False
    crop_variation: float = 0.2


class DownloadRequest(BaseModel):
    url: str = ""
    link_list_path: str = ""
    batch_mode: bool = False
    output_dir: str
    additional_args: str = "--quiet --no-warnings --progress"


class VideoToolResponse(BaseModel):
    ok: bool
    error: str | None = None
    message: str | None = None


class VideoToolStatusResponse(BaseModel):
    status: str  # "idle" | "running" | "completed" | "error"
    message: str | None = None
    error: str | None = None


@router.post("/extract-clips", response_model=VideoToolResponse)
def extract_clips(req: ExtractClipsRequest):
    validate_path(req.output_dir, must_exist=False, allow_file=False)
    if req.batch_mode:
        validate_path(req.directory, must_exist=True, allow_file=False)
    else:
        validate_path(req.video_path, must_exist=True, allow_file=True, allow_dir=False)
    service = VideoService.get_instance()
    result = service.extract_clips(
        video_path=req.video_path,
        directory=req.directory,
        batch_mode=req.batch_mode,
        output_dir=req.output_dir,
        time_start=req.time_start,
        time_end=req.time_end,
        output_subdirectories=req.output_subdirectories,
        split_at_cuts=req.split_at_cuts,
        max_length=req.max_length,
        fps=req.fps,
        remove_borders=req.remove_borders,
        crop_variation=req.crop_variation,
    )
    return VideoToolResponse(**result)


@router.post("/extract-images", response_model=VideoToolResponse)
def extract_images(req: ExtractImagesRequest):
    validate_path(req.output_dir, must_exist=False, allow_file=False)
    if req.batch_mode:
        validate_path(req.directory, must_exist=True, allow_file=False)
    else:
        validate_path(req.video_path, must_exist=True, allow_file=True, allow_dir=False)
    service = VideoService.get_instance()
    result = service.extract_images(
        video_path=req.video_path,
        directory=req.directory,
        batch_mode=req.batch_mode,
        output_dir=req.output_dir,
        time_start=req.time_start,
        time_end=req.time_end,
        output_subdirectories=req.output_subdirectories,
        images_per_second=req.images_per_second,
        blur_removal=req.blur_removal,
        remove_borders=req.remove_borders,
        crop_variation=req.crop_variation,
    )
    return VideoToolResponse(**result)


@router.post("/download", response_model=VideoToolResponse)
def download_videos(req: DownloadRequest):
    validate_path(req.output_dir, must_exist=False, allow_file=False)
    if req.batch_mode and req.link_list_path:
        validate_path(req.link_list_path, must_exist=True, allow_file=True, allow_dir=False)
    service = VideoService.get_instance()
    result = service.download_videos(
        url=req.url,
        link_list_path=req.link_list_path,
        batch_mode=req.batch_mode,
        output_dir=req.output_dir,
        additional_args=req.additional_args,
    )
    return VideoToolResponse(**result)


@router.get("/status", response_model=VideoToolStatusResponse)
def get_status():
    service = VideoService.get_instance()
    status = service.get_status()
    return VideoToolStatusResponse(**status)
