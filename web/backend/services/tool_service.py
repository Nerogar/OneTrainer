import base64
import concurrent.futures
import importlib.util
import io
import logging
import random
import re
import threading
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from web.backend.services._singleton import SingletonMixin

import requests
from PIL import ExifTags, Image

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 933_120_000

ToolStatus = Literal["idle", "running", "completed", "error"]

VALID_CAPTION_MODES = {"replace", "fill", "add"}

VALID_MASK_MODES = {"replace", "fill", "add", "subtract", "blend"}

CAPTION_MODEL_MAP: dict[str, str] = {
    "Blip": "BlipModel",
    "Blip2": "Blip2Model",
    "WD14 VIT v2": "WDModel",
}

MASK_MODEL_MAP: dict[str, str] = {
    "ClipSeg": "ClipSegModel",
    "Rembg": "RembgModel",
    "Rembg-Human": "RembgHumanModel",
    "Hex Color": "MaskByColor",
    "YOLO": "YOLOMaskAdapter",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

VALID_API_BACKENDS = {"openai", "gemini"}

VALID_CAPTION_API_MODES = {"replace", "fill", "add"}

_REFUSAL_PATTERNS = [
    r"^i('m| am) sorry",
    r"^i cannot",
    r"^i can't",
    r"^i'm unable to",
    r"^unfortunately",
    r"^as an ai",
    r"cannot (assist|help|provide|process|describe|generate)",
    r"not (able|going) to (describe|caption|tag|process|generate)",
    r"violates? (my|the|our) (guidelines|policies|terms)",
    r"content (policy|guidelines|violation)",
]


def _is_refusal(text: str) -> bool:
    if not text:
        return True
    lower = text.lower().strip()
    return any(re.search(p, lower) for p in _REFUSAL_PATTERNS)


def _clean_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return " ".join(text.split()).strip()


def _normalize_openai_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    return url


def _encode_image_base64(image_path: Path, max_dim: int = 2048) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
        if max_dim > 0 and max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def scan_image_folder(folder: str, include_subdirectories: bool) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    pattern = "**/*" if include_subdirectories else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and "-masklabel" not in p.stem
    )


def _extract_image_metadata(image_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        img = Image.open(image_path)
        exif_data = img.getexif()
        if exif_data:
            tag_map = {v: k for k, v in ExifTags.TAGS.items()}
            for name, tag_id in [
                ("camera_make", tag_map.get("Make")),
                ("camera_model", tag_map.get("Model")),
                ("lens", tag_map.get("LensModel")),
                ("iso", tag_map.get("ISOSpeedRatings")),
            ]:
                if tag_id and tag_id in exif_data:
                    result[name] = str(exif_data[tag_id])
            f_number_tag = tag_map.get("FNumber")
            if f_number_tag and f_number_tag in exif_data:
                val = exif_data[f_number_tag]
                if hasattr(val, "numerator"):
                    result["aperture"] = f"f/{val.numerator / val.denominator:.1f}"
                else:
                    result["aperture"] = f"f/{val}"
            exposure_tag = tag_map.get("ExposureTime")
            if exposure_tag and exposure_tag in exif_data:
                val = exif_data[exposure_tag]
                if hasattr(val, "numerator") and val.numerator and val.denominator:
                    if val.numerator < val.denominator:
                        result["exposure"] = f"1/{val.denominator // val.numerator}s"
                    else:
                        result["exposure"] = f"{val.numerator / val.denominator}s"
                else:
                    result["exposure"] = str(val)
    except Exception:
        pass

    try:
        raw = image_path.read_bytes()
        start_marker = b"<x:xmpmeta"
        end_marker = b"</x:xmpmeta>"
        start_idx = raw.find(start_marker)
        end_idx = raw.find(end_marker)
        if start_idx >= 0 and end_idx > start_idx:
            xmp_bytes = raw[start_idx : end_idx + len(end_marker)]
            xmp_str = xmp_bytes.decode("utf-8", errors="replace")
            root_el = ET.fromstring(xmp_str)
            ns = {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "dc": "http://purl.org/dc/elements/1.1/",
            }
            for desc in root_el.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"):
                for title_el in desc.iter("{http://purl.org/dc/elements/1.1/}title"):
                    li = title_el.find(".//rdf:li", ns)
                    if li is not None and li.text:
                        result["xmp_title"] = li.text
                for desc_el in desc.iter("{http://purl.org/dc/elements/1.1/}description"):
                    li = desc_el.find(".//rdf:li", ns)
                    if li is not None and li.text:
                        result["xmp_description"] = li.text
                for subj_el in desc.iter("{http://purl.org/dc/elements/1.1/}subject"):
                    keywords = [li.text for li in subj_el.findall(".//rdf:li", ns) if li.text]
                    if keywords:
                        result["xmp_keywords"] = ", ".join(keywords)
    except Exception:
        pass

    return result


class _RateLimiter:
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm if rpm > 0 else 0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self, cancel_check: Callable[[], bool] | None = None) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            deadline = self._last + self._interval
            if now < deadline:
                delay = deadline - now
                self._last = deadline
            else:
                delay = 0
                self._last = now
        while delay > 0:
            if cancel_check is not None and cancel_check():
                return
            chunk = min(delay, 1.0)
            time.sleep(chunk)
            delay -= chunk


class ToolService(SingletonMixin):
    def __init__(self) -> None:
        self._status: ToolStatus = "idle"
        self._progress: int = 0
        self._max_progress: int = 0
        self._error_message: str | None = None
        self._task_id: str | None = None
        self._thread: threading.Thread | None = None
        self._cancel_flag: bool = False
        self._lock = threading.Lock()

        self._captioning_model: Any = None
        self._masking_model: Any = None

    def _set_status(self, status: ToolStatus, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._error_message = error

    def _update_progress(self, current: int, total: int) -> None:
        with self._lock:
            self._progress = current
            self._max_progress = total

    def get_status(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "progress": self._progress,
                "max_progress": self._max_progress,
                "error": self._error_message,
                "task_id": self._task_id,
            }

    def _start_background_task(self, target, args: tuple, thread_name: str) -> dict:
        task_id = str(uuid.uuid4())
        with self._lock:
            if self._status == "running":
                return {"ok": False, "error": "A tool operation is already running"}
            self._status = "running"
            self._progress = 0
            self._max_progress = 0
            self._error_message = None
            self._task_id = task_id
            self._cancel_flag = False

        thread = threading.Thread(
            target=target,
            args=(*args, task_id),
            daemon=True,
            name=thread_name,
        )
        self._thread = thread
        thread.start()
        return {"ok": True, "task_id": task_id}

    def generate_captions(self, request: Any) -> dict:
        return self._start_background_task(
            self._caption_thread_fn,
            (request,),
            "OneTrainerWeb-caption-tool",
        )

    def _caption_thread_fn(self, request: Any, task_id: str) -> None:
        try:
            model = self._load_captioning_model(request.model)
            if model is None:
                self._set_status("error", f"Unknown captioning model: {request.model}")
                return

            mode = request.mode if request.mode in VALID_CAPTION_MODES else "fill"

            model.caption_folder(
                sample_dir=request.folder,
                initial_caption=request.initial_caption,
                caption_prefix=request.caption_prefix,
                caption_postfix=request.caption_postfix,
                mode=mode,
                progress_callback=self._progress_callback,
                error_callback=self._error_callback,
                include_subdirectories=request.include_subdirectories,
            )

            with self._lock:
                if self._status == "running":
                    self._status = "completed"

        except InterruptedError:
            logger.info("Caption generation cancelled by user")
            self._set_status("idle")
        except Exception:
            traceback.print_exc()
            self._set_status("error", "Caption generation failed -- check the Terminal panel for details")
        finally:
            self._thread = None
            self._release_models()

    def generate_masks(self, request: Any) -> dict:
        return self._start_background_task(
            self._mask_thread_fn,
            (request,),
            "OneTrainerWeb-mask-tool",
        )

    def _mask_thread_fn(self, request: Any, task_id: str) -> None:
        try:
            model_path = getattr(request, "model_path", None)
            model = self._load_masking_model(request.model, model_path=model_path)
            if model is None:
                self._set_status("error", f"Unknown masking model: {request.model}")
                return

            mode = request.mode if request.mode in VALID_MASK_MODES else "fill"
            prompts = [request.prompt] if request.prompt else []

            model.mask_folder(
                sample_dir=request.folder,
                prompts=prompts,
                mode=mode,
                threshold=request.threshold,
                smooth_pixels=request.smooth,
                expand_pixels=request.expand,
                alpha=request.alpha,
                progress_callback=self._progress_callback,
                error_callback=self._error_callback,
                include_subdirectories=request.include_subdirectories,
            )

            with self._lock:
                if self._status == "running":
                    self._status = "completed"

        except InterruptedError:
            logger.info("Mask generation cancelled by user")
            self._set_status("idle")
        except Exception:
            traceback.print_exc()
            self._set_status("error", "Mask generation failed -- check the Terminal panel for details")
        finally:
            self._thread = None
            self._release_models()

    def generate_captions_api(self, request: Any) -> dict:
        return self._start_background_task(
            self._caption_api_thread_fn,
            (request,),
            "OneTrainerWeb-caption-api-tool",
        )

    def _caption_api_thread_fn(self, request: Any, task_id: str) -> None:
        try:
            backend = request.backend
            if backend not in VALID_API_BACKENDS:
                self._set_status("error", f"Unknown API backend: {backend}")
                return

            images = scan_image_folder(request.folder, request.include_subdirectories)
            if not images:
                self._set_status("error", "No images found in the specified folder")
                return

            mode = request.mode if request.mode in VALID_CAPTION_API_MODES else "fill"
            prompts = [request.prompt] if request.prompt else ["Describe this image."]
            if request.additional_prompts:
                prompts.extend(p for p in request.additional_prompts if p.strip())

            rpm = getattr(request, "requests_per_minute", 0)
            rate_limiter = _RateLimiter(rpm)
            batch_size = max(1, getattr(request, "batch_size", 1))
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_maxsize=max(batch_size, 10))
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            processed = 0
            total = len(images)
            self._update_progress(0, total)

            def process_image(img_path: Path) -> None:
                nonlocal processed
                if self._cancel_flag:
                    return
                rate_limiter.wait(lambda: self._cancel_flag)
                if self._cancel_flag:
                    return

                txt_path = img_path.with_suffix(".txt")
                if mode == "fill" and txt_path.exists() and txt_path.read_text(encoding="utf-8").strip():
                    processed += 1
                    self._update_progress(processed, total)
                    return

                selected_prompt = random.choice(prompts)
                full_prompt = self._build_caption_prompt(selected_prompt, img_path, request)

                try:
                    if backend == "openai":
                        caption = self._call_openai_api(img_path, full_prompt, request, session)
                    else:
                        caption = self._call_gemini_api(img_path, full_prompt, request, session)

                    if _is_refusal(caption):
                        logger.warning("Refusal detected for %s: %s", img_path.name, caption[:80])
                        processed += 1
                        self._update_progress(processed, total)
                        return

                    final = (
                        (getattr(request, "caption_prefix", "") or "")
                        + caption
                        + (getattr(request, "caption_postfix", "") or "")
                    )

                    if mode == "replace" or not txt_path.exists():
                        txt_path.write_text(final, encoding="utf-8")
                    elif mode == "add":
                        existing = txt_path.read_text(encoding="utf-8")
                        txt_path.write_text(existing.rstrip("\n") + "\n" + final, encoding="utf-8")
                except Exception:
                    logger.warning("Failed to caption %s", img_path.name, exc_info=True)

                processed += 1
                self._update_progress(processed, total)

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {executor.submit(process_image, p): p for p in images}
                for future in concurrent.futures.as_completed(futures):
                    if self._cancel_flag:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    future.result()

            with self._lock:
                if self._status == "running":
                    self._status = "completed"

        except InterruptedError:
            logger.info("API caption generation cancelled")
            self._set_status("idle")
        except Exception:
            traceback.print_exc()
            self._set_status("error", "API caption generation failed -- check the Terminal panel for details")
        finally:
            self._thread = None

    def _build_caption_prompt(self, base_prompt: str, image_path: Path, request: Any) -> str:
        hints: list[str] = []
        if getattr(request, "pass_filename", False):
            hints.append(f"Filename: {image_path.stem}")
        if getattr(request, "pass_metadata", False):
            meta = _extract_image_metadata(image_path)
            if meta:
                hints.append("Image metadata: " + "; ".join(f"{k}: {v}" for k, v in meta.items()))
        if getattr(request, "pass_current_caption", False):
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists():
                content = txt_path.read_text(encoding="utf-8").strip()
                if content:
                    hints.append(f"Current caption: {content}")

        if hints:
            return "\n".join(hints) + "\n\n" + base_prompt
        return base_prompt

    def _post_with_retry(
        self,
        session: requests.Session,
        url: str,
        payload: dict,
        headers: dict,
        *,
        max_retries: int = 6,
        timeout: int = 120,
        label: str = "API",
    ) -> dict:
        r_json: dict = {}
        for attempt in range(max_retries):
            if self._cancel_flag:
                raise InterruptedError("Cancelled")
            try:
                response = session.post(url, json=payload, headers=headers, timeout=timeout)
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    wait = min(5 * (2**attempt), 60)
                    time.sleep(wait)
                    continue
                raise
            if response.status_code == 429 and attempt < 1:
                wait = min(int(response.headers.get("Retry-After", 10)), 30)
                time.sleep(wait)
                continue
            if response.status_code in (500, 502, 503) and attempt < max_retries - 1:
                wait = min(5 * (attempt + 1), 30)
                time.sleep(wait)
                continue
            response.raise_for_status()
            r_json = response.json()
            break
        else:
            raise RuntimeError(f"{label} retries exhausted after {max_retries} attempts")
        return r_json

    def _call_openai_api(
        self,
        image_path: Path,
        prompt: str,
        config: Any,
        session: requests.Session,
    ) -> str:
        api_url = _normalize_openai_url(getattr(config, "api_url", "http://localhost:1234/v1/chat/completions"))
        api_key = getattr(config, "api_key", "") or ""
        model = getattr(config, "model_name", "") or "local-model"
        system_prompt = getattr(config, "system_prompt", "") or ""
        temperature = getattr(config, "temperature", 0.6)
        max_tokens = getattr(config, "max_tokens", -1)
        enable_thinking = getattr(config, "enable_thinking", False)
        timeout = getattr(config, "timeout", 120)

        b64 = _encode_image_base64(image_path)

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "enable_thinking": enable_thinking,
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        r_json = self._post_with_retry(
            session,
            api_url,
            payload,
            headers,
            max_retries=6,
            timeout=timeout,
            label="OpenAI API",
        )

        if "choices" not in r_json or not r_json["choices"]:
            error_detail = r_json.get("error", {})
            msg = (
                error_detail.get("message", str(error_detail)) if isinstance(error_detail, dict) else str(error_detail)
            )
            raise ValueError(f"API error: {msg}" if msg else f"Unexpected response: {str(r_json)[:300]}")

        caption = r_json["choices"][0]["message"]["content"]
        return _clean_model_output(caption)

    def _call_gemini_api(
        self,
        image_path: Path,
        prompt: str,
        config: Any,
        session: requests.Session,
    ) -> str:
        api_key = getattr(config, "api_key", "") or ""
        if not api_key:
            raise ValueError("Gemini API key is required")

        model = getattr(config, "model_name", "") or "gemini-1.5-flash"
        model_for_url = model.split("/")[-1] if "/" in model else model
        system_prompt = getattr(config, "system_prompt", "") or ""
        temperature = getattr(config, "temperature", 0.6)
        max_tokens = getattr(config, "max_tokens", -1)
        timeout = getattr(config, "timeout", 120)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_for_url}:generateContent?key={api_key}"
        b64 = _encode_image_base64(image_path)

        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    ],
                }
            ],
            "generationConfig": {"temperature": temperature},
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": {"text": system_prompt}}
        if max_tokens > 0:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        headers = {"Content-Type": "application/json"}

        r_json = self._post_with_retry(
            session,
            url,
            payload,
            headers,
            max_retries=5,
            timeout=timeout,
            label="Gemini API",
        )

        try:
            candidate = r_json["candidates"][0]
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "SAFETY":
                raise ValueError(f"Safety filter blocked: {candidate.get('safetyRatings', 'Unknown')}")
            caption = candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            if "promptFeedback" in r_json:
                block_reason = r_json["promptFeedback"].get("blockReason", "unknown")
                raise ValueError(f"Prompt blocked: {block_reason}") from exc
            raise ValueError(f"Unexpected Gemini response: {str(r_json)[:300]}") from exc

        return _clean_model_output(caption)

    def preview_caption(self, request: Any) -> dict:
        folder = getattr(request, "folder", "")
        image_path_str = getattr(request, "image_path", None)
        backend = getattr(request, "backend", "")
        model_name = getattr(request, "model", "")

        if image_path_str:
            image_path = Path(image_path_str)
        else:
            images = scan_image_folder(folder, getattr(request, "include_subdirectories", False))
            if not images:
                return {"ok": False, "error": "No images found in the specified folder"}
            image_path = random.choice(images)

        if not image_path.exists():
            return {"ok": False, "error": f"Image not found: {image_path}"}

        prompt = getattr(request, "prompt", "") or "Describe this image."
        full_prompt = self._build_caption_prompt(prompt, image_path, request)

        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((512, 512), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            thumbnail_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            thumbnail_b64 = ""

        try:
            if backend in VALID_API_BACKENDS:
                session = requests.Session()
                if backend == "openai":
                    caption = self._call_openai_api(image_path, full_prompt, request, session)
                else:
                    caption = self._call_gemini_api(image_path, full_prompt, request, session)
            elif model_name in CAPTION_MODEL_MAP:
                model = self._load_captioning_model(model_name)
                if model is None:
                    return {"ok": False, "error": f"Unknown model: {model_name}"}
                caption = model.caption_image(str(image_path))
            else:
                return {"ok": False, "error": f"Unknown backend/model: {backend or model_name}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        return {
            "ok": True,
            "caption": caption,
            "prompt_used": full_prompt,
            "image_path": str(image_path),
            "image_base64": thumbnail_b64,
        }

    def cancel(self) -> dict:
        with self._lock:
            if self._status != "running":
                return {"ok": False, "error": "No tool operation is running"}
            self._cancel_flag = True
            self._status = "idle"
            self._error_message = None
        return {"ok": True}

    def _progress_callback(self, current: int, total: int) -> None:
        if self._cancel_flag:
            raise InterruptedError("Tool operation cancelled by user")
        self._update_progress(current, total)

    def _error_callback(self, filename: str) -> None:
        logger.warning("Tool error processing file: %s", filename)

    def _load_captioning_model(self, model_name: str) -> Any:
        class_name = CAPTION_MODEL_MAP.get(model_name)
        if class_name is None:
            return None

        current_type = type(self._captioning_model).__name__ if self._captioning_model else None
        if current_type == class_name:
            return self._captioning_model

        from modules.util.torch_util import default_device

        import torch

        self._release_models()
        logger.info("Loading %s captioning model...", model_name)
        module = importlib.import_module(f"modules.module.{class_name}")
        cls = getattr(module, class_name)
        self._captioning_model = cls(default_device, torch.float16)
        return self._captioning_model

    def _load_masking_model(self, model_name: str, model_path: str | None = None) -> Any:
        if model_name == "YOLO":
            if not importlib.util.find_spec("ultralytics"):
                raise ImportError(
                    "Ultralytics is not installed. Install it with: pip install ultralytics\n"
                    "Note: Ultralytics has its own license (AGPL-3.0). "
                    "It is not bundled with OneTrainer."
                )
            if not model_path:
                raise ValueError("YOLO model requires a .pt model file path")
            current_type = type(self._masking_model).__name__ if self._masking_model else None
            cached_path = getattr(self._masking_model, "_yolo_model_path", None)
            if current_type != "YOLOMaskAdapter" or cached_path != model_path:
                self._release_models()
                logger.info("Loading YOLO masking model from %s...", model_path)
                self._masking_model = YOLOMaskAdapter(model_path)
            return self._masking_model

        class_name = MASK_MODEL_MAP.get(model_name)
        if class_name is None:
            return None

        current_type = type(self._masking_model).__name__ if self._masking_model else None
        if current_type == class_name:
            return self._masking_model

        from modules.util.torch_util import default_device

        import torch

        self._release_models()
        logger.info("Loading %s masking model...", model_name)
        module = importlib.import_module(f"modules.module.{class_name}")
        cls = getattr(module, class_name)
        self._masking_model = cls(default_device, torch.float32)
        return self._masking_model

    def _release_models(self) -> None:
        freed = False
        if self._captioning_model is not None:
            self._captioning_model = None
            freed = True
        if self._masking_model is not None:
            self._masking_model = None
            freed = True

        if freed:
            with suppress(Exception):
                from modules.util.torch_util import torch_gc

                torch_gc()


class YOLOMaskAdapter:
    def __init__(self, model_path: str):
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._yolo_model_path = model_path

    def mask_folder(
        self,
        sample_dir: str,
        prompts: list[str] | None = None,
        mode: str = "replace",
        threshold: float = 0.3,
        smooth_pixels: int = 5,
        expand_pixels: int = 10,
        alpha: float = 1.0,
        progress_callback: Any = None,
        error_callback: Any = None,
        include_subdirectories: bool = False,
    ) -> None:
        import cv2
        import numpy as np

        images = scan_image_folder(sample_dir, include_subdirectories)
        total = len(images)

        for idx, img_path in enumerate(images):
            try:
                mask_path = img_path.parent / f"{img_path.stem}-masklabel.png"
                if mode == "fill" and mask_path.exists():
                    if progress_callback:
                        progress_callback(idx + 1, total)
                    continue

                results = self._model.predict(str(img_path), conf=threshold, verbose=False)
                result = results[0] if results else None

                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                combined_mask = np.zeros((h, w), dtype=np.uint8)

                if result and result.masks is not None:
                    for mask_data in result.masks.data:
                        binary = (mask_data.cpu().numpy() * 255).astype(np.uint8)
                        if binary.shape[:2] != (h, w):
                            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_LINEAR)
                        combined_mask = cv2.bitwise_or(combined_mask, binary)

                if smooth_pixels > 0:
                    ksize = smooth_pixels * 2 + 1
                    combined_mask = cv2.GaussianBlur(combined_mask, (ksize, ksize), 0)
                if expand_pixels > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
                    )
                    combined_mask = cv2.dilate(combined_mask, kernel)
                _, combined_mask = cv2.threshold(combined_mask, 128, 255, cv2.THRESH_BINARY)

                if mode in ("replace", "fill"):
                    cv2.imwrite(str(mask_path), combined_mask)
                elif mode == "add" and mask_path.exists():
                    existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    merged = cv2.bitwise_or(existing, combined_mask)
                    cv2.imwrite(str(mask_path), merged)
                elif mode == "subtract" and mask_path.exists():
                    existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    merged = cv2.bitwise_and(existing, cv2.bitwise_not(combined_mask))
                    cv2.imwrite(str(mask_path), merged)
                else:
                    cv2.imwrite(str(mask_path), combined_mask)

            except Exception:
                logger.warning("YOLO mask failed for %s", img_path.name, exc_info=True)
                if error_callback:
                    error_callback(str(img_path))

            if progress_callback:
                progress_callback(idx + 1, total)
