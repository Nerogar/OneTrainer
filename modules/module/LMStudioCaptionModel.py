import base64
import os
import re

from modules.module.BaseImageCaptionModel import BaseImageCaptionModel, CaptionSample

import requests

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LMStudioCaptionModel(BaseImageCaptionModel):
    """Captions images through an LM Studio OpenAI-compatible /v1 endpoint."""

    def __init__(self, server_url: str, system_prompt: str = "", user_prompt: str = ""):
        self.server_url = (server_url or "").strip().rstrip("/")
        self.system_prompt = system_prompt or ""
        self.user_prompt = user_prompt or ""
        self._model_id: str | None = None

    def _resolve_model_id(self) -> str:
        if self._model_id is None:
            response = requests.get(f"{self.server_url}/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("data") or []
            self._model_id = models[0]["id"] if models else "local-model"
        return self._model_id

    @staticmethod
    def _encode_image(image_filename: str) -> tuple[str, str]:
        ext = os.path.splitext(image_filename)[1].lower()
        mime_type = _MIME_TYPES.get(ext, "image/jpeg")
        with open(image_filename, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded, mime_type

    def generate_caption(
            self,
            caption_sample: CaptionSample,
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
    ):
        model_id = self._resolve_model_id()
        encoded_image, mime_type = self._encode_image(caption_sample.image_filename)

        messages = []
        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}},
                {"type": "text", "text": self.user_prompt},
            ],
        })

        response = requests.post(
            f"{self.server_url}/chat/completions",
            json={"model": model_id, "messages": messages},
            timeout=None,
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]

        generated = _THINK_RE.sub("", raw_text).replace("\n", " ").strip()
        return (caption_prefix + generated + caption_postfix).strip()
