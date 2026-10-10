import json
from contextlib import suppress

from modules.util import path_util

CAPTION_UI_SETTINGS_FILE = path_util.canonical_join(".", "caption_ui_settings.json")

DEFAULT_SERVER_URL = "http://localhost:1234/v1"
DEFAULT_SYSTEM_PROMPT = (
    "You write concise, factual captions for training images. "
    "Reply with a single caption and no extra commentary."
)
DEFAULT_USER_PROMPT = "Describe this image in one concise caption."


def load_caption_ui_settings() -> dict:
    settings = {
        "server_url": DEFAULT_SERVER_URL,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
    }
    with suppress(Exception), open(CAPTION_UI_SETTINGS_FILE, encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            for key in settings:
                if isinstance(data.get(key), str):
                    settings[key] = data[key]
    return settings


def save_caption_ui_settings(server_url: str, system_prompt: str, user_prompt: str) -> None:
    with suppress(Exception):
        path_util.write_json_atomic(CAPTION_UI_SETTINGS_FILE, {
            "server_url": server_url,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
