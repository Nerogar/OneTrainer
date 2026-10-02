"""Minimal UI translation layer.

The English source string is used as the lookup key, so untranslated strings
fall back to the original text. Translations live in resources/i18n/<lang>.json
and the selected language is stored in training_user_settings/ui_settings.json.

Keys containing {placeholders} are templates for text built with f-strings,
e.g. "Text Encoder {i} Learning Rate" -> "文字編碼器 {i} 學習率".
"""
import json
import os
import re
from pathlib import Path

DEFAULT_LANGUAGE = "zh_TW"
LANGUAGES = {
    "en": "English",
    "zh_TW": "繁體中文",
}

_ROOT = Path(__file__).resolve().parents[2]
_I18N_DIR = _ROOT / "resources" / "i18n"
_SETTINGS_PATH = _ROOT / "training_user_settings" / "ui_settings.json"

_language: str | None = None
_translations: dict[str, str] = {}
_templates: list[tuple[re.Pattern, str]] = []
_PLACEHOLDER = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")


def _read_settings() -> dict:
    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def get_language() -> str:
    if _language is None:
        _load()
    return _language


def set_language(language: str) -> None:
    """Persist the language; it takes effect on the next start of the UI."""
    settings = _read_settings()
    settings["language"] = language
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)


def _compile_template(key: str) -> re.Pattern:
    pattern, last, seen = "", 0, set()
    for m in _PLACEHOLDER.finditer(key):
        name = m.group(1)
        group = f"(?P={name})" if name in seen else f"(?P<{name}>.+?)"
        seen.add(name)
        pattern += re.escape(key[last:m.start()].replace("{{", "{").replace("}}", "}")) + group
        last = m.end()
    pattern += re.escape(key[last:].replace("{{", "{").replace("}}", "}"))
    return re.compile(pattern, re.DOTALL)


def _load() -> None:
    global _language, _translations, _templates
    language = os.environ.get("OT_LANGUAGE") or _read_settings().get("language") or DEFAULT_LANGUAGE
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    _language = language
    _translations = {}
    _templates = []
    if language != "en":
        try:
            with open(_I18N_DIR / f"{language}.json", encoding="utf-8") as f:
                _translations = {k: v for k, v in json.load(f).items() if v}
        except (OSError, ValueError):
            pass
    for key, value in _translations.items():
        if _PLACEHOLDER.search(key):
            _templates.append((_compile_template(key), value))


def t(text: str | None) -> str | None:
    """Translate a UI string, falling back to the original text."""
    if not text:
        return text
    if _language is None:
        _load()
    translated = _translations.get(text)
    if translated is not None:
        return translated
    for pattern, value in _templates:
        m = pattern.fullmatch(text)
        if m:
            values = m.groupdict()
            return _PLACEHOLDER.sub(lambda p, values=values: values.get(p.group(1), p.group(0)), value)
    return text
