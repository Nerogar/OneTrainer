import json
import os
import re

from PIL import Image

_ANGLE_BRACKET_SEGMENT_PATTERN = re.compile(r"<[^<>]*>")


def extract_metadata(path: str) -> dict:
    """Extract prompt and aspectratio from image metadata. Read-only."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.png':
        return _extract_png_metadata(path)
    else:
        return _extract_raw_metadata(path)


def _extract_png_metadata(path: str) -> dict:
    img = Image.open(path)
    try:
        info = img.info
    finally:
        img.close()

    for key in ('sui_image_params', 'parameters', 'Comment', 'prompt'):
        if key in info:
            try:
                data = json.loads(info[key])
                if isinstance(data, dict):
                    return {
                        'prompt': data.get('prompt', ''),
                        'aspectratio': data.get('aspectratio', ''),
                    }
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        'prompt': info.get('prompt', ''),
        'aspectratio': info.get('aspectratio', ''),
    }


def strip_angle_bracket_segments(prompt: str) -> str:
    """Remove metadata-like <...> segments from a prompt for caption export."""
    cleaned = _ANGLE_BRACKET_SEGMENT_PATTERN.sub("", prompt)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"^\s*,\s*|\s*,\s*$", "", cleaned)
    return cleaned.strip()


def _extract_raw_metadata(path: str) -> dict:
    """Scan raw file bytes for plaintext metadata (JPEG, WebP, etc.)."""
    CHUNK = 256 * 1024  # 256 KB covers metadata in most formats
    with open(path, 'rb') as f:
        raw = f.read(CHUNK)
        # Check for metadata markers in UTF-8 or UTF-16LE encoding
        has_marker = (b'"sui_image_params"' in raw or b'"prompt"' in raw
                      or b'"\x00p\x00r\x00o\x00m\x00p\x00t\x00"' in raw)
        if not has_marker:
            raw = raw + f.read()  # fall back to full read

    # Try UTF-8 first, then UTF-16 variants for generators that embed
    # metadata in wide-character encoding (e.g. SwarmUI WebP EXIF).
    for encoding in ('utf-8', 'utf-16-le', 'utf-16-be'):
        text = raw.decode(encoding, errors='surrogateescape' if encoding == 'utf-8' else 'ignore')

        prompt, aspectratio = _try_json_block(text)
        if prompt is not None:
            return {'prompt': prompt, 'aspectratio': aspectratio or ''}

        prompt = _find_json_string(text, 'prompt')
        if prompt:
            aspectratio = _find_json_string(text, 'aspectratio')
            return {'prompt': prompt, 'aspectratio': aspectratio or ''}

    return {'prompt': '', 'aspectratio': ''}


def _try_json_block(text: str) -> tuple[str | None, str | None]:
    """Try to find and parse a JSON object containing 'prompt'."""
    for block in _iter_json_objects(text):
        result = _parse_prompt_block(block)
        if result is not None:
            return result
    return None, None


def _parse_prompt_block(block: str) -> tuple[str, str | None] | None:
    try:
        data = json.loads(block)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and 'prompt' in data:
        return data.get('prompt'), data.get('aspectratio')
    return None


def _find_json_string(text: str, key: str) -> str | None:
    """Find a JSON string value by key. Handles escaped quotes."""
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text)
    if match:
        value = match.group(1)
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            return value.replace('\\"', '"').replace('\\\\', '\\')
    return None


def _iter_json_objects(text: str):
    """Yield balanced JSON object substrings from loose text."""
    depth = 0
    start = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if start is None:
            if ch == '{':
                start = i
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]
                start = None
                in_string = False
                escaped = False
