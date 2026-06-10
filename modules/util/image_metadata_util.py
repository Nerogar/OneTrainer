import json
import os
import re
import zlib

_ANGLE_BRACKET_SEGMENT_PATTERN = re.compile(r"<[^<>]*>")

_CLIP_TEXT_ENCODE_TYPES = frozenset(
    {
        "CLIPTextEncode",
        "CLIPTextEncodeSDXL",
        "CLIPTextEncodeSD3",
        "CLIPTextEncodeFlux",
        "CLIPTextEncodeHunyuanDiT",
    },
)

_SAMPLER_TYPES = frozenset(
    {
        "KSampler",
        "KSamplerAdvanced",
        "SamplerCustom",
        "SamplerCustomAdvanced",
    },
)


def extract_metadata(path: str) -> dict:
    """Extract prompt and aspectratio from image metadata. Read-only."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        return _extract_png_metadata(path)
    else:
        return _extract_raw_metadata(path)


def _extract_png_metadata(path: str) -> dict:
    info = _read_png_text_chunks(path)

    for key in ("sui_image_params", "parameters", "Comment", "prompt"):
        if key not in info:
            continue
        value = info[key]
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                if "prompt" in data:
                    return {
                        "prompt": data["prompt"],
                        "aspectratio": data.get("aspectratio", ""),
                    }
                prompt = _extract_comfyui_prompt(data)
                if prompt:
                    return {"prompt": prompt, "aspectratio": ""}
        except (json.JSONDecodeError, TypeError):
            if key == "parameters":
                prompt = _extract_a1111_prompt(value)
                if prompt:
                    return {"prompt": prompt, "aspectratio": ""}

    return {
        "prompt": info.get("prompt", ""),
        "aspectratio": info.get("aspectratio", ""),
    }


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_png_text_chunks(path: str) -> dict[str, str]:
    """Read text metadata from PNG tEXt/zTXt/iTXt chunks without decoding image data."""
    info: dict[str, str] = {}
    try:
        with open(path, "rb") as f:
            if f.read(8) != _PNG_SIGNATURE:
                return info
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                length = int.from_bytes(header[:4], "big")
                chunk_type = header[4:8]
                if chunk_type == b"IEND":
                    break
                if chunk_type in (b"tEXt", b"zTXt", b"iTXt"):
                    data = f.read(length)
                    f.seek(4, 1)  # skip CRC
                    _parse_png_text_chunk(chunk_type, data, info)
                else:
                    f.seek(length + 4, 1)  # skip data + CRC
    except Exception:
        pass
    return info


def _parse_png_text_chunk(chunk_type: bytes, data: bytes, info: dict[str, str]):
    """Parse a single PNG text chunk into the info dict."""
    try:
        sep = data.index(b"\x00")
        if chunk_type == b"tEXt":
            info[data[:sep].decode("latin-1")] = data[sep + 1 :].decode("latin-1")
        elif chunk_type == b"zTXt":
            # format: keyword \x00 compression_method compressed_text
            info[data[:sep].decode("latin-1")] = zlib.decompress(data[sep + 2 :]).decode("latin-1")
        elif chunk_type == b"iTXt":
            key = data[:sep].decode("utf-8")
            compression_flag = data[sep + 1]
            rest = data[sep + 3 :]  # skip compression_flag + compression_method
            sep2 = rest.index(b"\x00")  # end of language tag
            rest = rest[sep2 + 1 :]
            sep3 = rest.index(b"\x00")  # end of translated keyword
            text_data = rest[sep3 + 1 :]
            if compression_flag:
                info[key] = zlib.decompress(text_data).decode("utf-8")
            else:
                info[key] = text_data.decode("utf-8")
    except Exception:
        pass


def strip_angle_bracket_segments(prompt: str) -> str:
    """Remove metadata-like <...> segments from a prompt for caption export."""
    cleaned = _ANGLE_BRACKET_SEGMENT_PATTERN.sub("", prompt)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"^\s*,\s*|\s*,\s*$", "", cleaned)
    cleaned = cleaned.strip()
    # Flatten multi-line prompts to a single line (e.g. SwarmUI "line1\nline2" style)
    cleaned = re.sub(r"\s*\n\s*", ", ", cleaned)
    return cleaned


def _extract_a1111_prompt(text: str) -> str:
    """Extract the positive prompt from A1111/Forge plain-text parameters."""
    if not text or not text.strip():
        return ""
    # Everything before "Negative prompt:" line
    idx = text.find("\nNegative prompt:")
    if idx != -1:
        return text[:idx].strip()
    # Or before common parameter lines
    for marker in ("\nSteps:", "\nSampler:", "\nSize:", "\nModel:"):
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


def _extract_comfyui_prompt(data: dict) -> str:
    """Extract the positive prompt from a ComfyUI workflow node dict."""
    text_nodes: dict[str, str] = {}
    sampler = None

    for node_id, node in data.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        if cls in _CLIP_TEXT_ENCODE_TYPES:
            for text_key in ("text", "text_g", "text_l"):
                val = inputs.get(text_key)
                if isinstance(val, str) and val.strip():
                    text_nodes[str(node_id)] = val
                    break

        if cls in _SAMPLER_TYPES and sampler is None:
            sampler = node

    if not text_nodes:
        return ""

    # Trace sampler's positive conditioning to a text encoder
    if sampler:
        pos_ref = sampler.get("inputs", {}).get("positive")
        if isinstance(pos_ref, list) and pos_ref:
            pos_id = str(pos_ref[0])
            if pos_id in text_nodes:
                return text_nodes[pos_id]

    # Fallback: longest text
    return max(text_nodes.values(), key=len)


def _has_metadata_marker(raw: bytes) -> bool:
    """Byte-level (C-speed) check for a prompt/params marker in UTF-8,
    UTF-16-LE, or UTF-16-BE encoding."""
    return (
        b'"sui_image_params"' in raw
        or b'"prompt"' in raw
        or b'"\x00p\x00r\x00o\x00m\x00p\x00t\x00"' in raw  # UTF-16-LE
        or b'\x00"\x00p\x00r\x00o\x00m\x00p\x00t\x00"' in raw  # UTF-16-BE
    )


def _extract_raw_metadata(path: str) -> dict:
    """Scan raw file bytes for plaintext metadata (JPEG, WebP, etc.).

    Cost discipline matters here — this runs once per file over entire
    datasets. Files without any prompt marker return after pure byte-level
    ``in`` checks (no decode, no scanning). Files with a marker take a
    C-speed regex first; the character-by-character balanced-JSON scan is a
    last resort for layouts the regex can't see."""
    CHUNK = 256 * 1024  # 256 KB covers metadata in most formats
    with open(path, "rb") as f:
        raw = f.read(CHUNK)
        if not _has_metadata_marker(raw):
            raw += f.read()  # metadata may sit past the first chunk
            if not _has_metadata_marker(raw):
                return {"prompt": "", "aspectratio": ""}

    # Try UTF-8 first, then UTF-16 variants for generators that embed
    # metadata in wide-character encoding (e.g. SwarmUI WebP EXIF).
    for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
        text = raw.decode(encoding, errors="surrogateescape" if encoding == "utf-8" else "ignore")

        prompt = _find_json_string(text, "prompt")
        if prompt:
            aspectratio = _find_json_string(text, "aspectratio")
            return {"prompt": prompt, "aspectratio": aspectratio or ""}

        prompt, aspectratio = _try_json_block(text)
        if prompt is not None:
            return {"prompt": prompt, "aspectratio": aspectratio or ""}

    return {"prompt": "", "aspectratio": ""}


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
    if isinstance(data, dict) and "prompt" in data:
        return data.get("prompt"), data.get("aspectratio")
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
            return value.replace('\\"', '"').replace("\\\\", "\\")
    return None


def _iter_json_objects(text: str):
    """Yield balanced JSON object substrings from loose text."""
    depth = 0
    start = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]
                start = None
                in_string = False
                escaped = False
