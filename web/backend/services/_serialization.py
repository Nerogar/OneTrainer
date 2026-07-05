import base64
import io
from typing import Any


def serialize_sample(sampler_output: Any) -> dict:
    from modules.util.enum.FileType import FileType

    file_type: FileType = sampler_output.file_type
    data = sampler_output.data

    if file_type == FileType.IMAGE:
        buf = io.BytesIO()
        data.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"file_type": "IMAGE", "format": "png", "data": encoded}

    if file_type == FileType.VIDEO:
        if data is None:
            return {"file_type": "VIDEO", "format": "raw", "data": None}
        if isinstance(data, bytes):
            encoded = base64.b64encode(data).decode("ascii")
            return {"file_type": "VIDEO", "format": "raw", "data": encoded}
        return {"file_type": "VIDEO", "format": "raw", "data": None}

    if file_type == FileType.AUDIO:
        if data is None:
            return {"file_type": "AUDIO", "format": "raw", "data": None}
        encoded = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else None
        return {"file_type": "AUDIO", "format": "raw", "data": encoded}

    return {"file_type": str(file_type), "format": "unknown", "data": None}
