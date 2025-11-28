from enum import Enum


class VideoFormat(Enum):
    PNG_IMAGE_SEQUENCE = 'PNG_IMAGE_SEQUENCE'
    JPG_IMAGE_SEQUENCE = 'JPG_IMAGE_SEQUENCE'
    MP4 = 'MP4'

    def __str__(self):
        return self.value

    def extension(self) -> str:
        match self:
            case VideoFormat.PNG_IMAGE_SEQUENCE:
                return ".png"
            case VideoFormat.JPG_IMAGE_SEQUENCE:
                return ".jpg"
            case VideoFormat.MP4:
                return ".mp4"
            case _:
                return ""

    def pil_format(self) -> str:
        match self:
            case VideoFormat.PNG_IMAGE_SEQUENCE:
                return "PNG"
            case VideoFormat.JPG_IMAGE_SEQUENCE:
                return "JPEG"
            case _:
                return ""
