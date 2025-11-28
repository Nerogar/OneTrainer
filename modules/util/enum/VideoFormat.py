from modules.util.enum.BaseEnum import BaseEnum


class VideoFormat(BaseEnum):
    PNG_IMAGE_SEQUENCE = 'PNG_IMAGE_SEQUENCE'
    JPG_IMAGE_SEQUENCE = 'JPG_IMAGE_SEQUENCE'
    MP4 = 'MP4'

    def pretty_print(self):
        return {
            VideoFormat.PNG_IMAGE_SEQUENCE: "PNG Image Sequence",
            VideoFormat.JPG_IMAGE_SEQUENCE: "JPG Image Sequence",
            VideoFormat.MP4: "MP4",
        }[self]

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
