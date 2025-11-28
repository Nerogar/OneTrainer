from modules.util.enum.BaseEnum import BaseEnum


class ImageOptimization(BaseEnum):
    NONE = "none"
    PNG = "png"
    WEBP = "webp"
    JXL = "jxl"

    def pretty_print(self):
        return {
            ImageOptimization.NONE: "None",
            ImageOptimization.PNG: "Optimize PNGs",
            ImageOptimization.WEBP: "Convert to WebP",
            ImageOptimization.JXL: "Convert to JPEG XL",
        }[self]
