from modules.util.enum.BaseEnum import BaseEnum


class ImageOperations(BaseEnum):
    NONE = "none"
    VERIFY_IMG = "verify_img"
    SEQUENTIAL_RENAME = "sequential_rename"
    PROCESS_ALPHA = "process_alpha"
    RESIZE_LARGE_IMG = "resize_large_image"
    OPTIMIZE_PNG = "optimize_png"
    CONVERT_WEBP = "convert_webp"
    CONVERT_JXL = "convert_jxl"

    def pretty_print(self):
        return {
            ImageOperations.NONE: "No operation",
            ImageOperations.VERIFY_IMG: "Verifying images",
            ImageOperations.SEQUENTIAL_RENAME: "Sequential renaming",
            ImageOperations.PROCESS_ALPHA: "Processing transparent images",
            ImageOperations.RESIZE_LARGE_IMG: "Resizing large images",
            ImageOperations.OPTIMIZE_PNG: "Optimizing PNGs",
            ImageOperations.CONVERT_WEBP: "Converting to WebP",
            ImageOperations.CONVERT_JXL: "Converting to JPEG XL",
        }[self]
