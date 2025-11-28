from modules.util.enum.BaseEnum import BaseEnum


class ImageMegapixels(BaseEnum):
    ONE_MEGAPIXEL = 1_048_576
    COMPUTE_PROOF_MEGAPIXEL_THRESHOLD = 4_194_304
    MIDDLEGROUND_MEGAPIXEL_THRESHOLD = 8_388_608
    FUTURE_PROOF_MEGAPIXEL_THRESHOLD = 16_777_216
    CUSTOM = -1

    def __str__(self):
        return str(self.value)

    def pretty_print(self):
        return {
            ImageMegapixels.ONE_MEGAPIXEL: "1MP",
            ImageMegapixels.COMPUTE_PROOF_MEGAPIXEL_THRESHOLD: "Compute Proof (4MP)",
            ImageMegapixels.MIDDLEGROUND_MEGAPIXEL_THRESHOLD: "Middleground (8MP)",
            ImageMegapixels.FUTURE_PROOF_MEGAPIXEL_THRESHOLD: "Future Proof (16MP)",
            ImageMegapixels.CUSTOM: "Custom",
        }[self]
