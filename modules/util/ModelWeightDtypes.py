from modules.util.enum.DataType import DataType


class ModelWeightDtypes:
    def __init__(
            self,
            text_encoder: DataType,
            unet: DataType,
            vae: DataType,
            lora: DataType,
            embedding: DataType,
    ):
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.lora = lora
        self.embedding = embedding

    def all_dtypes(self) -> list:
        return [
            self.text_encoder,
            self.unet,
            self.vae,
            self.lora,
            self.embedding,
        ]
