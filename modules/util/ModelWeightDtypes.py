import inspect

from modules.util.enum.DataType import DataType


class ModelWeightDtypes:
    def __init__(
            self,
            unet: DataType,
            prior: DataType,
            text_encoder: DataType,
            text_encoder_2: DataType,
            vae: DataType,
            effnet_encoder: DataType,
            decoder: DataType,
            decoder_text_encoder: DataType,
            decoder_vqgan: DataType,
            lora: DataType,
            embedding: DataType,
    ):
        self.unet = unet
        self.prior = prior
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.effnet_encoder = effnet_encoder
        self.decoder = decoder
        self.decoder_text_encoder = decoder_text_encoder
        self.decoder_vqgan = decoder_vqgan
        self.lora = lora
        self.embedding = embedding

    def all_dtypes(self) -> list:
        return [
            self.unet,
            self.prior,
            self.text_encoder,
            self.text_encoder_2,
            self.vae,
            self.effnet_encoder,
            self.decoder,
            self.decoder_text_encoder,
            self.decoder_vqgan,
            self.lora,
            self.embedding,
        ]

    @staticmethod
    def from_single_dtype(dtype:DataType):
        params = [dtype for _ in set(inspect.signature(ModelWeightDtypes).parameters.keys())]
        return ModelWeightDtypes(*params)
