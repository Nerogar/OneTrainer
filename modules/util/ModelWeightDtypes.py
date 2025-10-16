import inspect

from modules.util.enum.DataType import DataType


class ModelWeightDtypes:
    def __init__(
            self,
            train_dtype: DataType,
            fallback_train_dtype: DataType,
            unet: DataType,
            prior: DataType,
            transformer: DataType,
            text_encoder: DataType,
            text_encoder_2: DataType,
            text_encoder_3: DataType,
            text_encoder_4: DataType,
            vae: DataType,
            effnet_encoder: DataType,
            decoder: DataType,
            decoder_text_encoder: DataType,
            decoder_vqgan: DataType,
            lora: DataType,
            embedding: DataType,
    ):
        self.train_dtype = train_dtype
        self.fallback_train_dtype = fallback_train_dtype

        self.unet = unet
        self.prior = prior
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.text_encoder_4 = text_encoder_4
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
            self.transformer,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
            self.text_encoder_4,
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
