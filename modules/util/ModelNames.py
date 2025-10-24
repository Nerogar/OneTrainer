class EmbeddingName:
    def __init__(
            self,
            uuid: str,
            model_name: str,
    ):
        self.uuid = uuid
        self.model_name = model_name


class ModelNames:
    def __init__(
            self,
            base_model: str = "",
            prior_model: str = "",
            transformer_model: str = "",
            effnet_encoder_model: str = "",
            decoder_model: str = "",
            text_encoder_4: str = "",
            vae_model: str = "",
            lora: str = "",
            embedding: EmbeddingName | None = None,
            additional_embeddings: list[EmbeddingName] | None = None,
            include_text_encoder: bool = True,
            include_text_encoder_2: bool = True,
            include_text_encoder_3: bool = True,
            include_text_encoder_4: bool = True,
    ):
        self.base_model = base_model
        self.prior_model = prior_model
        self.transformer_model = transformer_model
        self.effnet_encoder_model = effnet_encoder_model
        self.decoder_model = decoder_model
        self.text_encoder_4 = text_encoder_4
        self.vae_model = vae_model
        self.lora = lora
        self.embedding = embedding
        self.additional_embeddings = [] if additional_embeddings is None else additional_embeddings
        self.include_text_encoder = include_text_encoder
        self.include_text_encoder_2 = include_text_encoder_2
        self.include_text_encoder_3 = include_text_encoder_3
        self.include_text_encoder_4 = include_text_encoder_4

    def all_embedding(self):
        if self.embedding is not None:
            return self.additional_embeddings + [self.embedding]
        else:
            return self.additional_embeddings
