class ModelNames:
    def __init__(
            self,
            base_model: str = "",
            prior_model: str = "",
            effnet_encoder_model: str = "",
            decoder_model: str = "",
            vae_model: str = "",
            lora: str = "",
            embedding: list[str] = None,
    ):
        self.base_model = base_model
        self.prior_model = prior_model
        self.effnet_encoder_model = effnet_encoder_model
        self.decoder_model = decoder_model
        self.vae_model = vae_model
        self.lora = lora
        self.embedding = [] if embedding is None else embedding
