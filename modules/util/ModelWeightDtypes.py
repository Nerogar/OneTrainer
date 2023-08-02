class ModelWeightDtypes:
    def __init__(
            self,
            text_encoder,
            unet,
            vae,
            lora,
    ):
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.lora = lora
