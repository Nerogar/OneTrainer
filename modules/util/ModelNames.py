class ModelNames:
    def __init__(
            self,
            base_model: str = "",
            lora: str = "",
            embedding: list[str] = [],
    ):
        self.base_model = base_model
        self.lora = lora
        self.embedding = embedding
