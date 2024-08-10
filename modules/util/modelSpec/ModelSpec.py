from datetime import datetime


class ModelSpec:
    def __init__(
            self,
            architecture: str = "",
            implementation: str = "",
            title: str = "",
            description: str = "",
            author: str = "",
            date: str = datetime.now().strftime("%Y-%m-%d"),
            hash_sha256: str = "",
            license: str = "",
            thumbnail: str = "",
            resolution: str = "",
            prediction_type: str = "",
            usage_hint: str = "",
    ):
        self.sai_model_spec = "1.0.0"
        self.architecture = architecture
        self.implementation = implementation
        self.title = title
        self.description = description
        self.author = author
        self.date = date
        self.hash_sha256 = hash_sha256
        self.license = license
        self.thumbnail = thumbnail
        self.resolution = resolution
        self.prediction_type = prediction_type
        self.usage_hint = usage_hint

    def __is_required(self, key: str) -> bool:
        return key in [
            "sai_model_spec",
            "architecture",
            "implementation",
            "title",
        ]

    def to_dict(self):
        data = {}
        for (key, value) in vars(self).items():
            if self.__is_required(key) or (value is not None and value != ""):
                data["modelspec." + key] = value

        return data

    @staticmethod
    def from_dict(data) -> 'ModelSpec':
        model_spec = ModelSpec()
        for (key, value) in vars(model_spec).items():
            try:
                if isinstance(value, str):
                    setattr(model_spec, key, data["modelspec." + key])
            except:
                pass

        return model_spec
