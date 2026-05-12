from modules.logger.base_logger import BaseLogger


class DummyLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_metrics(self, metrics, step=None):
        pass

    def log_hyperparams(self, params):
        pass

    def log_parameters(self, params):
        pass

    def log_artifact(self, local_path, artifact_path=None):
        pass

    def save(self):
        pass

    def close(self):
        pass
