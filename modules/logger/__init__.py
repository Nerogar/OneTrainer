from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLogger(ABC):
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    @abstractmethod
    def log_image(self, tag: str, image: Any, step: int) -> None:
        pass

    @abstractmethod
    def log_text(self, tag: str, text: str, step: int) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            raise ImportError("tensorboard not installed. Install with: pip install tensorboard")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        self.writer.add_image(tag, image, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, step)

    def flush(self) -> None:
        self.writer.flush()


class WandbLogger(BaseLogger):
    def __init__(self, project: str, run_name: str = None, config: Dict = None):
        try:
            import wandb
            wandb.init(project=project, name=run_name, config=config)
            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.wandb.log({tag: value}, step=step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        self.wandb.log({tag: self.wandb.Image(image)}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.wandb.log({tag: text}, step=step)

    def flush(self) -> None:
        self.wandb.finish()


def get_logger(config) -> BaseLogger:
    logger_type = getattr(config, 'logger_type', 'tensorboard')
    if logger_type == 'tensorboard':
        log_dir = getattr(config, 'log_dir', 'logs')
        return TensorboardLogger(log_dir)
    elif logger_type == 'wandb':
        project = getattr(config, 'wandb_project', 'OneTrainer')
        run_name = getattr(config, 'wandb_run_name', None)
        wandb_config = getattr(config, 'wandb_config', None)
        return WandbLogger(project, run_name, wandb_config)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
