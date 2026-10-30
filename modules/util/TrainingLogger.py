import os
from typing import Any

from modules.util.config.TrainConfig import TrainConfig

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import wandb
from PIL import Image


class TrainingLogger:
    """Logs training metrics and samples to Tensorboard, and optionally to Weights & Biases."""

    def __init__(self, config: TrainConfig, tensorboard_log_dir: str):
        self.config = config
        self.tensorboard: SummaryWriter | None = None
        self.wandb = None
        self._wandb_buffer: dict[int, dict[str, Any]] = {}
        self._wandb_max_step: int = -1

        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.tensorboard = SummaryWriter(
            os.path.join(tensorboard_log_dir, config.run_name)
        )

        if config.wandb:
            api_key = self.config.secrets.wandb_api_key.strip()
            project = self.config.wandb_project.strip() or "OneTrainer"
            run_name = self.config.run_name

            self.wandb = wandb.init(
                project=project,
                name=run_name,
                config=self.config.to_pack_dict(secrets=False),
                dir=self.config.workspace_dir,
                settings=wandb.Settings(api_key=api_key) if api_key else None,
            )

    def _wandb_log(self, data: dict[str, Any], step: int):
        if self.wandb is None:
            return
        if step in self._wandb_buffer:
            self._wandb_buffer[step].update(data)
        else:
            self._wandb_buffer[step] = data

        self._wandb_max_step = max(self._wandb_max_step, step)

        for s in sorted(self._wandb_buffer):
            if s >= self._wandb_max_step:
                break
            entry = self._wandb_buffer.pop(s)
            self.wandb.log(entry, step=s)

    def _wandb_flush_all(self):
        if self.wandb is None:
            return
        for s in sorted(self._wandb_buffer):
            entry = self._wandb_buffer.pop(s)
            self.wandb.log(entry, step=s)

    def add_scalar(self, tag: str, value: float, step: int):
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag, value, step)
        if self.wandb is not None:
            self._wandb_log({tag: value}, step)

    def add_image(self, tag: str, image: Any, step: int):
        if self.tensorboard is not None:
            self.tensorboard.add_image(
                tag,
                pil_to_tensor(image) if isinstance(image, Image.Image) else image,
                step,
            )
        if self.wandb is not None:
            self._wandb_log({tag: wandb.Image(image)}, step)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
            self.tensorboard = None
        if self.wandb is not None:
            self._wandb_flush_all()
            self.wandb.finish()
            self.wandb = None
