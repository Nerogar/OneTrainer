import os
import wandb
from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        self.checkpoint_dir = config.get('checkpoint_dir')
        self.run = wandb.init(
            project=config.get('project', 'default'),
            name=config.get('name'),
            tags=config.get('tags'),
            config=config
        )

    def log_metrics(self, metrics, step):
        if metrics:
            wandb.log(metrics, step=step)

    def log_image(self, key, image, step):
        wandb.log({key: wandb.Image(image)}, step=step)

    def log_histogram(self, key, values, step):
        wandb.log({key: wandb.Histogram(values)}, step=step)

    def close(self):
        if self.checkpoint_dir and os.path.isdir(self.checkpoint_dir):
            for root, _, files in os.walk(self.checkpoint_dir):
                for file in files:
                    if file.endswith('.ckpt'):
                        wandb.save(os.path.join(root, file))
        if self.run:
            self.run.finish()
