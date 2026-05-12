from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class LogWith(Enum):
    TENSORBOARD = 'tensorboard'
    WANDB = 'wandb'
    NONE = 'none'


@dataclass
class Settings:
    log_with: LogWith = LogWith.NONE
    wandb_project: str = ''
    wandb_run_name: str = ''
    wandb_tags: List[str] = field(default_factory=list)
    wandb_resume: bool = False
    wandb_run_id: Optional[str] = None
