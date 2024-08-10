import json
import os

import torch
from tqdm import tqdm

from modules.dataLoader import StableDiffusionFineTuneDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc


class GenerateLossesModel:
    """Based on train args, writes a JSON instead of a model with filenames mapped to losses,
    in order of decreasing loss."""
    config: TrainConfig
    train_device: torch.device
    temp_device: torch.device
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: StableDiffusionFineTuneDataLoader
    model: BaseModel

    def __init__(self, config: TrainConfig, output_path: str):
        # Create a copy of args because we will mutate
        # the batch size and gradient accumulation steps.
        config = TrainConfig.default_values().from_dict(config.to_dict())
        config.batch_size = 1
        config.gradient_accumulation_steps = 1

        self.config = config
        self.output_path = output_path
        self.train_device = torch.device(self.config.train_device)
        self.temp_device = torch.device(self.config.temp_device)
    
    def start(self):
        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model_loader = create.create_model_loader(self.config.model_type, self.config.training_method)
        self.model_setup = create.create_model_setup(
            self.config.model_type,
            self.train_device,
            self.temp_device,
            self.config.training_method,
            self.config.debug_mode,
        )

        model_names = self.config.model_names()

        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
        )
        self.model.train_config = self.config

        self.model_setup.setup_model(self.model, self.config)
        self.model.eval()
        self.model.train_progress = TrainProgress()
        torch_gc()

        self.data_loader = create.create_data_loader(
            self.train_device,
            self.temp_device,
            self.model,
            self.config.model_type,
            self.config.training_method,
            self.config,
            self.model.train_progress,
        )

        self.data_loader.get_data_set().start_next_epoch()
        step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step")

        self.model_setup.setup_train_device(self.model, self.config)

        filename_loss_list: list[tuple[str, float]] = []
        # Don't really need a backward pass here, so we can make the calculation MUCH faster.
        with torch.inference_mode():
            for epoch_step, batch in enumerate(step_tqdm):
                    model_output_data = self.model_setup.predict(
                        self.model,
                        batch,
                        self.config,
                        self.model.train_progress,
                        deterministic=True,
                    )
                    loss = self.model_setup.calculate_loss(
                        self.model,
                        batch,
                        model_output_data,
                        self.config,
                    )
                    filename_loss_list.append((batch['image_path'][0], float(loss)))

        # Sort such that highest loss comes first
        filename_loss_list.sort(key=lambda x: x[1], reverse=True)
        filename_to_loss: dict[str, float] = {x[0]: x[1] for x in filename_loss_list}
        with open(self.output_path, "w") as f:
            json.dump(filename_to_loss, f, indent=4)