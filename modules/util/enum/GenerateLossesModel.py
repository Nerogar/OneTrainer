import torch
from modules.dataLoader import StableDiffusionFineTuneDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.args.TrainArgs import TrainArgs
from modules.util.TrainProgress import TrainProgress
from modules.util import create
from modules.util.torch_util import torch_gc
from torch.nn import Parameter
from tqdm import tqdm
from modules.util.dtype_util import allow_mixed_precision
from contextlib import nullcontext
import os
import json


class GenerateLossesModel:
    """Based on train args, writes a JSON instead of a model with filenames mapped to losses,
    in order of decreasing loss."""
    args: TrainArgs
    train_device: torch.device
    temp_device: torch.device
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: StableDiffusionFineTuneDataLoader
    model: BaseModel
    parameters: list[Parameter]

    def __init__(self, args: TrainArgs):
        # Create a copy of args because we will mutate
        # the batch size and gradient accumulation steps.
        args = TrainArgs.default_values().from_dict(args.to_dict())
        args.batch_size = 1
        args.gradient_accumulation_steps = 1

        self.args = args
        self.train_device = torch.device(self.args.train_device)
        self.temp_device = torch.device(self.args.temp_device)
    
    def start(self):
        if self.args.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model_loader = create.create_model_loader(self.args.model_type, self.args.training_method)
        self.model_setup = create.create_model_setup(
            self.args.model_type,
            self.train_device,
            self.temp_device,
            self.args.training_method,
            self.args.debug_mode,
        )

        model_names = self.args.model_names()

        self.model = self.model_loader.load(
            model_type=self.args.model_type,
            model_names=model_names,
            weight_dtypes=self.args.weight_dtypes(),
        )

        self.model_setup.setup_train_device(self.model, self.args)
        self.model_setup.setup_model(self.model, self.args)
        self.model.eval()
        self.model.train_progress = TrainProgress()
        torch_gc()

        self.data_loader = create.create_data_loader(
            self.train_device,
            self.temp_device,
            self.model,
            self.args.model_type,
            self.args.training_method,
            self.args,
            self.model.train_progress,
        )

        self.parameters = list(self.model_setup.create_parameters(self.model, self.args))

        self.data_loader.get_data_set().start_next_epoch()
        step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step")

        if allow_mixed_precision(self.args):
            forward_context = torch.autocast(self.train_device.type, dtype=self.args.train_dtype.torch_dtype())
        else:
            forward_context = nullcontext()

        filename_loss_list: list[tuple[str, float]] = []
        # Don't really need a backward pass here, so we can make the calculation MUCH faster.
        with forward_context, torch.inference_mode():
            for epoch_step, batch in enumerate(step_tqdm):
                    model_output_data = self.model_setup.predict(
                        self.model,
                        batch,
                        self.args,
                        self.model.train_progress,
                        deterministic=True,
                    )
                    loss = self.model_setup.calculate_loss(
                        self.model,
                        batch,
                        model_output_data,
                        self.args,
                    )
                    filename_loss_list.append((batch['image_path'][0], float(loss)))

        # Sort such that highest loss comes first
        filename_loss_list.sort(key=lambda x: x[1], reverse=True)
        filename_to_loss: dict[str, float] = {x[0]: x[1] for x in filename_loss_list}
        save_filename = f"{os.path.splitext(self.args.output_model_destination)[0]}.json"
        with open(save_filename, "w") as f:
            json.dump(filename_to_loss, f)