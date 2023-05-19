import os
import sys

sys.path.append(os.getcwd())

import torch

from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util import create
from modules.util.args.SampleArgs import SampleArgs


def main():
    args = SampleArgs.parse_args()
    device = torch.device("cuda")

    training_method = TrainingMethod.FINE_TUNE
    if args.embedding_name is not None:
        training_method = TrainingMethod.EMBEDDING

    model_loader = create.create_model_loader(args.model_type, training_method=training_method)
    model_setup = create.create_model_setup(args.model_type, device, device, training_method=training_method)

    print("Loading model " + args.base_model_name)
    model = model_loader.load(args.model_type, args.base_model_name, args.embedding_name)
    model_setup.setup_eval_device(model)

    model_sampler = create.create_model_sampler(
        model=model,
        model_type=args.model_type,
        train_device=device
    )

    print("Sampling " + args.destination)
    model_sampler.sample(prompt=args.prompt, size=(512, 512), seed=42, destination=args.destination)


if __name__ == '__main__':
    main()
