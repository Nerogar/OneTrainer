import os
import sys

sys.path.append(os.getcwd())

import torch

from modules.util import create
from modules.util.args.SampleArgs import SampleArgs


def main():
    args = SampleArgs.parse_args()
    device = torch.device("cuda")

    model_loader = create.create_model_loader(args.model_type)
    model_setup = create.create_model_setup(args.model_type, device, device)

    print("Loading model " + args.base_model_name)
    model = model_loader.load(args.base_model_name, args.model_type)
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
