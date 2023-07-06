# OneTrainer

OneTrainer is a one-stop solution for all your stable diffusion training needs.

<a href="https://discord.gg/KwgcQd5scF"><img src="https://discord.com/api/guilds/1102003518203756564/widget.png" alt="OneTrainer Discord"/></a>

## Features

- **Supported models**: Stable Diffusion 1.5, 2.0, 2.1 and inpainting models
- **Model formats**: diffusers and ckpt models
- **Training methods**: Full fine-tuning, LoRA, embeddings
- **Masked Training**: Let the training focus on just certain parts of the samples.
- **Automatic backups**: Fully back up your training progress regularly during training. This includes all information
  to seamlessly continue training.
- **Image augmentation**: Apply random transforms such as rotation, brightness, contrast or saturation to each image
  sample to quickly create a more diverse dataset.
- **Tensorboard**: A simple tensorboard integration to track the training progress.
- **Multiple prompts per image**: Train the model on multiple different prompts per image sample.
- **Noise Scheduler Rescaling**: From the paper
  [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)
- **EMA**: Train you own EMA model. Optionally keep EMA weights in CPU memory to reduce VRAM usage.

## Planned Features

While OneTrainer already has many useful features, it is still being developed and improved. Here are some features that
are planned for the future:

- **Different models**: Training on all currently released Stable Diffusion models, this includes the base models (1.x
  and 2.x), inpainting models, the depth to image model, and SDXL once it is released. Other publicly released models
  might also be supported in the future.
- **VAE fine-tuning**: Already implemented, but with limited functionality
- **Tooling around dataset management**: Automatic tagging, sorting of images, etc.

A more detailed list can be found [here](ThingsToAdd.md).

## Installation

Installing OneTrainer requires Python 3.10. You can download Python here https://www.python.org/downloads/windows/.
Then follow these steps:

Automatic installation (Windows)

- Clone the repository `git clone https://github.com/Nerogar/OneTrainer.git`
- Run `install.bat`

Manual installation (Windows and other systems)

- Clone the repository `git clone https://github.com/Nerogar/OneTrainer.git`
- Navigate into the cloned directory `cd OneTrainer`
- Set up a virtual environment `python -m venv venv`
- Activate the new venv `venv\scripts\activate`
- Install the requirements `pip install -r requirements.txt`

## Updating

Automatic update

- Run `update.bat`

Manual update

- Pull changes `git pull`
- Activate the venv `venv\scripts\activate`
- Re-Install all requirements `pip install -r requirements.txt --force-reinstall`

## Usage

To start the UI, run `start-ui.bat`. [You can find a quick start guide here.](docs/QuickStartGuide.md)

If you need more control, OneTrainer supports two modes of operation. Command line only, and a UI.
All commands need to be run inside the active venv created during installation.

All functionality is split into different scrips located in the `scripts` directory. This currently includes:

- `train.py` The central training script
- `train_ui.py` A UI for training
- `convert_model.py` A utility to convert between different model formats
- `sample.py` A utility to sample any model

To learn more about the different parameters, execute `<scipt-name> -h`. For example `python scripts\train.py -h`

## Contributing

Contributions are always welcome in any form. You can open issues, participate in discussions, or even open pull
requests for new or improved functionality. You can find more information [here](docs/Contributing.md)

Before you start looking at the code, I recommend reading about the project structure [here](docs/ProjectStructure.md).
For in depth discussions, you should consider joining the [Discord](https://discord.gg/KwgcQd5scF) server.

## Related Projects

- **[MGDS](https://github.com/Nerogar/mgds)**: A custom dataset implementation for Pytorch that is built around the idea
  of a node based graph.
- **[StableTuner](https://github.com/devilismyfriend/StableTuner)**: Another training application for Stable Diffusion.
  OneTrainer takes a lot of inspiration from StableTuner and wouldn't exist without it.
- **[Visions of Chaos](https://softology.pro/voc.htm)**: A collection of machine learning tools that also includes
  OneTrainer.