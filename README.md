# OneTrainer

OneTrainer is a one-stop solution for all your stable diffusion training needs.

<a href="https://discord.gg/KwgcQd5scF"><img src="https://discord.com/api/guilds/1102003518203756564/widget.png" alt="OneTrainer Discord"/></a>

## Features

- **Supported models**: Stable Diffusion 1.5 and inpainting models
- **Model formats**: diffusers and ckpt models
- **Training methods**: Full fine-tuning, LoRA, embeddings
- **Masked Training**: Let the training focus on just certain parts of the samples.
- **Automatic backups**: Fully back up your training progress regularly during training. This includes all information
  to seamlessly continue training.

## Planned Features

It is currently still in very early development. Planned features include:

- **User friendly UI**: I'm still evaluating options. The most likely solution is a web based UI.
- **Different models**: Training on all currently released Stable Diffusion models, this includes the base models (1.x
  and 2.x), inpainting models, the depth to image model, and SDXL once it is released
- **Different training methods**: Fine-Tuning, LoRA, Embeddings, ControlNet
- **VAE fine tuning**: Already implemented, but with limited functionality

## Installation

Installing OneTrainer requires Python 3.10 or newer. Then follow these steps:

Automatic installation

- Clone the repository `git clone https://github.com/Nerogar/OneTrainer.git`
- Run `install.bat`

Manual installation

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

To start the UI, run `start-ui.bat`

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