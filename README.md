# OneTrainer

OneTrainer is a one-stop solution for all your stable diffusion training needs.

## Features

- **Supported models**: Stable Diffusion 1.5 and inpainting models
- **Model formats**: diffusers and ckpt models
- **Training methods**: Full fine-tuning, LoRA, embeddings
- **Masked Training**: Let the training focus on just certain parts of the samples.
- **Automatic backups**: Fully back up your training progress regularly during training. This includes all information to seamlessly continue training. 

## Planned Features

It is currently still in very early development. Planned features include:

- **User friendly UI**: I'm still evaluating options. The most likely solution is a web based UI.
- **Different models**: Training on all currently released Stable Diffusion models, this includes the base models (1.x
  and 2.x), inpainting models, the depth to image model, and SDXL once it is released
- **Different training methods**: Fine-Tuning, LoRA, Embeddings, ControlNet
- **VAE fine tuning**: Already implemented, but with limited functionality

## Installation

Installing OneTrainer requires Python 3.10 or newer. Then follow these steps:

- Clone the repository `git clone git@github.com:Nerogar/OneTrainer.git`
- Navigate into the cloned directory `cd OneTrainer`
- Set up a virtual environment `python -m venv venv`
- Activate the new venv `venv\scripts\activate`
- Install the requirements `pip install -r requirements_torch1.txt`

## Usage

OneTrainer is currently command line only. All commands need to be run inside the active venv created during
installation.

All functionality is split into different scrips located in the `scripts` directory. This currently includes:

- `train.py` The central training script
- `convert_model.py` A utility to convert between different model formats
- `sample.py` A utility to sample any model

To learn more about the different parameters, execute `<scipt-name> -h`. For example `scripts\train.py -h`