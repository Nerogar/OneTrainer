# OneTrainer

OneTrainer is a one-stop solution for all your stable diffusion training needs.

It is currently still in very early development. Planned features include:
- **Different models**: Training on all currently released Stable Diffusion models, this includes the base models (1.x and 2.x), inpainting models and the depth to image model
- **Different training methods**: Fine-Tuning, LoRA, Embeddings, ControlNet
- **VAE fine tuning**
- **Masked Training**: Let the training focus on just certain parts of the samples.

## Installation

Installing OneTrainer requires Python 3.10 or newer. Then follow these steps:
- Clone the repository `git clone git@github.com:Nerogar/OneTrainer.git`
- Navigate into the cloned directory `cd OneTrainer`
- Set up a virtual environment `python -m venv venv`
- Activate the new venv `venv\scripts\activate`
- Install the requirements `pip install -r requirements_torch1.txt`

## Usage

OneTrainer is currently command line only. All commands need to be run inside the active venv created during installation.

All functionality is split into different scrips located in the `scripts` directory. This currently includes:
- `train.py` The central training script
- `convert_model.py` A utility to convert between different model formats
- `sample.py` A utility to sample any model

To learn more about the different parameters, execute `<scipt-name> -h`. For example `scripts\train.py -h`