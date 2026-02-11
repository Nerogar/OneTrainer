# OneTrainer

OneTrainer is a one-stop solution for all your Diffusion training needs.

<a href="https://discord.gg/KwgcQd5scF"><img src="https://discord.com/api/guilds/1102003518203756564/widget.png" alt="OneTrainer Discord"/></a><br>

## Features

-   **Supported models**: Z-Image, Qwen Image, FLUX.1, Flux.2 Dev and Klein, Chroma, Stable Diffusion 1.5, 2.0, 2.1, 3.0, 3.5, SDXL, Würstchen-v2, Stable Cascade,
    PixArt-Alpha, PixArt-Sigma, Sana, Hunyuan Video and inpainting models
-   **Model formats**: diffusers and ckpt models
-   **Training methods**: Full fine-tuning, LoRA, embeddings
-   **Masked Training**: Let the training focus on just certain parts of the samples
-   **Automatic backups**: Fully back up your training progress regularly during training. This includes all information to seamlessly continue training
-   **Image augmentation**: Apply random transforms such as rotation, brightness, contrast or saturation to each image sample to quickly create a more diverse dataset
-   **TensorBoard**: A simple TensorBoard integration to track the training progress
-   **Multiple prompts per image**: Train the model on multiple different prompts per image sample
-   **Noise Scheduler Rescaling**: From the paper
    [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)
-   **EMA**: Train your own EMA model. Optionally keep EMA weights in CPU memory to reduce VRAM usage
-   **Aspect Ratio Bucketing**: Automatically train on multiple aspect ratios at a time. Just select the target resolutions, buckets are created automatically
-   **Multi-Resolution Training**: Train multiple resolutions at the same time
-   **Dataset Tooling**: Automatically caption your dataset using BLIP, BLIP2 and WD-1.4, or create masks for masked training using ClipSeg or Rembg
-   **Model Tooling**: Convert between different model formats from a simple UI
-   **Sampling UI**: Sample the model during training without switching to a different application

![OneTrainerGUI.gif](resources/images/OneTrainerGUI.gif)

> [!NOTE]
> Explore our 📚 wiki for essential tips and tutorials after installing. Start [here!](https://github.com/Nerogar/OneTrainer/wiki).
> For command-line usage, see the [CLI Mode section](#cli-mode).


## Installation

> [!IMPORTANT]
> Installing OneTrainer requires Python >=3.10 and <3.13.
> You can download Python at https://www.python.org/downloads/windows/.
> Then follow the below steps.

#### Automatic installation

1. Clone the repository `git clone https://github.com/Nerogar/OneTrainer.git`
2. Run:
    - Windows: Double click or execute `install.bat`
    - Linux and Mac: Execute `install.sh`

#### Manual installation

1. Clone the repository `git clone https://github.com/Nerogar/OneTrainer.git`
2. Navigate into the cloned directory `cd OneTrainer`
3. Set up a virtual environment `python -m venv venv`
4. Activate the new venv:
    - Windows: `venv\scripts\activate`
    - Linux and Mac: Depends on your shell, activate the venv accordingly
5. Install the requirements `pip install -r requirements.txt`

> [!Tip]
> Some Linux distributions are missing required packages for instance: On Ubuntu you must install `libGL`:
>
> ```bash
> sudo apt-get update
> sudo apt-get install libgl1
> ```
>
> Additionally it's been reported Alpine, Arch and Xubuntu Linux may be missing `tkinter`. Install it via `apk add py3-tk` for Alpine and `sudo pacman -S tk` for Arch.

## Updating

#### Automatic update

-   Run `update.bat` or `update.sh`

#### Manual update

1. Cd to folder containing the repo `cd OneTrainer`
2. Pull changes `git pull`
3. Activate the venv `venv/scripts/activate`
4. Re-install all requirements `pip install -r requirements.txt --force-reinstall`

## Usage

OneTrainer can be used in **two primary modes**: a graphical user interface (GUI) and a **command-line interface (CLI)** for finer control.

For a technically focused quick start, see the [Quick Start Guide](docs/QuickStartGuide.md) and for a broader overview, see the [Overview documentation](docs/Overview.md). Otherwise visit [our wiki!](https://github.com/Nerogar/OneTrainer)

### GUI Mode

#### Windows

-   To start the UI, navigate to the OneTrainer folder and double-click `start-ui.bat`

#### Unix-based systems

-   Execute `start-ui.sh` and the GUI will pop up.

### CLI Mode

If you need more control or a headless approach OT also supports the command-line interface. All commands **need** to be run inside the active venv created during installation.

All functionality is split into different scripts located in the `scripts` directory. This currently includes:

-   `train.py` The central training script
-   `train_ui.py` A UI for training
-   `caption_ui.py` A UI for manual or automatic captioning and mask creation for masked training
-   `convert_model_ui.py` A UI for model conversions
-   `convert_model.py` A utility to convert between different model formats
-   `sample.py` A utility to sample any model
-   `create_train_files.py` A utility to create files needed when training only from the CLI
-   `generate_captions.py` A utility to automatically create captions for your dataset
-   `generate_masks.py` A utility to automatically create masks for your dataset
-   `calculate_loss.py` A utility to calculate the training loss of every image in your dataset

To learn more about the different parameters, execute `<script-name> -h`. For example `python scripts\train.py -h`

If you are on Mac or Linux, you can also read [the launch script documentation](LAUNCH-SCRIPTS.md) for detailed information about how to run OneTrainer and its various scripts on your system.

## Troubleshooting

For general troubleshooting or questions, ask in [Discussions](https://github.com/Nerogar/OneTrainer/discussions), check the [Wiki](https://github.com/Nerogar/OneTrainer/wiki) or join our [Discord](https://discord.gg/KwgcQd5scF).

If you encounter a reproducible error you first must run update.bat or update.sh and confirm the issue is still able to be reproduced. Then export anonymized debug information to help us solve an issue you are facing and upload it as part of your Github Issues submission.

-   On Windows double click `export_debug.bat`
-   On Unix-based systems execute `./run-cmd.sh generate_debug_report`

These will both create a `debug_report.log`.

> [!WARNING]
> We require this file for GitHub issues going forward. Failure to provide it or not manually providing the necessary info will lead to the issue being closed in most circumstances

## Contributing

Contributions are always welcome in any form. You can open issues, participate in discussions, or even open pull
requests for new or improved functionality. You can find more information about contributing [here](docs/Contributing.md).

Before you start looking at the code, I recommend reading about the project structure [here](docs/ProjectStructure.md).
For in depth discussions, you should consider joining the [Discord](https://discord.gg/KwgcQd5scF) server.

You also **NEED** to **install the required developer dependencies** for your current user and enable the Git commit hooks, via the following commands (works on all platforms; Windows, Linux and Mac):

> [!IMPORTANT]
> Be sure to run those commands _without activating your venv or Conda environment_, since [pre-commit](https://pre-commit.com/) is supposed to be installed outside any environment.

```sh
cd OneTrainer
pip install -r requirements-dev.txt
pre-commit install
```

Now all of your commits will automatically be verified for common errors and code style issues, so that code reviewers can focus on the architecture of your changes without wasting time on style/formatting issues, thus greatly improving the chances that your pull request will be accepted quickly and effortlessly.

## Related Projects

-   **[MGDS](https://github.com/Nerogar/mgds)**: A custom dataset implementation for Pytorch that is built around the idea of a node based graph.
-   **[Stability Matrix](https://github.com/LykosAI/StabilityMatrix)**: A swiss-army knife installer which wraps and installs a broad range of diffusion software packages including OneTrainer
-   **[Visions of Chaos](https://softology.pro/voc.htm)**: A collection of machine learning tools that also includes OneTrainer.
-   **[StableTuner](https://github.com/devilismyfriend/StableTuner)**: A now defunct (archived) training application for Stable Diffusion. OneTrainer takes a lot of inspiration from StableTuner and wouldn't exist without it.
