# OneTrainer Launch Scripts


## Mac and Linux Systems

### The launch system consists of the following scripts:

- `install.sh`: Ensures that you have a valid Python runtime environment and installs all requirements if necessary.
- `update.sh`: Updates OneTrainer to the latest version and upgrades any outdated requirements in your Python runtime environment.
- `start-ui.sh`: Launches the main OneTrainer interface.
- `run-cmd.sh`: Executes a custom script (such as "train"), and supports providing command-line arguments. See the "running custom script commands" guide section for more details.


### All of the scripts accept the following *optional* environment variables to customize their behavior:

- `OT_CONDA_CMD`: Sets a custom Conda command or an absolute path to the binary (useful when it isn't in the user's `PATH`). If nothing is provided, we detect and use `CONDA_EXE` if available, which is a variable that's set by Conda itself and always points at the user's installed Conda binary.

- `OT_CONDA_ENV`: Sets the name of the Conda environment. Defaults to `onetrainer`.

- `OT_PYTHON_CMD`: Sets the Host's Python executable. It's used for creating the Python Venvs. Defaults to `python`.

- `OT_PYTHON_VENV`: Sets the name (or an absolute/relative path) of the Python Venv. If a name or relative path is used, it will be relative to the OneTrainer directory. Defaults to `.venv`.

- `OT_PREFER_VENV`: If set to `true`, Conda will be ignored even if it exists on the system, and Python Venv will be used instead. This ensures that people who use `pyenv` (to choose which Python version to run on the host) can easily set up their desired Python Venv environments. Defaults to `false`.

- `OT_CUDA_LOWMEM_MODE`: If set to `true`, it enables aggressive garbage collection in PyTorch to help with low-memory GPUs. Defaults to `false`.

- `OT_PLATFORM_REQUIREMENTS`: Allows you to override which platform-specific "requirements" file you want to install. Defaults to `detect`, which automatically detects whether you have an AMD or NVIDIA GPU. But people with multi-GPU systems can use this setting to force a specific GPU acceleration framework's requirements. Valid values are `requirements-rocm.txt` for AMD, `requirements-cuda.txt` for NVIDIA, and `requirements-default.txt` for non-AMD/NVIDIA systems.

- `OT_SCRIPT_DEBUG`: If set to `true`, it enables additional debug logging in the scripts. Defaults to `false`.


### Examples of how to use the custom environment variables:

- You can provide custom environment variables directly on the command line, as follows: `env OT_PREFER_VENV="true" OT_CUDA_LOWMEM_MODE="true" OT_PLATFORM_REQUIREMENTS="requirements-cuda.txt" ./start-ui.sh`.
- You can add them to your user's persistent environment variables, so that they are always active. The process varies depending on your operating system. On Linux, you can place them in `~/.config/environment.d/onetrainer.conf` (on all Systemd-based distros), which is a plaintext file with *one variable per line,* such as `OT_CUDA_LOWMEM_MODE="true"`. Beware that changes to `environment.d` requires a *complete system restart* to take effect (there is no command for reloading them live). To verify that your environment has been set persistently, you can then open a terminal window and run `printenv <variable name>` (such as `printenv OT_CUDA_LOWMEM_MODE`) to see if your custom values have taken effect.
- If you're launching OneTrainer from your own, custom scripts, then you can instead `export` the new values (which tells the shell to pass those environment variables onto child processes). For example, by having a line such as `export OT_CUDA_LOWMEM_MODE="true"` before your script calls `./OneTrainer/start-ui.sh`.
- If you're running OneTrainer inside a Docker/Podman container, you can instead use the [ENV](https://docs.docker.com/reference/dockerfile/#env) instruction in your `Dockerfile` / `Containerfile` to set the variables, such as `ENV OT_CUDA_LOWMEM_MODE="true"`.


### Running custom script commands:

- Always use `run-cmd.sh`, which automatically validates the chosen target script's name, configures the runtime environment correctly, and then runs the target script with your given command-line arguments.
- For example, to run the training CLI script, you would use `./run-cmd.sh train --config-path <path to your config>`.
- The names of all valid scripts can be seen in OneTrainer's `scripts/` directory.


### Creating your own launch scripts:

- If you want to automate various OneTrainer CLI tasks, then you should call `run-cmd.sh` from your own scripts (see previous guide section), since it's capable of running *any* OneTrainer command with your own command-line arguments.
- To run multiple tasks in the same scripts, you should perform separate calls to `run-cmd.sh`. Run it as many times as required for all the custom scripts and command-line arguments that you want to perform in your own script.
- It's highly recommended that you use `set -e` at the top of your own scripts (see `install.sh` for an example of that), since it tells Bash to exit your script if any of the OneTrainer commands fail. Otherwise your script will continue running even if a previous step has failed, which is usually not what you want!
