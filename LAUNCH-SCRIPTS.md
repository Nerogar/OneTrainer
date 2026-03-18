# OneTrainer Launch Scripts


## Mac and Linux Systems

### The launch system consists of the following scripts:

- `install.sh`: Ensures that you have a valid Python runtime environment and installs all requirements if necessary.
- `update.sh`: Updates OneTrainer to the latest version and upgrades any outdated requirements in your Python runtime environment.
- `start-ui.sh`: Launches the main OneTrainer interface.
- `run-cmd.sh`: Executes a custom script (such as "train"), and supports providing command-line arguments. See the "running custom script commands" guide section for more details.


### All of the scripts accept the following *optional* environment variables to customize their behavior:

- `OT_CUDA_LOWMEM_MODE`: If set to `true`, it enables aggressive garbage collection in PyTorch to help with low-memory GPUs. Defaults to `false`.

- `OT_PLATFORM`: Allows you to override which platform you want to install for. Defaults to `detect`, which automatically detects whether you have an AMD or NVIDIA GPU, and falls back to CPU on failure. But people with multi-GPU systems can use this setting to force a specific GPU acceleration framework's requirements. Valid values are `rocm` for AMD, `cuda` for NVIDIA, and `default` for non-AMD/NVIDIA systems.

- `OT_SCRIPT_DEBUG`: If set to `true`, it enables additional debug logging in the scripts. Defaults to `false`.


### Examples of how to use the custom environment variables:

- You can provide custom environment variables directly on the command line, as follows: `env OT_PLATFORM="rocm" OT_CUDA_LOWMEM_MODE="true" ./start-ui.sh`.
- You can add them to your user's persistent environment variables, so that they are always active. The process varies depending on your operating system. On Linux, you can place them in `~/.config/environment.d/onetrainer.conf` (on all Systemd-based distros), which is a plaintext file with *one variable per line,* such as `OT_CUDA_LOWMEM_MODE="true"`. Beware that changes to `environment.d` requires a *complete system restart* to take effect (there is no command for reloading them live). To verify that your environment has been set persistently, you can then open a terminal window and run `printenv <variable name>` (such as `printenv OT_CUDA_LOWMEM_MODE`) to see if your custom values have taken effect.
- If you're launching OneTrainer from your own, custom scripts, then you can instead `export` the new values (which tells the shell to pass those environment variables onto child processes). For example, by having a line such as `export OT_CUDA_LOWMEM_MODE="true"` before your script calls `./OneTrainer/start-ui.sh`.
- If you're running OneTrainer inside a Docker/Podman container, you can instead use the [ENV](https://docs.docker.com/reference/dockerfile/#env) instruction in your `Dockerfile` / `Containerfile` to set the variables, such as `ENV OT_CUDA_LOWMEM_MODE="true"`.


### Running custom script commands:

- Always use `run-cmd.sh` when you want to execute any of OneTrainer's CLI tasks. It automatically validates the chosen target script's name, configures the runtime environment correctly, and then runs the target script with your given command-line arguments.
- For example, to run the training CLI script, you would use `./run-cmd.sh train --config-path <path to your config>`.
- The names of all valid scripts can be seen in OneTrainer's `scripts/` sub-directory.
- To learn more about the available command-line arguments for each script, you can execute them with the `-h` (help) argument: `./run-cmd.sh <script name> -h`. For example, if you want to learn more about the "train" script, you would run `./run-cmd.sh train -h`.


### Creating your own launch scripts and automating tasks:

- If you want to automate various OneTrainer CLI tasks, then you should call `run-cmd.sh` from your own scripts (see previous guide section), since it's capable of running *any* OneTrainer command with your own command-line arguments.
- To run multiple tasks in the same scripts, you should perform separate calls to `run-cmd.sh`. Run it as many times as required for all the custom scripts and command-line arguments that you want to perform in your own script.
- It's highly recommended that you use `set -e` at the top of your own scripts (see `install.sh` for an example of that), since it tells Bash to exit your script if any of the OneTrainer commands fail. Otherwise your script will continue running even if a previous step has failed, which is usually not what you want!
