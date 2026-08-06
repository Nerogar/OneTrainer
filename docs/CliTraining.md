# Training from CLI

All training functionality is available through the CLI command `python scripts/train.py`. The training configuration is
stored in a `.json` file that is passed to this script.

Some options require specifying paths to files with a specific
layout. These files can be created using the create_train_files.py script. You can call the script like
this `python scripts/create_train_files.py -h`.

To simplify the creation of the training config, you can export your settings from the UI by using the export button.
This will create a single file that contains every setting.

### REST server

There is also a simple REST implementation that lets a client on the same machine start and stop training, request
samples and backups, and read the progress of a run. It needs the two extra packages
in `requirements-api.txt`, then:

`python scripts/train_rest_server.py --port 7800`

Note this deliberately binds to `127.0.0.1` only, so that no other machine can reach it and remotely initiate
training. This opens up opportunities for other web clients running on the machine (like ComfyUI) to start training
from their own software, and for automation tools.

Every endpoint is one call onto the same `TrainCallbacks` / `TrainCommands` pair the UI uses, so it drives local,
multi-GPU and cloud runs without any backend-specific code. `GET /docs` on the running server lists the endpoints.
