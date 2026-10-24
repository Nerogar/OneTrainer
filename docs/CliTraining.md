# Training from CLI

All training functionality is available through the CLI command `python scripts/train.py`. The training configuration is
stored in a `.json` file that is passed to this script.

Some options require specifying paths to files with a specific
layout. These files can be created using the create_train_files.py script. You can call the script like
this `python scripts/create_train_files.py -h`.

To simplify the creation of the training config, you can export your settings from the UI by using the export button.
This will create a single file that contains every setting.
