from util.import_util import script_imports

script_imports()

import json

from modules.util import create
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig


def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()

    if args.preset_path is not None:
        with open(args.preset_path, "r") as f:
            train_config.from_dict(json.load(f), migrate=False)

    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f), migrate=args.preset_path is None)

    for config_value in args.config_values or []:
        key, _, value = config_value.partition("=")
        *parent_keys, leaf_key = key.split(".")
        target = train_config
        for parent_key in parent_keys:
            target = getattr(target, parent_key)
        if target.types[leaf_key] is bool:
            value = value.lower() in ("true", "1", "yes")
        target.from_dict({leaf_key: value}, migrate=False)

    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict=json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    trainer = create.create_trainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()


if __name__ == '__main__':
    main()
