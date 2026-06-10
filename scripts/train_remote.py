from util.import_util import script_imports

script_imports()

import json
import os
import pickle
import threading
import traceback
from contextlib import suppress

from modules.util import create
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig


def write_request(filename,name, *params):
    try:
        with suppress(FileNotFoundError):
            os.rename(filename,filename+'.write')
        with open(filename+'.write', 'ab') as f:
            pickle.dump(name,f)
            pickle.dump(params,f)
        os.rename(filename+'.write',filename)
    except Exception:
        #TrainCallbacks is suppressing all exceptions; at least print them:
        traceback.print_exc()
        raise

def close_pipe(filename):
    with open(filename, 'wb'): #send EOF by closing
        os.remove(filename)



def command_thread_function(commands: TrainCommands,filename : str,stop_event):
    while not stop_event.is_set():
        try:
            with open(filename, 'rb') as f:
                remote_commands=pickle.load(f)
        except FileNotFoundError:
            break
        except EOFError:
            continue

        commands.merge(remote_commands)


def main():
    args = TrainArgs.parse_args()
    if args.callback_path:
        callbacks = TrainCallbacks(
            on_update_train_progress=lambda *fargs:write_request(args.callback_path,"on_update_train_progress",*fargs),
            on_update_status=lambda *fargs:write_request(args.callback_path,"on_update_status",*fargs),
            on_sample_default=lambda *fargs:write_request(args.callback_path,"on_sample_default",*fargs),
            on_update_sample_default_progress=lambda *fargs:write_request(args.callback_path,"on_update_sample_default_progress",*fargs),
            on_sample_custom=lambda *fargs:write_request(args.callback_path,"on_sample_custom",*fargs),
            on_update_sample_custom_progress=lambda *fargs:write_request(args.callback_path,"on_update_sample_custom_progress",*fargs),
        )
    else:
        callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))
    train_config.cloud.enabled=False

    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict=json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    trainer = create.create_trainer(train_config, callbacks, commands)

    if args.command_path:
        stop_event=threading.Event()
        command_thread = threading.Thread(target=command_thread_function,args=(commands,args.command_path,stop_event))
        command_thread.start()

    try:
        trainer.start()
        trainer.train()

    finally:
        if args.command_path:
            stop_event.set()
            close_pipe(args.command_path)
            command_thread.join()

        trainer.end()



if __name__ == '__main__':
    main()
