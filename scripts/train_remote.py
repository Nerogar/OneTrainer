from util.import_util import script_imports

script_imports()

import json
import pickle
import os
import threading

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig


def write_request(filename,name, *params):
    try: os.rename(filename,filename+'.write')
    except FileNotFoundError: pass
    with open(filename+'.write', 'ab') as f:
        pickle.dump(name,f)
        pickle.dump(params,f)
    os.rename(filename+'.write',filename)
        
def close_pipe(filename):
    with open(filename, 'wb') as f: #send EOF by closing
        os.remove(filename)

        
    
def command_thread_function(commands: TrainCommands,filename : str,stop_event):
    while not stop_event.is_set():
        try:
            with open(filename, 'rb') as f:
                remote_commands=pickle.load(f)
        except FileNotFoundError: break
        except EOFError: continue
            
        if remote_commands.get_stop_command(): commands.stop()
        for entry in remote_commands.get_and_reset_sample_custom_commands():
            commands.sample_custom(entry)
        if remote_commands.get_and_reset_sample_default_command(): commands.sample_default()
        if remote_commands.get_and_reset_backup_command(): commands.backup()
        if remote_commands.get_and_reset_save_command(): commands.save()

        

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
    else: callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenericTrainer(train_config, callbacks, commands)

    if args.command_path:
        stop_event=threading.Event()
        command_thread = threading.Thread(target=command_thread_function,args=(commands,args.command_path,stop_event))
        command_thread.start()

    try:
        trainer.start()
        trainer.train()

    finally:
        stop_event.set()
        if args.command_path: close_pipe(args.command_path)

        command_thread.join()
        trainer.end()
        


if __name__ == '__main__':
    main()
