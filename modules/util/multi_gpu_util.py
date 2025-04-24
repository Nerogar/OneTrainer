from collections import deque

from modules.util.commands.TrainCommands import TrainCommands

import torch


def is_enabled() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def rank() -> int:
    return torch.distributed.get_rank() if is_enabled() else 0

def is_master() -> bool:
    return rank() == 0

def world_size() -> int:
    return torch.distributed.get_world_size() if is_enabled() else 1

#execute code sequentially in all ranks, using a for loop:
def sequential(enabled: bool = True):
    if enabled:
        for current in range(world_size()):
            if current == rank():
                yield()
            if is_enabled():
                torch.distributed.barrier()
    else:
        yield()

#execute code first only on rank 0, then on all other ranks in parallel:
def master_first(enabled: bool = True):
    if enabled:
        for current in [True, False]:
            if current == is_master():
                yield()
            if is_enabled():
                torch.distributed.barrier()
    else:
        yield()

def distributed_enumerate(iterable, distribute: bool=True):
    if distribute:
        for i, x in enumerate(iterable):
            if i % world_size() == rank():
                yield i, x
    elif is_master():
        for i, x in enumerate(iterable):
            yield i, x


def reduce_tensor_mean(tensor):
    if is_enabled():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= world_size()

async_deque = deque()
in_transfer = 0


def reduce_grads_mean(params: list[torch.Tensor], after_reduce=None, async_op: bool=False, max_buffer: int=0):
    assert not async_op or max_buffer > 0
    if not is_enabled() and after_reduce is None:
        return
    for param in params:
        if param.requires_grad and param.grad is not None:
            if is_enabled():
                if async_op:
                    global async_deque
                    global in_transfer
                    size = param.grad.numel() * param.grad.element_size()
                    complete_previous_async_ops(next_size=size, max_buffer=max_buffer)
                    work = torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, async_op=True)
                    async_deque.append((work, param, after_reduce))
                    in_transfer += size
                else:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, async_op=False)
                    param.grad /= world_size() #TODO stochastic rounding?
                    if after_reduce is not None:
                        after_reduce(param)
            elif after_reduce is not None:
                after_reduce(param)

def complete_previous_async_ops(next_size: int=0, max_buffer: int=0):
    global async_deque
    global in_transfer
    while async_deque and (
        in_transfer + next_size > max_buffer
        or async_deque[0][0].is_completed()
    ):
        work, param, after_reduce = async_deque.popleft()
        work.wait()
        in_transfer -= param.grad.numel() * param.grad.element_size()
        param.grad /= world_size() #TODO stochastic rounding?
        if after_reduce is not None:
            after_reduce(param)

def finish_async():
    complete_previous_async_ops(max_buffer=0)



def parameter_divergence(params: list[torch.Tensor]):
    if not is_enabled():
        return 0.0
    diff = 0.0

    #TODO this can take an unnecessary amount of VRAM during full finetuning with many GPUs and large individual tensors.
    #use send()/recv() instead or chunk up the tensors
    for param in params:
        #some parameters could be offloaded to GPU during full-finetuning. We have to ignore those:
        if param.requires_grad and param.is_cuda:
            param_list = [torch.zeros_like(param) for _ in range(world_size())] if is_master() else None
            torch.distributed.gather(param, param_list, dst=0)
            if is_master():
                for r in range(1, world_size()):
                    diff += torch.sum(torch.abs(param_list[0] - param_list[r]))

    return diff if is_master() else None



global_commands = None

def set_global_commands(commands: TrainCommands):
    #passing self.commands directly to the GenericTrainer of the main process can result in unsynced behaviour:
    #for example, the main process could know earlier of a stop command than all other processes, because it is set directly by the UI.
    #to avoid this, global_commands is used by the UI, and merged into the commands of each process in sync_commands()

    global global_commands
    global_commands = commands


def sync_commands(commands: TrainCommands):
    if is_enabled():
        object_list= [global_commands] if is_master() else [None]
        torch.distributed.broadcast_object_list(object_list, src=0)
        commands.merge(object_list[0])
