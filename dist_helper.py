import os
import sys
from contextlib import suppress
from dataclasses import dataclass

import torch
from torch import distributed as dist
from torch import nn


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def is_dist_initialized() -> bool:
    try:
        return dist.is_initialized()
    except:
        return False


devnull = open(os.devnull, 'w')


class _RedirectStdStreams(object):
    def __init__(self, stdout=devnull, stderr=devnull, stdin=devnull):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._stdin = stdin or sys.stdin

    def __enter__(self):
        self.old_stdout, self.old_stderr, self.old_stdin = sys.stdout, sys.stderr, sys.stdin
        self.old_stdout.flush()
        self.old_stderr.flush()
        self.old_stdin.flush()
        sys.stdout, sys.stderr, sys.stdin = self._stdout, self._stderr, self._stdin

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        self._stdin.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        sys.stdin = self.old_stdin


@dataclass(frozen=True)
class DistributedEnv(metaclass=_Singleton):
    rank: int = int(os.environ.get("RANK", 0))
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))

    @property
    def on_master(self) -> bool:
        return self.local_rank == 0

    @property
    def has_multiple_process(self) -> bool:
        """this flag can change during the code"""
        return self.world_size > 1

    @property
    def is_initialized(self) -> bool:
        return is_dist_initialized()

    def barrier(self) -> None:
        if self.is_initialized:
            dist.barrier()

    def ipdb_set_trace(self):
        if not self.has_multiple_process:
            breakpoint()
        elif not self.is_initialized:
            # disable the input method for other processes
            context = _RedirectStdStreams() if not self.on_master else suppress()
            with context:
                if self.on_master:
                    breakpoint()
        else:
            """in a true distributed training process."""
            if self.on_master:
                breakpoint()
            dist.barrier()


def init_distributed_mode(distributed_env: DistributedEnv, init_method: str = None):
    torch.cuda.set_device(distributed_env.local_rank)
    dist.init_process_group(backend="nccl", init_method=init_method,
                            world_size=distributed_env.world_size, rank=distributed_env.rank)

    distributed_env.barrier()


def average_gradients(model: nn.Module):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
