import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]) == 0
    return True


def _setup_for_distributed(is_master):
    import builtins as builtin

    builtin_print = builtin.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtin.print = print


def init_distributed_mode(config):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.gpu = int(os.environ["LOCAL_RANK"])
        config.dist_url = "env://"
    elif "SLURM_PROCID" in os.environ:
        config.rank = int(os.environ["SLURM_PROCID"])
        config.gpu = config.rank % torch.cuda.device_count()
        config.world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        config.dist_url = "env://"
    else:
        config.distributed = False
        config.rank = 0
        config.world_size = 1
        config.gpu = 0
        print("Not using distributed mode")
        return

    config.distributed = True
    torch.cuda.set_device(config.gpu)
    dist.init_process_group(
        backend="nccl",
        init_method=config.dist_url,
        world_size=config.world_size,
        rank=config.rank,
    )
    dist.barrier()
    _setup_for_distributed(config.rank == 0)
