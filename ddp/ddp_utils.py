import os
import numpy as np
import torch
import torch.distributed as dist
import random

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12360"
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl",
                            rank=rank,
                            world_size=world_size,)
    # Set random seed
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    # enable fp32
    torch.set_float32_matmul_precision("high")
    
    # torch._dynamo.config.optimize_ddp = False

def ddp_cleanup():
    dist.destroy_process_group()