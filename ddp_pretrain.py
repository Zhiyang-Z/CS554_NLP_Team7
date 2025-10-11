import torch
import torch.multiprocessing as mp
from ddp.ddp_main import ddp_main
import os
# os.environ["WANDB_MODE"] = "disabled"

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"detected {world_size} GPUs.")
    print("spawn processes...")

    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size)