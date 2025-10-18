import torch
import torch.multiprocessing as mp
from ddp.ddp_main import ddp_main
import os
import random
import numpy as np
import argparse

# os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

def get_args():
    parser = argparse.ArgumentParser(description="Pretrain LLM Parameters")

    parser.add_argument("--pretrain", dest="pretrain", action="store_true", help="Do pretrain.")
    parser.add_argument("--sft", dest="sft", action="store_true", help="Do supervised finetuning.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    args = get_args()

    world_size = torch.cuda.device_count()
    print(f"detected {world_size} GPUs.")
    print("spawn processes...")

    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size)