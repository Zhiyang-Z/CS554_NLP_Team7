import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from torch.utils.data import DataLoader
# from ddp.ddp_trainer import Trainer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer
from dataloaders.fineweb import PretrainDataset
from tqdm import tqdm

def ddp_main(rank: int, world_size: int):
    print("ddp setup...")
    ddp_setup(rank, world_size)
    print("ddp setup done.")

    # dataset
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    # First split by node
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = PretrainDataset(dataset, tokenizer, pretrain_len=1024)

    dataloader = DataLoader(dataset,
                            batch_size=256,
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=2,
    )

    for i, data in tqdm(enumerate(dataloader)):
        pass # test speed
    # model
    # model defination
    # model.train()
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of parameters: {total_params}")
    # optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # train
    # mark = f'{model_size}_' + ('2D' if lon_lat_emb else '1D')
    # trainer = Trainer(dataloader, model, optimizer, mark, data_path, city)
    # print('training start...')
    # trainer.train()

    ddp_cleanup()