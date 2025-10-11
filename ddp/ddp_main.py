import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from ddp.ddp_trainer import Pre_Trainer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer
from dataloaders.fineweb import PretrainDataset
from model.gpt import GPT
from torch.distributed.optim import ZeroRedundancyOptimizer

def ddp_main(rank: int, world_size: int):
    print("ddp setup...")
    ddp_setup(rank, world_size)
    print("ddp setup done.")

    # dataset
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42) # shuffle it, just a routine, we will handle shuffle for each iteration in training loop.
    # split by node
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"The vocabulary size is {tokenizer.vocab_size}, special tokens: {tokenizer.all_special_tokens}")
    dataset = PretrainDataset(dataset, tokenizer, pretrain_len=2048)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=2)
    # # test dataloader speed
    # for i, data in tqdm(enumerate(dataloader)):
    #     pass
    # model defination
    model = GPT(v_size = tokenizer.vocab_size,
                train_length = 2048,
                n_dim = 1280,
                n_layer = 36,
                n_head = 20,
                ff_ratio = 4.0,
                ff_dropout = 0.1,
                device = None,
                ex_ratio = 1.2)
    # model = GPT(v_size = tokenizer.vocab_size,
    #             train_length = 2048,
    #             n_dim = 768,
    #             n_layer = 1,
    #             n_head = 6,
    #             ff_ratio = 4.0,
    #             ff_dropout = 0.1,
    #             device = f'cuda:{rank}',
    #             ex_ratio = 1.2)
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # optimizer
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-4)
    # train
    pre_trainer = Pre_Trainer(dataloader, model, optimizer)
    print('training start...')
    pre_trainer.train()

    ddp_cleanup()