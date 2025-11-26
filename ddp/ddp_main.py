import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from ddp.ddp_trainer import SFT_Trainer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import ConcatDataset
from dataloaders.ultrachat import SFTDataset
from model.gpt import GPT
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data.distributed import DistributedSampler
import yaml
import math
from tqdm import tqdm
import numpy as np

def pad_and_truncate(batch):
    max_len = 4096
    batch_max_len = -1

    for item in batch:
        assert item[0].shape == item[1].shape
        if item[0].shape[0] > batch_max_len: batch_max_len = item[0].shape[0]
    batch_align_len = min(batch_max_len, max_len)
    batch_padded = []
    for item in batch:
        assert item[0].shape == item[1].shape
        if batch_align_len > item[0].shape[0]: # pad to fixed length.
            padding_len = batch_align_len - item[0].shape[0]
            input_padding, label_padding = np.array([0] * padding_len), np.array([-100] * padding_len)
            batch_padded.append((np.append(item[0], input_padding)[:-1], np.append(item[1], label_padding)[1:])) # shifted in dataloader
        else: # truncate
            batch_padded.append((item[0][0:batch_align_len][:-1], item[1][0:batch_align_len][1:])) # shifted in dataloader

    return default_collate(batch_padded)

def ddp_main(rank: int, world_size: int, resume: bool):
    print("ddp setup...")
    ddp_setup(rank, world_size)
    print("ddp setup done.")
    # configure
    with open("sft_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"The original vocabulary size is {len(tokenizer)}, special tokens/id: {tokenizer.all_special_tokens}/{tokenizer.eos_token_id}")
    # enlarge tokens for instruction finetuning
    new_special_tokens = ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    tokenizer.user_start_token_id = tokenizer.convert_tokens_to_ids("<|user_start|>")
    tokenizer.user_end_token_id = tokenizer.convert_tokens_to_ids("<|user_end|>")
    tokenizer.assistant_start_token_id = tokenizer.convert_tokens_to_ids("<|assistant_start|>")
    tokenizer.assistant_end_token_id = tokenizer.convert_tokens_to_ids("<|assistant_end|>")
    print(f"The enlarged vocabulary size is {len(tokenizer)}, special token_id: \
          [{tokenizer.eos_token_id}, {tokenizer.user_start_token_id}, {tokenizer.user_end_token_id}, {tokenizer.assistant_start_token_id}, {tokenizer.assistant_end_token_id}]")
    # dataset, combine datasets together for different abilities.
    sft_datasets_list = []
    # for talking ability
    sft_datasets_list.append(load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'everyday-conversations', split="train"))
    sft_datasets_list.append(load_dataset("lmsys/lmsys-chat-1m", split="train").filter(lambda x: x["language"] == "English"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'smol-magpie-ultra', split="train"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'smol-constraints', split="train"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'smol-rewrite', split="train"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'smol-summarize', split="train"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'explore-instruct-rewriting', split="train"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'openhermes-100k', split="train"))
    sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'systemchats-30k', split="train"))
    sft_datasets_list.append(load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft"))
    # For math ability
    # sft_datasets_list.append(load_dataset("openai/gsm8k", "main", split="train"))
    # sft_datasets_list.append(load_dataset("qintongli/GSM-Plus-v0", "default", split="test"))
    # sft_datasets_list.append(load_dataset("HuggingFaceTB/smoltalk", 'metamathqa-50k', split="train"))
    # sft_datasets_list.append(load_dataset("tiedong/goat", split="train"))

    sft_dataset = ConcatDataset(sft_datasets_list)
    dataset = SFTDataset(sft_dataset, tokenizer=tokenizer)
    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=config['sft']['batch_size_per_gpu'],
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=2,
                            sampler=DistributedSampler(dataset, shuffle=True, seed=7),
                            collate_fn=pad_and_truncate)
    # test dataloader speed
    # lens = []
    # total_tok, trained_tok = 0, 0
    # for x, y in tqdm(dataloader):
    #     total_tok += y.shape[0] * y.shape[1]
    #     trained_tok += (y != -100).sum()
    # # print(f"max len is: {np.sort(np.array(dataset.max_len))[-5000:-1]}")
    # print(f"total tokens: {total_tok}, trained tokens: {trained_tok}")
    # exit()
    # model defination
    model = GPT(v_size = len(tokenizer),
                train_length = config['sft']['pretrain_length'],
                n_dim = config['model']['n_dim'],
                n_layer = config['model']['n_layer'],
                n_head = config['model']['n_head'],
                dim_head = config['model']['dim_head'],
                ff_ratio = config['model']['ff_ratio'],
                dropout_rate = config['model']['dropout_rate'],
                device = f'cuda:{rank}',
                ex_ratio = config['model']['ex_ratio'])

    model = model.to(f'cuda:{rank}')
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # optimizer
    optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                        optimizer_class=torch.optim.AdamW,
                                        lr=config['sft']['learning_rate'],
                                        betas=(0.9, 0.95),
                                        weight_decay=config['sft']['weight_decay'])
    # lr scheduler
    # Warmup (LR: 0 → base LR)
    # scheduler_warmup = LinearLR(optimizer, start_factor=1.0e-6, end_factor=1, total_iters=config['sft']['warmup_iters'])
    # # Cosine decay after warmup
    # scheduler_decay = CosineAnnealingLR(optimizer, T_max=config['sft']['lr_decay_iters'], eta_min=config['sft']['min_learning_rate'])
    # # Constant LR after decay
    # scheduler_constant = ConstantLR(optimizer, factor=config['sft']['min_learning_rate'] / config['sft']['learning_rate'], total_iters=1e9) # keep constant forever
    # # Combine them sequentially
    # scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay, scheduler_constant],
    #                          milestones=[config['sft']['warmup_iters'], config['sft']['warmup_iters'] + config['sft']['lr_decay_iters']])
    
    # load state for continue SFT
    checkpoint = torch.load(config['path']['load'], "cpu")
    model.load_state_dict({k: v for k, v in checkpoint['model_state_dict'].items() if not (k.startswith("rope_") or k == "embedding.weight" or k == 'out.weight')}, strict=False) # strict=False to ignore the rope param.
    model.embedding.weight.data[0:tokenizer.vocab_size] = checkpoint['model_state_dict']["embedding.weight"]
    model.out.weight.data[0:tokenizer.vocab_size] = checkpoint['model_state_dict']["out.weight"]
    if not torch.equal(model.embedding.weight.data, model.out.weight.data):
        print("weight tying failed.")
    print(f"checkpoint loaded from {config['path']['load']}")
    # train
    grad_accum_steps = math.ceil(config['sft']['batch_size'] / (world_size*config['sft']['batch_size_per_gpu']))
    print(f'grad accum steps: {grad_accum_steps}')
    pre_trainer = SFT_Trainer(dataloader, model, optimizer, None, grad_accum_steps, config, False)
    print('training start...')
    pre_trainer.train()
    # print('test start...')
    # pre_trainer.test()

    ddp_cleanup()