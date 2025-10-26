import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from ddp.ddp_trainer import Pre_Trainer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer
from dataloaders.ultrachat import SFTDataset
from model.gpt import GPT
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data.distributed import DistributedSampler
import yaml
import math
from tqdm import tqdm
import numpy as np

def drop_long(batch):
    batch_max_len = -1
    idx, long_idx = -1, []
    for item in batch:
        assert item[0].shape == item[1].shape
        idx += 1
        if item[0].shape[0] > 2048:
            long_idx.append(idx)
            continue
        if item[0].shape[0] > batch_max_len: batch_max_len = item[0].shape[0]
    batch_align_len = 2048 + 1
    batch_extend = []
    idx = -1
    for item in batch:
        idx += 1
        if idx in long_idx:
            input_padding, label_padding = np.array([0] * batch_align_len), np.array([-100] * batch_align_len)
            batch_extend.append((input_padding[:-1], label_padding[1:])) # shifted in dataloader
            continue
        assert batch_align_len > item[0].shape[0] and item[0].shape == item[1].shape
        input_padding, label_padding = np.array([0] * (batch_align_len - item[0].shape[0])), np.array([-100] * (batch_align_len - item[0].shape[0]))
        batch_extend.append((np.append(item[0], input_padding)[:-1], np.append(item[1], label_padding)[1:])) # shifted in dataloader
    return default_collate(batch_extend)

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
    # dataloader
    dataset = SFTDataset(dataset_name=config['dataset']['name'], subname=config['dataset']['subname'],
                              tokenizer=tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=config['pretraining']['batch_size_per_gpu'],
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=2,
                            sampler=DistributedSampler(dataset, shuffle=True, seed=0),
                            collate_fn=drop_long)
    # # test dataloader speed
    # lens = []
    # for i, data in tqdm(enumerate(dataloader)):
    #     print(data)
    # exit()
    # model defination
    model = GPT(v_size = tokenizer.vocab_size,
                train_length = config['pretraining']['pretrain_length'],
                n_dim = config['model']['n_dim'],
                n_layer = config['model']['n_layer'],
                n_head = config['model']['n_head'],
                dim_head = config['model']['dim_head'],
                ff_ratio = config['model']['ff_ratio'],
                ff_dropout = config['model']['ff_dropout'],
                device = f'cuda:{rank}',
                ex_ratio = config['model']['ex_ratio'])

    model = model.to(f'cuda:{rank}')
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # optimizer
    optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                        optimizer_class=torch.optim.AdamW,
                                        lr=config['pretraining']['learning_rate'],
                                        betas=(0.9, 0.95),
                                        weight_decay=config['pretraining']['weight_decay'])
    # # lr scheduler
    # # Warmup (LR: 0 → base LR)
    # scheduler_warmup = LinearLR(optimizer, start_factor=1.0e-6, end_factor=1, total_iters=config['pretraining']['warmup_iters'])
    # # Cosine decay after warmup
    # scheduler_decay = CosineAnnealingLR(optimizer, T_max=config['pretraining']['lr_decay_iters'], eta_min=config['pretraining']['min_learning_rate'])
    # # Constant LR after decay
    # scheduler_constant = ConstantLR(optimizer, factor=config['pretraining']['min_learning_rate'] / config['pretraining']['learning_rate'], total_iters=1e9) # keep constant forever
    # # Combine them sequentially
    # scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay, scheduler_constant],
    #                          milestones=[config['pretraining']['warmup_iters'], config['pretraining']['warmup_iters'] + config['pretraining']['lr_decay_iters']])
    
    # load state for continue SFT
    checkpoint = torch.load(config['path']['load'], "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"checkpoint loaded from {config['path']['load']}")
    # after loading parameters, we can enlarge vocabulary now.
    model.enlarge_voc(len(new_special_tokens), -100)
    print(f"vocabulary enlarged.")
    # train
    grad_accum_steps = math.ceil(config['pretraining']['batch_size'] / (world_size*config['pretraining']['batch_size_per_gpu']))
    print(f'grad accum steps: {grad_accum_steps}')
    pre_trainer = Pre_Trainer(dataloader, model, optimizer, None, grad_accum_steps, config, False)
    print('training start...')
    pre_trainer.train()
    # print('test start...')
    # pre_trainer.test()

    ddp_cleanup()