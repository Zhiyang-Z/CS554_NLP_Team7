import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from ddp.ddp_trainer import Pre_Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataloaders.smoltalk import SFTDataset
from model.gpt import GPT
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data.distributed import DistributedSampler
import yaml
import math
from tqdm import tqdm

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
    tokenizer.add_special_tokens({"pad_token": "<|pad|>",
                                  "additional_special_tokens": new_special_tokens})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    tokenizer.user_start_token_id = tokenizer.convert_tokens_to_ids("<|user_start|>")
    tokenizer.user_end_token_id = tokenizer.convert_tokens_to_ids("<|user_end|>")
    tokenizer.assistant_start_token_id = tokenizer.convert_tokens_to_ids("<|assistant_start|>")
    tokenizer.assistant_end_token_id = tokenizer.convert_tokens_to_ids("<|assistant_end|>")
    print(f"The enlarged vocabulary size is {len(tokenizer)}, special token_id: \
          [{tokenizer.eos_token_id}, {tokenizer.pad_token_id}, {tokenizer.user_start_token_id}, {tokenizer.user_end_token_id}, {tokenizer.assistant_start_token_id}, {tokenizer.assistant_end_token_id}]")
    # dataloader
    dataset = SFTDataset(dataset_name=config['dataset']['name'], subname=config['dataset']['subname'],
                              tokenizer=tokenizer, pad_length=2048)
    dataloader = DataLoader(dataset,
                            batch_size=config['pretraining']['batch_size_per_gpu'],
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=4, # For IterableDataset, it's useless to launch multiple workers???
                            drop_last=True,
                            prefetch_factor=2,
                            sampler=DistributedSampler(dataset, shuffle=True, seed=0))
    # test dataloader speed
    # lens = []
    # for i, data in tqdm(enumerate(dataloader)):
    #     pass
    # print(max(lens))
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
    model.enlarge_voc(len(new_special_tokens)+1, tokenizer.pad_token_id) # +1 for pad token
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