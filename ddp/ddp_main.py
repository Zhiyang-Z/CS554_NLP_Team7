import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from ddp.ddp_trainer import Pre_Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataloaders.fineweb import PretrainDataset
from model.gpt import GPT
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.distributed.optim import ZeroRedundancyOptimizer
import yaml
import math

def ddp_main(rank: int, world_size: int):
    print("ddp setup...")
    ddp_setup(rank, world_size)
    print("ddp setup done.")
    # configure
    with open("pretrain_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"The vocabulary size is {tokenizer.vocab_size}, special tokens: {tokenizer.all_special_tokens}")
    dataset = PretrainDataset(dataset_name=config['dataset']['name'], subname=config['dataset']['subname'],
                              tokenizer=tokenizer, pretrain_len=config['pretraining']['pretrain_length']+1) # +1 for target shift
    dataloader = DataLoader(dataset,
                            batch_size=config['pretraining']['batch_size_per_gpu'],
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=1, # For IterableDataset, it's useless to launch multiple workers???
                            drop_last=True,
                            prefetch_factor=2)
    # # test dataloader speed
    # for i, data in tqdm(enumerate(dataloader)):
    #     pass
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
    # lr scheduler
    # Warmup (LR: 0 → base LR)
    scheduler_warmup = LinearLR(optimizer, start_factor=1.0e-6, end_factor=1, total_iters=config['pretraining']['warmup_iters'])
    # Cosine decay after warmup
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=config['pretraining']['lr_decay_iters'], eta_min=config['pretraining']['min_learning_rate'])
    # Constant LR after decay
    scheduler_constant = ConstantLR(optimizer, factor=config['pretraining']['min_learning_rate'] / config['pretraining']['learning_rate'], total_iters=1e9) # keep constant forever
    # Combine them sequentially
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay, scheduler_constant],
                             milestones=[config['pretraining']['warmup_iters'], config['pretraining']['warmup_iters'] + config['pretraining']['lr_decay_iters']])
    
    # load state for continue training
    # if config['path']['load'] is not None:
    #     state_dict = torch.load(config['path']['load'], "cpu")['model_state_dict']
    #     model.load_state_dict(state_dict)
    #     print(f"model loaded from {config['path']['load']}")
    # train
    grad_accum_steps = math.ceil(config['pretraining']['batch_size'] / (world_size*config['pretraining']['batch_size_per_gpu']*config['pretraining']['pretrain_length']))
    print(f'grad accum steps: {grad_accum_steps}')
    pre_trainer = Pre_Trainer(dataloader, model, optimizer, scheduler, grad_accum_steps, config['path']['save'])
    print('training start...')
    pre_trainer.train()
    # print('test start...')
    # pre_trainer.test()

    ddp_cleanup()