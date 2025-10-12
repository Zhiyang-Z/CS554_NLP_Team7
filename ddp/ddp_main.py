import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from ddp.ddp_trainer import Pre_Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataloaders.fineweb import PretrainDataset
from model.gpt import GPT
from torch.distributed.optim import ZeroRedundancyOptimizer
import yaml

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
                              tokenizer=tokenizer, pretrain_len=config['pretraining']['pretrain_length'])
    dataloader = DataLoader(dataset,
                            batch_size=config['pretraining']['batch_size_per_gpu'],
                            pin_memory=True,
                            shuffle=False, # must be False when use DDP
                            num_workers=2,
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

    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # optimizer
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=config['pretraining']['learning_rate'])
    # load state for continue training
    # if config['path']['load'] is not None:
    #     model.load_state_dict(config['path']['load'])
    # train
    grad_accum_steps = config['pretraining']['batch_size'] // (world_size*config['pretraining']['batch_size_per_gpu']*config['pretraining']['pretrain_length'])
    print(f'grad accum steps: {grad_accum_steps}')
    pre_trainer = Pre_Trainer(dataloader, model, optimizer, grad_accum_steps)
    print('training start...')
    pre_trainer.train()

    ddp_cleanup()