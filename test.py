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
from eval.eval import Pretrain_Eval

rank = 0
# configure
with open("pretrain_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"The vocabulary size is {tokenizer.vocab_size}, special tokens: {tokenizer.all_special_tokens}")
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

# load state for continue training
checkpoint = torch.load(config['path']['load'], "cpu")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"checkpoint loaded from {config['path']['load']}")

evaluator = Pretrain_Eval(device=f'cuda:{rank}')
hellaswag_acc = evaluator.eval("mmlu", tokenizer, model)
print(f"HellaSwag Accuracy: {hellaswag_acc}")
