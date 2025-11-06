import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from model.gpt import GPT
import yaml
import math
from eval.eval_choice import Eval_Choice
from dataloaders.eval_data import EvalDataset
from eval.eval_choice import make_dataloader4eval_choice

rank = 1
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
            dropout_rate = config['model']['dropout_rate'],
            device = f'cuda:{rank}',
            ex_ratio = 1.1)

model = model.to(f'cuda:{rank}')

# load state for continue training
checkpoint = torch.load("/home/zhiyang/projects/CS554_NLP_Team7/saved_models/latest.pt", "cpu")
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
print(f"checkpoint loaded from {config['path']['load']}")

dataset = EvalDataset(load_dataset("hellaswag", split="validation"), tokenizer)
dataloader = make_dataloader4eval_choice(dataset, 8)
evaluator = Eval_Choice(dataloader, model, device=f'cuda:{rank}')
evaluator.eval()
# hellaswag_acc = evaluator.eval("mmlu", tokenizer, model)
# print(f"HellaSwag Accuracy: {hellaswag_acc}")