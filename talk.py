import torch
from transformers import AutoTokenizer
from model.gpt import GPT
import yaml
# rom eval.eval import Pretrain_Eval

rank = 0
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

# model defination
model = GPT(v_size = tokenizer.vocab_size + 4,
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

# load state for continue training
checkpoint = torch.load(f"{config['path']['save']}/latest_sft.pt", "cpu")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"checkpoint loaded from {config['path']['load']}")

model.eval()
roles, cur_role = ['user', 'assistant'], 0
role_start_toks, role_end_toks = [tokenizer.user_start_token_id, tokenizer.assistant_start_token_id], [tokenizer.user_end_token_id, tokenizer.assistant_end_token_id]
_ = model(torch.tensor([[tokenizer.eos_token_id]]).to(f'cuda:{rank}'))
while True:
    print("human: ")
    user_input = input()
    if user_input.startswith(':new topic'):
        model.clear_kv_cache()
        assert cur_role == 0
        _ = model(torch.tensor([[tokenizer.eos_token_id]]).to(f'cuda:{rank}'))
        continue
    tokens = []
    tokens.append(role_start_toks[cur_role])
    tokens += (tokenizer(user_input, truncation=False, max_length=None)['input_ids'])
    tokens.append(role_end_toks[cur_role])
    # input into model
    for token in tokens:
        token_gpu = torch.tensor([[token]]).to(f'cuda:{rank}')
        _ = model(token_gpu)
    # machine start to speak
    cur_role = 1 - cur_role # switch role
    print("machine: ")
    assert cur_role == 1
    tokens = [role_start_toks[cur_role]]
    while tokens[-1] != role_end_toks[cur_role]:
        token_gpu = torch.tensor([[tokens[-1]]]).to(f'cuda:{rank}')
        next_token_logits = model(token_gpu)[0,-1,:] # / 0.8
        k = 256 # top k sample
        topk_logits, topk_indices = torch.topk(next_token_logits, k)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        next_token = topk_indices[torch.multinomial(topk_probs, 1)].cpu().item()
        print(tokenizer.decode(next_token, skip_special_tokens=True), end="")
        tokens.append(next_token)
    token_gpu = torch.tensor([[tokens[-1]]]).to(f'cuda:{rank}')
    _ = model(token_gpu)[0,-1,:] # / 0.8
    # print(tokenizer.decode(tokens, skip_special_tokens=True))
    print("\n", end="")
    cur_role = 1 - cur_role # switch role
