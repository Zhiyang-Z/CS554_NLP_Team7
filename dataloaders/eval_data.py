from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Dataset for testing choice question.
class EvalDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # each time, return context + 4 choices.
        question = self.dataset[idx]
        context, endings, label = question["ctx"]+" ", question["endings"], question["label"]
        choices, max_len = [], -1
        context_tok = self.tokenizer(context, truncation=False, max_length=None)['input_ids']
        assert context_tok[0] != self.eot_id
        context_tok.insert(0, self.eot_id) # add eot at the beginning
        for ending in endings:
            ending_tok = self.tokenizer(ending, truncation=False, max_length=None)['input_ids']
            input_tok = context_tok + ending_tok
            if len(input_tok) > max_len: max_len = len(input_tok)
            choices.append(input_tok)
        assert len(choices) == 4 # only A, B, C, D four choices
        # padding
        for i in range(4):
            l = len(choices[i])
            if l < max_len:
                choices[i] = choices[i] + ([-100]*(max_len - l))
        
        return np.array(choices), np.array([len(context_tok)]), np.array([int(label)])
    
if __name__ == '__main__':
    dataset = load_dataset("hellaswag", split="validation")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eval_dataset = EvalDataset(dataset, tokenizer)
    for item in eval_dataset:
        print(item[0].shape, item[1].shape)