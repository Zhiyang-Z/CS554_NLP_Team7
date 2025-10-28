from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import copy
import re

class GSM8KDataset(Dataset):
    def __init__(self, dataset_name, subname, tokenizer, pad_length):
        self.dataset_name, self.subname = dataset_name, subname
        self.dataset = load_dataset(dataset_name, name=subname, split="train")
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id
        self.pad_id = -100
        self.user_start_id, self.user_end_id = tokenizer.user_start_token_id, tokenizer.user_end_token_id
        self.assistant_start_id, self.assistant_end_id = tokenizer.assistant_start_token_id, tokenizer.assistant_end_token_id
        self.pad_length = pad_length
        
    def __len__(self):
        return len(self.dataset)
    
    def _extract_answer(self, answer):
        # Extract only the final numerical answer using regex
        # Answer format: "...some steps...\n#### 72"
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer)
        if match:
            return match.group(1)
        return answer.strip()
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        answer = self._extract_answer(item['answer'])
        
        # Build conversation: <EOS><|user_start|>Q<|user_end|><|assistant_start|>A<|assistant_end|>
        item_tok = [self.eot_id]
        item_tok.append(self.user_start_id)
        item_tok += self.tokenizer(question, truncation=False, max_length=None)['input_ids']
        item_tok.append(self.user_end_id)
        item_tok.append(self.assistant_start_id)
        item_tok += self.tokenizer(answer, truncation=False, max_length=None)['input_ids']
        item_tok.append(self.assistant_end_id)
        
        # Pad or skip if too long
        if len(item_tok) > self.pad_length:
            return None
        item_tok += ([self.pad_id] * (self.pad_length - len(item_tok)))
        item_tok = np.array(item_tok)
        
        # Mask user content in labels
        label_tok = copy.deepcopy(item_tok)
        user_start_idx, user_end_idx = np.where(item_tok == self.user_start_id)[0], np.where(item_tok == self.user_end_id)[0]
        for i in range(user_start_idx.shape[0]):
            label_tok[user_start_idx[i]:(user_end_idx[i] + 1)] = self.pad_id
        label_tok[label_tok == self.assistant_start_id] = self.pad_id
        
        return item_tok[:-1], label_tok[1:]

class GSM8KTestDataset(Dataset):
    def __init__(self, dataset_name, subname, tokenizer):
        self.dataset_name, self.subname = dataset_name, subname
        self.dataset = load_dataset(dataset_name, name=subname, split="test")
        self.tokenizer = tokenizer
        self.user_start_id, self.user_end_id = tokenizer.user_start_token_id, tokenizer.user_end_token_id
        self.assistant_start_id = tokenizer.assistant_start_token_id
        self.eot_id = tokenizer.eos_token_id
    
    def __len__(self):
        return len(self.dataset)
    
    def _extract_answer(self, answer):
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer)
        if match:
            return match.group(1)
        return answer.strip()
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        final_answer = self._extract_answer(item['answer'])
        
        # Build prompt: <EOS><|user_start|>Q<|user_end|><|assistant_start|>
        prompt_tok = [self.eot_id, self.user_start_id]
        prompt_tok += self.tokenizer(question, truncation=False, max_length=None)['input_ids']
        prompt_tok += [self.user_end_id, self.assistant_start_id]
        
        return {
            'question': question,
            'final_answer': final_answer,
            'prompt_tokens': np.array(prompt_tok)
        }