from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

class SFTDataset(Dataset):
    def __init__(self, dataset_name, subname, tokenizer, pad_length):
        # NOTICE: dataset is a streaming dataset, make sure to split it by node and worker
        self.dataset_name, self.subname = dataset_name, subname
        self.dataset = load_dataset(dataset_name, subname, split="train")
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.user_start_id, self.user_end_id = tokenizer.user_start_token_id, tokenizer.user_end_token_id
        self.assistant_start_id, self.assistant_end_id = tokenizer.assistant_start_token_id, tokenizer.assistant_end_token_id
        # set max length
        # everyday-conversations has max length 294
        self.pad_length = pad_length

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item_text = self.dataset[idx]['messages']
        item_tok = [self.eot_id]
        roles, cur_role = ['user', 'assistant'], 0
        role_start_toks, role_end_toks = [self.user_start_id, self.assistant_start_id], [self.user_end_id, self.assistant_end_id]
        # take turns to speak
        for turn in item_text:
            assert turn['role'] == roles[cur_role]
            # add current role tokens
            item_tok.append(role_start_toks[cur_role])
            item_tok.extend(self.tokenizer(turn['content'], truncation=False, max_length=None)['input_ids'])
            item_tok.append(role_end_toks[cur_role])
            cur_role = 1 - cur_role # switch role
        # there do exits some samples that end with user, we drop them.
        # assert cur_role == 0 # should end with assistant, wait for user.

        return np.array(item_tok, dtype=np.int32)
