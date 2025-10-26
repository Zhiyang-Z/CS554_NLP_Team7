from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import copy

class SFTDataset(Dataset):
    def __init__(self, dataset_name, subname, tokenizer):
        # NOTICE: dataset is a streaming dataset, make sure to split it by node and worker
        self.dataset_name, self.subname = dataset_name, subname
        self.dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id
        self.pad_id = -100 # use -100 for padding
        self.user_start_id, self.user_end_id = tokenizer.user_start_token_id, tokenizer.user_end_token_id
        self.assistant_start_id, self.assistant_end_id = tokenizer.assistant_start_token_id, tokenizer.assistant_end_token_id

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item_text = self.dataset[idx]['messages']
        item_tok = [self.eot_id]
        roles, cur_role = ['user', 'assistant'], 0
        role_start_toks, role_end_toks = [self.user_start_id, self.assistant_start_id], [self.user_end_id, self.assistant_end_id]
        # take turns to speak
        for turn in item_text:
            assert turn['role'] == roles[cur_role], f"{self.dataset[idx]}"
            # add current role tokens
            item_tok.append(role_start_toks[cur_role])
            item_tok += (self.tokenizer(turn['content'], truncation=False, max_length=None)['input_ids'])
            item_tok.append(role_end_toks[cur_role])
            cur_role = 1 - cur_role # switch role
        # there do exits some samples that end with user.
        assert cur_role == 0 # should end with assistant, wait for user.
        # # pad the seq to fixed-length
        # assert self.pad_length > len(item_tok), f"{self.pad_length} not great than {len(item_tok)}, case: {self.dataset[idx]}"
        # item_tok += ([self.pad_id] * (self.pad_length - len(item_tok)))
        item_tok = np.array(item_tok)
        # calculate corresponding label, mask user content
        label_tok = copy.deepcopy(item_tok)
        user_start_idx, user_end_idx = np.where(item_tok == self.user_start_id)[0], np.where(item_tok == self.user_end_id)[0]
        assert user_start_idx.shape[0] == user_end_idx.shape[0] # user start and end should be paired
        for i in range(user_start_idx.shape[0]):
            label_tok[user_start_idx[i]:(user_end_idx[i] + 1)] = self.pad_id
        # also mask <assistant_start>, <eos>
        label_tok[label_tok == self.assistant_start_id] = self.pad_id
        label_tok[label_tok == self.eot_id] = self.pad_id

        return item_tok, label_tok
