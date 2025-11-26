from torch.utils.data import Dataset
import numpy as np
import copy

class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id
        self.pad_id = -100 # use -100 for padding
        self.user_start_id, self.user_end_id = tokenizer.user_start_token_id, tokenizer.user_end_token_id
        self.assistant_start_id, self.assistant_end_id = tokenizer.assistant_start_token_id, tokenizer.assistant_end_token_id

        self.max_len = []

    def __len__(self):
        return len(self.dataset)
    
    def _get_conversation(self, idx):
        item_text = self.dataset[idx]['messages'] if 'messages' in self.dataset[idx] else self.dataset[idx]['conversation']
        item_tok = [self.eot_id]
        roles, cur_role = ['user', 'assistant'], 0
        role_start_toks, role_end_toks = [self.user_start_id, self.assistant_start_id], [self.user_end_id, self.assistant_end_id]
        # take turns to speak
        meet_system, system_add_start = False, False
        for turn in item_text:
            if turn['role'] == 'system':
                assert False, f"Temporarily disable training on <SYSTEM>, exception: {self.dataset[idx]}"
                assert meet_system == False
                assert 'user' == roles[cur_role], f"{self.dataset[idx]}" # system must be at begining.
                meet_system = True
                system_add_start = True
                item_tok.append(role_start_toks[cur_role])
                item_tok += (self.tokenizer(turn['content']+'\n', truncation=False, max_length=None)['input_ids'])
                continue
            assert turn['role'] == roles[cur_role], f"{self.dataset[idx]}"
            # add current role tokens
            if not system_add_start:
                item_tok.append(role_start_toks[cur_role])
            system_add_start = False
            item_tok += (self.tokenizer(turn['content'], truncation=False, max_length=None)['input_ids'])
            item_tok.append(role_end_toks[cur_role])
            cur_role = 1 - cur_role # switch role
        # there do exits some samples that end with user.
        # assert cur_role == 0 # should end with assistant, wait for user.
        item_tok = np.array(item_tok)
        # calculate corresponding label, mask user content
        label_tok = copy.deepcopy(item_tok)
        user_start_idx, user_end_idx = np.where(item_tok == self.user_start_id)[0], np.where(item_tok == self.user_end_id)[0]
        assist_start_idx, assist_end_idx = np.where(item_tok == self.assistant_start_id)[0], np.where(item_tok == self.assistant_end_id)[0]
        assert user_start_idx.shape[0] == user_end_idx.shape[0], f"{self.tokenizer.decode(item_tok, skip_special_tokens=False)}" # user start and end should be paired
        assert assist_start_idx.shape[0] == assist_end_idx.shape[0], f"{self.tokenizer.decode(item_tok, skip_special_tokens=False)}" # assist start and end should be paired
        for i in range(user_start_idx.shape[0]):
            label_tok[user_start_idx[i]:(user_end_idx[i] + 1)] = self.pad_id
        # also mask <assistant_start>, <eos>
        label_tok[label_tok == self.assistant_start_id] = self.pad_id
        label_tok[label_tok == self.eot_id] = self.pad_id

        return item_tok, label_tok
    
    def _get_math_question(self, idx):
        question, answer = None, None
        if 'question' in self.dataset[idx]:
            question, answer = self.dataset[idx]['question'], self.dataset[idx]['answer' if 'answer' in self.dataset[idx] else 'solution']
        elif 'input' in self.dataset[idx]:
            question, answer = self.dataset[idx]['input'], self.dataset[idx]['output']
        else:
            raise RuntimeError('dataset format is wrong!')
        item_tok = [self.eot_id]
        item_tok.append(self.user_start_id)
        item_tok += (self.tokenizer(question, truncation=False, max_length=None)['input_ids'])
        item_tok.append(self.user_end_id)
        item_tok.append(self.assistant_start_id)
        item_tok += (self.tokenizer(answer, truncation=False, max_length=None)['input_ids'])
        item_tok.append(self.assistant_end_id)
        # padding now
        item_tok = np.array(item_tok)
        # calculate corresponding label, mask user content
        label_tok = copy.deepcopy(item_tok)
        user_start_idx, user_end_idx = np.where(item_tok == self.user_start_id)[0], np.where(item_tok == self.user_end_id)[0]
        assist_start_idx, assist_end_idx = np.where(item_tok == self.assistant_start_id)[0], np.where(item_tok == self.assistant_end_id)[0]
        assert user_start_idx.shape[0] == user_end_idx.shape[0], f"{self.tokenizer.decode(item_tok, skip_special_tokens=False)}" # user start and end should be paired
        assert assist_start_idx.shape[0] == assist_end_idx.shape[0], f"{self.tokenizer.decode(item_tok, skip_special_tokens=False)}" # assist start and end should be paired
        for i in range(user_start_idx.shape[0]):
            label_tok[user_start_idx[i]:(user_end_idx[i] + 1)] = self.pad_id
        # also mask <assistant_start>, <eos>
        label_tok[label_tok == self.assistant_start_id] = self.pad_id
        label_tok[label_tok == self.eot_id] = self.pad_id

        return item_tok, label_tok
    
    def __getitem__(self, idx):
        if 'messages' in self.dataset[idx] or 'conversation' in self.dataset[idx]:
            return self._get_conversation(idx)
        elif 'question' in self.dataset[idx] or 'input' in self.dataset[idx]:
            return self._get_math_question(idx)
        else:
            raise RuntimeError('dataset format is wrong!')
