
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data._utils.collate import default_collate

# class
class Eval_Choice:
    def __init__(self, dataloader, model, device):
        self.dataloader, self.model, self.device = dataloader, model, device

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        correct, total = 0, 0
        for q_batch, ctx_l, k_batch in tqdm(self.dataloader):
            assert q_batch.shape[0] == k_batch.shape[0] and q_batch.shape[1] == 4 # A, B, C, D choices
            N = q_batch.shape[0]
            self.model.clear_kv_cache()
            q_batch, ctx_l, k_batch = q_batch.reshape(4*N, -1).to(self.device), ctx_l.to(self.device), k_batch.to(self.device) # question and key
            q_batch_input, q_batch_label = q_batch[:,:-1].contiguous(), q_batch[:,1:].contiguous()
            padding_mask_input, padding_mask_label = q_batch_input == -100, q_batch_label == -100
            q_batch_input[padding_mask_input] = 0 # dummy token 0 for padding
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model(q_batch_input)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            q_batch_label[padding_mask_label] = 0 # dummy label 0 for padding
            log_probs_select = log_probs.gather(2, q_batch_label.unsqueeze(-1)).squeeze(-1)
            # mask the padding positions
            log_probs_select[padding_mask_label] = 0
            log_probs_select = log_probs_select.reshape(N, 4, -1) # (N, 4, L)
            # mask the context part, have to use for-loop (inefficient)
            ctx_l = (ctx_l - 1).squeeze() # -1 for <eos>
            for i in range(N):
                log_probs_select[i,:,0:ctx_l[i]] = 0
            # calculate valid length
            l_total = q_batch_label.shape[1]
            l_padding = padding_mask_label.sum(dim=-1).reshape(N, 4)
            l = -ctx_l[:,None] - l_padding + l_total # (N, 4)
            score = log_probs_select.sum(dim=-1) / l
            model_select = score.argmax(dim=-1)
            correct += (model_select == k_batch.squeeze(-1)).sum().cpu().item()
            total += N
        # print(f"total: {total}, correct: {correct}, correct rate: {correct/total}")
        return correct/total * 100

def eval_pad(batch):
    batch_max_len = -1
    for item in batch:
        assert item[0].shape[0] == 4 and item[1].shape[0] == 1 and item[2].shape[0] == 1
        if item[0].shape[1] > batch_max_len: batch_max_len = item[0].shape[1]
    batch_align_len = batch_max_len
    batch_padded = []
    for item in batch:
        assert item[0].shape[0] == 4 and item[1].shape[0] == 1 and item[2].shape[0] == 1
        if batch_align_len > item[0].shape[1]: # pad to fixed length.
            padding_len = batch_align_len - item[0].shape[1]
            input_padding = np.full((4, padding_len), fill_value=-100)
            batch_padded.append((np.append(item[0], input_padding, axis=1), item[1], item[2]))
        else:
            batch_padded.append((item[0], item[1], item[2]))
    return default_collate(batch_padded)

def make_dataloader4eval_choice(dataset, batch_size):
    return DataLoader(dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False, # need to shuffle
                    num_workers=4,
                    drop_last=False,
                    prefetch_factor=2,
                    collate_fn=eval_pad)