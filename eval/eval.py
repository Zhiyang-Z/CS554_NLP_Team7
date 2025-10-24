from datasets import load_dataset
import torch
from tqdm import tqdm

# class
class Pretrain_Eval:
    def __init__(self, device):
        self.eval_name = ["hellaswag", "mmlu"]
        self.device = device

    def eval(self, cur_eval, tokenizer, model):
        assert cur_eval in self.eval_name, f"Evaluation {cur_eval} not supported."
        model.eval()
        if cur_eval == "hellaswag":
            dataset = load_dataset("hellaswag", split="validation")
            return self.eval_hellaswag(dataset, tokenizer, model)
        elif cur_eval == "mmlu":
            dataset = load_dataset("cais/mmlu", "all", split="validation")
            return self.eval_mmlu(dataset, tokenizer, model)
    
    @torch.no_grad()
    def eval_hellaswag(self, dataset, tokenizer, model):
        correct = 0
        test_len = len(dataset)
        print("Evaluating HellaSwag...")
        for item in tqdm(dataset):
            ans_losses = []
            context, endings, labels = item["ctx"]+" ", item["endings"], item["label"]
            for ending in endings:
                inputs = [tokenizer.eos_token_id] + tokenizer(context + ending, truncation=False, max_length=None)['input_ids']
                inputs_tensor = torch.tensor([inputs]).to(self.device)
                model.clear_kv_cache()
                logits = model(inputs_tensor)
                ans_logits, y = logits[:, :-1, :].contiguous(), inputs_tensor[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(ans_logits.view(-1, ans_logits.size(-1)), y.view(-1), reduction='mean')
                ans_losses.append(loss.item())
            predicted_label = ans_losses.index(min(ans_losses))
            if predicted_label == int(labels):
                correct += 1
        accuracy = correct / test_len
        return accuracy
    
    @torch.no_grad()
    def eval_mmlu(self, dataset, tokenizer, model):
        correct = 0
        test_len = len(dataset)
        print("Evaluating HellaSwag...")
        for item in tqdm(dataset):
            ans_losses = []
            context, endings, labels = item["question"]+" ", item["choices"], item["answer"]
            for ending in endings:
                inputs = [tokenizer.eos_token_id] + tokenizer(context + ending, truncation=False, max_length=None)['input_ids']
                inputs_tensor = torch.tensor([inputs]).to(self.device)
                model.clear_kv_cache()
                logits = model(inputs_tensor)
                ans_logits, y = logits[:, :-1, :].contiguous(), inputs_tensor[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(ans_logits.view(-1, ans_logits.size(-1)), y.view(-1), reduction='mean')
                ans_losses.append(loss.item())
            predicted_label = ans_losses.index(min(ans_losses))
            if predicted_label == int(labels):
                correct += 1
        accuracy = correct / test_len
        return accuracy
