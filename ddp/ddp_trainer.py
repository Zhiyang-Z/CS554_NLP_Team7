import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
import os

import wandb
from tqdm import tqdm

class Pre_Trainer:
    def __init__(
        self,
        train_data_loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_accum_steps: int,
    ) -> None:
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.device = f'cuda:{self.rank}'

        self.model = model
        self.model = self.model.to(self.device)
        self.model = DDP(self.model)
        # compile should be after DDP, refer to https://pytorch.org/docs/main/notes/ddp.html
        self.model = torch.compile(self.model)
        
        self.model.train()

        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

        # Creates a GradScaler for mixed precision training.
        self.scaler = torch.GradScaler()

        self.grad_accum_steps = grad_accum_steps

        if self.rank == 0:
            wandb.init(project="CS554_NLP", entity="zhiyang_zhang-worcester-polytechnic-institute")

    def train(self):
        step = -1
        avg_loss = torch.zeros((1,), device=self.device)
        avg_grad_norm = torch.zeros((1,), device=self.device)
        for epoch in range(2000000000): # termination is decide by human.
            # Here, we shuffle dataset for each iteration
            self.train_data_loader.dataset.set_and_shuffle_dataset(46+epoch)
            self.optimizer.zero_grad() # clear remainder when iterating dataset.
            avg_loss.zero_()
            avg_grad_norm.zero_()
            for i, data in tqdm(enumerate(self.train_data_loader)):
                self.model.train()
                data = data.to(self.device)
                x, y = data[:,:-1].contiguous(), data[:,1:].contiguous() # input and label
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    logits = self.model(x)
                    voc_size = logits.shape[-1]
                    loss = self.loss_fn(logits.view(-1, voc_size), y.view(-1))
                    loss = loss / self.grad_accum_steps
                    avg_loss[0] += loss.item()
                self.scaler.scale(loss).backward()
                if (i + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    avg_grad_norm[0] = grad_norm.item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    step += 1
                    self.optimizer.zero_grad()
                    # collect training info
                    try:
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                        dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.AVG)
                    except Exception as e:
                        print(f"rank {self.rank} all_reduce failed at step {step}: {e}")
                        raise
                    if self.rank == 0:
                        wandb.log({"epoch": epoch}, commit = False)
                        wandb.log({"grad_norm": avg_grad_norm.item()}, commit = False)
                        wandb.log({"loss": avg_loss[0].item()}, commit=True)
                    avg_loss.zero_()
                    avg_grad_norm.zero_()

                    if step % 70 == 0 and self.rank == 0:
                        self.test()

                    if step % 70 == 0 and self.rank == 0:
                        torch.save({
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'epoch': epoch,
                                    'global_step': step
                                }, f"/home/zzhang18/proj/CS554_NLP_Team7/saved_models/{step}.pt")

                    if dist.is_initialized(): dist.barrier()

    @torch.no_grad
    def sample(self, ini_input):
        self.model.eval()
        self.model.module.clear_kv_cache()
        tokenizer = self.train_data_loader.dataset.tokenizer
        tokens = tokenizer(ini_input, truncation=False, max_length=None)['input_ids']
        prompt = torch.tensor(tokens).unsqueeze(0).to(self.device)
        ans_token = []
        next_token_logits = self.model.module(prompt)[0,-1,:]
        k = 200 # top k sample
        topk_logits, topk_indices = torch.topk(next_token_logits, k)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        next_token = topk_indices[torch.multinomial(topk_probs, 1)].cpu().item()
        ans_token.append(next_token)
        while not (len(ans_token) >= 128 or ans_token[-1] == tokenizer.eos_token_id):
            last_token = torch.tensor([ans_token[-1]]).unsqueeze(0).to(self.device)
            next_token_logits = self.model.module(last_token)[0,-1,:]
            k = 200 # top k sample
            topk_logits, topk_indices = torch.topk(next_token_logits, k)
            topk_probs = torch.softmax(topk_logits, dim=-1)
            next_token = topk_indices[torch.multinomial(topk_probs, 1)].cpu().item()
            ans_token.append(next_token)
        ans_text = tokenizer.decode(tokens+ans_token, skip_special_tokens=True)
        self.model.module.clear_kv_cache()
        return ans_text
        
    @torch.no_grad
    def test(self):
        # Qualitative test
        tests = {}
        # 1. Language Understanding
        lan_tests = ['The dog is chasing the ', 'He go to school every day. Here is grammar error in this sentence, and the correct one is: ']
        tests['language'] = lan_tests
        # 2. common sense
        cs_test = ['The capital city of the United States is ', 'World War II ends in ', 'After putting ice cube into hot water, the hot water will ']
        tests['common sense'] = cs_test
        # 3. reasoning
        re_test = ['17 * 24 = ', 'Tom is taller than Jerry. Jerry is taller than Bob. Thus the tallest one is ', 'Bird is to fly as fish is to ']
        tests['reasoning'] = re_test
        # 4. Writing
        wr_test = ['My name is Vivek, I am a student in AI major, this is my introduction: ', 'Here is a joke: ']
        tests['writing'] = wr_test

        ans = {}
        for key, prompts in tests.items():
            # ans[key] = []
            for prompt in prompts:
                model_ans = self.sample(prompt)
                # ans[key].append(model_ans)
                wandb.log({f"{key}: ": wandb.Html(model_ans)})
