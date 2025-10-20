import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import math

import wandb
from tqdm import tqdm

class Pre_Trainer:
    def __init__(
        self,
        train_data_loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        grad_accum_steps: int,
        config: dict,
        resume: bool
    ) -> None:
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.device = f'cuda:{self.rank}'
        self.config = config
        self.save_path = self.config['path']['save']
        self.resume = resume

        self.model = model
        self.model = self.model.to(self.device)
        self.model = DDP(self.model)
        # compile should be after DDP, refer to https://pytorch.org/docs/main/notes/ddp.html
        self.model = torch.compile(self.model)
        
        self.model.train()

        self.train_data_loader = train_data_loader
        self.optimizer, self.scheduler = optimizer, scheduler
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

        # Creates a GradScaler for mixed precision training.
        self.scaler = torch.GradScaler()
        if self.resume:
            self.scaler.load_state_dict(torch.load(self.config['path']['load'], "cpu")['scaler_state_dict'])

        self.grad_accum_steps = grad_accum_steps

        if self.rank == 0:
            wandb.init(project="Final", entity="CS554_NLP")

    def train(self):
        if self.resume: checkpoint = torch.load(self.config['path']['load'], "cpu")
        step = checkpoint['global_step'] if self.resume else 0
        avg_loss = torch.zeros((1,), device=self.device)
        avg_grad_norm = torch.zeros((1,), device=self.device)
        for epoch in range(checkpoint['epoch'] if self.resume else 0, 2000000000): # termination is decide by human.
            # Here, we shuffle dataset for each epoch
            self.train_data_loader.dataset.set_and_shuffle_dataset(46+epoch)
            if self.resume:
                self.train_data_loader.dataset.dataset.load_state_dict(checkpoint['dataset_state'])
            self.optimizer.zero_grad(set_to_none = True) # clear remainder when iterating dataset.
            avg_loss.zero_()
            avg_grad_norm.zero_()
            self.resume = False # only use resume for first epoch
            for i, data in tqdm(enumerate(self.train_data_loader)):
                self.model.train()
                data = data.to(self.device)
                x, y = data[:,:-1].contiguous(), data[:,1:].contiguous() # input and label
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(x)
                    voc_size = logits.shape[-1]
                    loss = self.loss_fn(logits.view(-1, voc_size), y.view(-1))
                    loss = loss / self.grad_accum_steps
                    avg_loss[0] += loss.item()
                if (i + 1) % self.grad_accum_steps == 0:
                    # backprop the accumulated gradients    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    avg_grad_norm[0] = grad_norm.item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    step += 1
                    self.optimizer.zero_grad(set_to_none = True)
                    # collect training info
                    try:
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                        dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.AVG)
                    except Exception as e:
                        print(f"rank {self.rank} all_reduce failed at step {step}: {e}")
                        raise

                    save_freq = 375 # 144 for 1.3B_2A100_1.35it/s, 3600 for 0.125B_4L40S_3it/s
                    if step % save_freq == 1: # collect complete optimizer state before saving
                        # self.optimizer.consolidate_state_dict(to=0)
                        if self.rank == 0:
                            self.test(step)
                            torch.save({
                                        # 'dataset_state': self.train_data_loader.dataset.dataset.state_dict(),
                                        'model_state_dict': self.model.module.state_dict(),
                                        # 'optimizer_state_dict': optim_to_save.state_dict(),
                                        'scaler_state_dict': self.scaler.state_dict(),
                                        'scheduler_state_dict': self.scheduler.state_dict(),
                                        'epoch': epoch,
                                        'global_step': step
                                    }, f"{self.save_path}/latest.pt")
                    
                    if self.rank == 0:
                        wandb.log({"epoch": epoch}, step=step, commit = False)
                        wandb.log({"grad_norm": avg_grad_norm.item()}, step=step, commit = False)
                        wandb.log({"lr": self.scheduler.get_last_lr()[0]}, step=step, commit = False)
                        wandb.log({"perplexity": math.exp(avg_loss[0].item())}, step=step, commit = False)
                        wandb.log({"loss": avg_loss[0].item()}, step=step, commit=True)
                    avg_loss.zero_()
                    avg_grad_norm.zero_()

                    if dist.is_initialized(): dist.barrier()
                else:
                    with self.model.no_sync():
                        self.scaler.scale(loss).backward()

    @torch.no_grad
    def sample(self):
        self.model.eval()
        self.model.module.clear_kv_cache()
        tokenizer = self.train_data_loader.dataset.tokenizer
        ans_token = [tokenizer.eos_token_id]
        while (not (len(ans_token) >= 256 or ans_token[-1] == tokenizer.eos_token_id)) or len(ans_token) == 1:
            last_token = torch.tensor([[ans_token[-1]]]).to(self.device)
            next_token_logits = self.model.module(last_token)[0,-1,:] # / 0.8
            k = 200 # top k sample
            topk_logits, topk_indices = torch.topk(next_token_logits, k)
            topk_probs = torch.softmax(topk_logits, dim=-1)
            next_token = topk_indices[torch.multinomial(topk_probs, 1)].cpu().item()
            ans_token.append(next_token)
        ans_text = tokenizer.decode(ans_token, skip_special_tokens=True)
        self.model.module.clear_kv_cache()
        return ans_text
        
    @torch.no_grad
    def test(self, step):
        # wandb 0.19.10 works fine.
        sample_text = ""
        for i in range(8):
            sample = self.sample()
            sample_text += f"=== Sample {i+1} ===<br>{sample}<br><br>"
        # wandb.log({f"sample_text": wandb.Html(sample_text)}, step=step, commit = False)
        print(sample_text)
