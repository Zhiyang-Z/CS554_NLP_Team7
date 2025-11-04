import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import numpy as np
import math

from model.rope import precompute_cos_sin, apply_rotary_pos_emb

class Attention(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_head: int,
        dim_head: int,
        max_cache_len: int,
        dropout_rate,
        device
    ):
        super().__init__()
        self.inner_dim = dim_head * n_head
        self.n_head, self.dim_head = n_head, dim_head
        self.to_qkv = nn.Linear(n_dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, n_dim)
        self.dropout_rate = dropout_rate
        self.res_dropout = nn.Dropout(self.dropout_rate)
        self.device = device
        # KV cache
        self.cached_k, self.cached_v = None, None
        self.cur_cached_len, self.max_cache_len = 0, max_cache_len

    def forward(self, x: torch.Tensor, rope_param):
        B, L, D = x.shape
        rope_cos, rope_sin = rope_param
        # do attention operation
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "B L (h d) -> B h L d", h=self.n_head)
        k = rearrange(k, "B L (h d) -> B h L d", h=self.n_head)
        v = rearrange(v, "B L (h d) -> B h L d", h=self.n_head)
        # rotary embedding and cache (if reference)
        if self.training:
            with torch.autocast(device_type='cuda', enabled=False):
                q, k = apply_rotary_pos_emb(q,
                                            k,
                                            rope_cos[0:L, :],
                                            rope_sin[0:L, :])
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True, dropout_p=self.dropout_rate)
        else:
            assert self.cur_cached_len < self.max_cache_len, "Cache length exceeds maximum limit."
            assert self.cur_cached_len >= 0
            with torch.autocast(device_type='cuda', enabled=False):
                q, k = apply_rotary_pos_emb(q,
                                            k,
                                            rope_cos[self.cur_cached_len : (self.cur_cached_len + L), :],
                                            rope_sin[self.cur_cached_len : (self.cur_cached_len + L), :])
            if self.cur_cached_len == 0:
                # the first time to allocate memory when inference.
                self.cached_k = torch.empty((B, self.n_head, self.max_cache_len, self.dim_head), device=self.device)
                self.cached_v = torch.empty((B, self.n_head, self.max_cache_len, self.dim_head), device=self.device)
            # cache KV
            self.cached_k[:, :, self.cur_cached_len : (self.cur_cached_len + L), :] = k
            self.cached_v[:, :, self.cur_cached_len : (self.cur_cached_len + L), :] = v
            self.cur_cached_len += L
            # current Q, K, V
            q, k, v = q, self.cached_k[:, :, 0:self.cur_cached_len, :], self.cached_v[:, :, 0:self.cur_cached_len, :]
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "B h L d -> B L (h d)")
        x = x.to(q.dtype)

        # linear proj
        x = self.res_dropout(self.to_out(x))
        return x
    
    def clear_kv_cache(self):
        self.cached_k, self.cached_v = None, None
        self.cur_cached_len = 0
        # print('cache reset.') # for debug

class Decoder_Block(nn.Module):
    def __init__(
        self,
        n_dim = 1280,
        n_head = 20,
        dim_head = 64,
        ff_ratio = 4.0,
        dropout_rate = 0.1,
        train_length = 1024,
        ex_scale = 1.2,
        device = ''
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_dim)
        # assert n_dim % n_head == 0, "n_dim % n_head != 0, please check."
        self.att = Attention(n_dim, n_head, dim_head, int(train_length * ex_scale), dropout_rate, device)
        self.norm2 = nn.LayerNorm(n_dim)
        # Feed Forward
        ff_hidden_dim = int(n_dim * ff_ratio)
        self.ff1 = nn.Linear(n_dim, ff_hidden_dim)
        self.ff2 = nn.Linear(ff_hidden_dim, n_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # model device
        self.device = device

    def forward(self, x, rope_param):
        y = self.norm1(x)
        x = x + self.att(y, rope_param)
        y = self.norm2(x)
        x = x + self.dropout(self.ff2(F.gelu(self.ff1(y))))
        return x
    
    def clear_kv_cache(self):
        self.att.clear_kv_cache()

class GPT(nn.Module):
    def __init__(
        self,
        v_size = 50257,
        train_length = 1024,
        n_dim = 1280,
        n_layer = 36,
        n_head = 20,
        dim_head = 64,
        ff_ratio = 4.0,
        dropout_rate = 0.1,
        device = '',
        ex_ratio = 1.2,
    ):
        super().__init__()
        self.v_size, self.train_length = v_size, train_length
        self.n_dim, self.n_layer, self.n_head, self.dim_head = n_dim, n_layer, n_head, dim_head
        self.ff_ratio, self.dropout_rate = ff_ratio, dropout_rate
        self.device = device
        self.ex_ratio = ex_ratio
        # define transformer layers
        self.embedding = nn.Embedding(self.v_size, self.n_dim)
        self.decoder_layers = nn.ModuleList(
            [Decoder_Block(self.n_dim, self.n_head, self.dim_head, self.ff_ratio, self.dropout_rate, self.train_length, self.ex_ratio, self.device) for _ in range(self.n_layer)])
        # final output
        self.final_norm = nn.LayerNorm(self.n_dim)
        self.out = nn.Linear(self.n_dim, self.v_size, bias=False)
        # weight tying
        self.embedding.weight = self.out.weight
        # cache the rotary embedding cos/sin parameters
        rope_cos, rope_sin = precompute_cos_sin(int(train_length * ex_ratio), self.dim_head, device)
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)
        # initialize parameters
        print('transformer initializing...')
        self.apply(self._ini_para)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('to_out.weight') or pn.endswith('ff2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layer))

    def _ini_para(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x):
        # x shape: [Batch, Length]
        x = self.embedding(x)
        # x += self.pos_embed[:, :x.shape[1], :].to(x.dtype)
        for layer in self.decoder_layers:
            x = layer(x, (self.rope_cos, self.rope_sin))
        x = self.final_norm(x)
        x = self.out(x)
        return x
    
    def clear_kv_cache(self):
        for layer in self.decoder_layers:
            layer.clear_kv_cache()

            
if __name__ == "__main__":
    print(torch.backends.cuda.flash_sdp_enabled())      # FlashAttention available?
    print(torch.backends.cuda.mem_efficient_sdp_enabled())  # MemEfficient available?
    print(torch.backends.cuda.math_sdp_enabled())       # Math fallback
    gpt = GPT(device=f'cuda:0').to(torch.device('cuda'))
    gpt.eval()
    dummy_input = torch.zeros((4, 32), dtype=torch.int, device=torch.device('cuda'))
    out = gpt(dummy_input)
        