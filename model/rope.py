import torch
import torch.nn.functional as F
import math
from typing import Tuple

def rotary_pos_emb_frequencies(head_dim: int, device, ex_ratio=1, base: float = 10000.0) -> torch.Tensor:
    """
    Returns frequencies of shape (head_dim//2,)
    frequencies: [theta_1, theta_2, theta_3,..., theta_(d/2)]
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE."
    half = head_dim // 2
    inv_freq = (1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))) * (1.0 / ex_ratio)
    print(f"rotary embedding will be interpolated with ratio: {ex_ratio}")
    return inv_freq.to(device)  # shape (half,)

def precompute_cos_sin(seq_len: int, head_dim: int, device, ex_ratio=1, dtype=torch.float32, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for positions [0..seq_len-1].
    Returns cos, sin each of shape (seq_len, 1, head_dim) to broadcast over (B, H, L, D).
    """
    print(f'max length: {seq_len}')
    inv_freq = rotary_pos_emb_frequencies(head_dim, device, ex_ratio, base=base)  # (half,), frequencies: [theta_1, theta_2, theta_3,..., theta_(d/2)]
    positions = torch.arange(seq_len, device=device, dtype=dtype)  # (seq_len,)
    # outer -> (seq_len, half)
    angle = torch.einsum("p,d->pd", positions, inv_freq)  # (seq_len, half)
    # build cos/sin for interleaved dims: [cos0, cos1, ...] and [sin0, sin1, ...] need to be expanded to head_dim
    # create (seq_len, head_dim) interleaved [angle0, angle0, angle1, angle1, ...]
    angle = torch.repeat_interleave(angle, 2, dim=-1)  # (seq_len, head_dim)
    cos = torch.cos(angle)  # (seq_len, head_dim)
    sin = torch.sin(angle)  # (seq_len, head_dim)
    return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)

def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """
    extract the matrix for multipling with cos and sin
    """
    x1 = x[..., ::2]  # even dims
    x2 = x[..., 1::2]  # odd dims
    # new interleaved: (-x2, x1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)  # flatten last two dims back to head_dim

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: shape (B, H, L, D_head)
    cos, sin: shape (L, 1, D_head) or broadcastable to q/k
    returns rotated q,k with same shape.
    """
    # assume q/k are float dtype consistent with cos/sin
    # rotate pairs and apply:
    q_rot = (q * cos) + (rotate_every_two(q) * sin)
    k_rot = (k * cos) + (rotate_every_two(k) * sin)
    return q_rot, k_rot