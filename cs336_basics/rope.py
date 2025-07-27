from typing import Tuple
from torch import nn
import torch


def apply_rot(
    x_even: torch.Tensor, x_odd: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return x_even * cos - x_odd * sin, x_even * sin + x_odd * cos


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None) -> None:
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )  # (d_k // 2,)
        positions = torch.arange(
            0, max_seq_len, dtype=dtype, device=device
        )  # (seq_len,)
        sinusoid_inp = torch.outer(positions, inv_freq)  # （seq_len, d_k // 2)

        cos_vals = torch.cos(sinusoid_inp)
        sin_vals = torch.sin(sinusoid_inp)

        self.register_buffer("cos_vals", cos_vals, persistent=False)
        self.register_buffer("sin_vals", sin_vals, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin_vals[token_positions]
        cos = self.cos_vals[token_positions]

        x_even, x_odd = x[..., ::2], x[..., 1::2]
        rot_even, rot_odd = apply_rot(x_even, x_odd, cos, sin)
        return torch.stack((rot_even, rot_odd), dim=-1).reshape_as(x)