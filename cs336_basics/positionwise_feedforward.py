from torch import nn
import torch
from .linear import Linear
from einops import einsum


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 8 / 3 * d_model
        self.w1 = Linear(
            in_features=d_model,
            out_features=self.d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = Linear(
            in_features=self.d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.w3 = Linear(
            in_features=d_model,
            out_features=self.d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
