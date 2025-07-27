from torch import nn
import torch

from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        max_seq_len: int = None,
        theta: float = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x), token_positions
        )
        x = x + self.ffn(self.ln2(x))
        return x
