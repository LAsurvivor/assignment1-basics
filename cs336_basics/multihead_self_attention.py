import torch
from torch import nn
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RoPE
from .linear import Linear
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.dk, max_seq_len=max_seq_len)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        Q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.num_heads)

        if hasattr(self, "rope"):
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attention_output = scaled_dot_product_attention(Q, K, V, mask=~mask)

        attention_output = rearrange(attention_output, "b h s d -> b s (h d)")
        return self.output_proj(attention_output)
