import torch
from torch import nn
from .scaled_dot_product_attention import scaled_dot_product_attention
from .rope import RoPE
from .linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        token_positions: torch.Tensor = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.token_positions = token_positions

        self.q_weight = Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.k_weight = Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.v_weight = Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.o_weight = Linear(
            in_features=d_model,
            out_features=d_model,
        )

        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.dk, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        Q = (
            self.q_weight(x)
            .view(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
        )
        K = (
            self.k_weight(x)
            .view(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
        )
        V = (
            self.v_weight(x)
            .view(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
        )

        if hasattr(self, 'rope') and self.token_positions is not None:
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attention_output = scaled_dot_product_attention(Q, K, V, mask=~mask)

        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.o_weight(attention_output)
