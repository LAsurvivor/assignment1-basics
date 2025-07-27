from torch import nn
import torch
from einops import einsum


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        self.W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        
        nn.init.trunc_normal_(
            self.W, mean=0.0, std=1, a=-3, b=3
        )
        self.W = nn.Parameter(self.W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
