import math
from einops import einsum
import torch
from .softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    d_k = Q.size(-1)

    scores = einsum(Q, K, "... q d,... k d -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    softmax_scores = softmax(scores, dim=-1)

    return einsum(softmax_scores, V, "... q k, ... k d -> ... q d")
