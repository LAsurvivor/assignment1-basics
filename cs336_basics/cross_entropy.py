import torch


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]

    flat_logits = logits.view(-1, vocab_size)  # [B, V]
    flat_targets = targets.view(-1)  # [B]

    max_logits = flat_logits.max(dim=-1, keepdim=True).values  # [B, 1]
    stabilized = flat_logits - max_logits  # [B, V]

    logsumexp = torch.log(torch.exp(stabilized).sum(dim=-1))  # [B]

    true_logits = stabilized.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(
        -1
    )  # [B]

    loss = -true_logits + logsumexp  # [B]

    return loss.mean()
