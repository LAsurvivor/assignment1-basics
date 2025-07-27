def gradient_clipping(params: list, max_l2_norm: float, eps: float = 1e-6) -> None:
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    total_norm = total_norm_sq**0.5

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
