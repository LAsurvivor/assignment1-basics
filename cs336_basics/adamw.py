import math
from typing import Callable, Optional
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]  # Get the beta values.
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                state = self.state[p]  # Get state associated with p.

                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                lr_t = lr * (bias_correction2 ** 0.5) / bias_correction1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-lr_t)

                if wd != 0:
                    p.data.add_(p.data, alpha=-wd * lr)

        return loss
