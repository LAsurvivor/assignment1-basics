import math


def lr_cosine_schedule(
    t: int, alpha_max: float, alpha_min: float, tw: int, tc: int
) -> float:
    # Warm-up phase
    if t < tw:
        return alpha_max * (t / tw)
    # Cosine annealing phase
    if t <= tc:
        progress = (t - tw) / (tc - tw)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine * (alpha_max - alpha_min)
    # Post-annealing
    return alpha_min
