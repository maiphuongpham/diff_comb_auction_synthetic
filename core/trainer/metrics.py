import torch

def safe_pos(x, eps=1e-12):
    return torch.clamp(x, min=eps)

def jain_index(u):
    u = safe_pos(u)
    s1, s2 = u.sum(), (u**2).sum()
    A = u.numel()
    return (s1 * s1) / (A * s2 + 1e-12)

def coefficient_of_variation(u):
    u = safe_pos(u)
    mu, sd = u.mean(), u.std(unbiased=False)
    return sd / (mu + 1e-12)

def gini_index(u):
    u = safe_pos(u)
    A = u.numel()
    diffs = (u.unsqueeze(0) - u.unsqueeze(1)).abs()
    return diffs.mean() / (2.0 * (u.mean() + 1e-12))

def soft_min_utility(u, temp=0.1):
    # smooth approximation of min(u)
    smin = -temp * torch.log(torch.exp(-u / temp).sum())
    return smin

def fairness_metrics(u):
    """
    Compute all fairness metrics at once.
    Args:
        u: torch.Tensor [A] or [B, A] of per-airline utilities.
    Returns:
        dict of fairness scores
    """
    if u.dim() == 2:
        u = u.mean(dim=0)

    jain = jain_index(u)
    cv = coefficient_of_variation(u)
    gini = gini_index(u)
    smin = soft_min_utility(u)

    return {
        "jain_index": jain.item(),
        "cv": cv.item(),
        "gini": gini.item(),
        "soft_min_utility": smin.item(),
        "1_minus_jain": (1.0 - jain).item(),  # for loss comparability
    }

def jain_fairness_loss(u):
    """Loss for training (minimize unfairness)."""
    return 1.0 - jain_index(safe_pos(u))
