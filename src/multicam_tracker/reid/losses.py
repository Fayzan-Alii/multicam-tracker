import torch


def pairwise_euclidean(x: torch.Tensor) -> torch.Tensor:
    sq = (x * x).sum(dim=1, keepdim=True)
    d2 = sq + sq.T - 2.0 * x @ x.T
    # The clamp keeps sqrt differentiable at zero distance (every sample to itself).
    return d2.clamp(min=1e-12).sqrt()


def batch_hard_triplet_loss(feats: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """For each anchor, the farthest positive and the closest negative in the batch
    (Hermans et al., "In Defense of the Triplet Loss")."""
    dist = pairwise_euclidean(feats)
    same = labels[:, None] == labels[None, :]
    hardest_pos = torch.where(same, dist, torch.zeros_like(dist)).max(dim=1).values
    hardest_neg = torch.where(~same, dist, torch.full_like(dist, float("inf"))).min(dim=1).values
    valid = torch.isfinite(hardest_neg)
    if not valid.any():
        return feats.sum() * 0.0
    return torch.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()
