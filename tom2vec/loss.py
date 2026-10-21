import torch
import torch.nn.functional as F


def tom_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"this should not happen")
    cos = F.cosine_similarity(predicted.float(), target.detach().float(), dim=-1)
    return (1.0 - cos).sum(dim=-1).mean()
