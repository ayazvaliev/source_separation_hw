import torch


def gather_by_perm(preds: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Reorder preds [B, S, T] according to perm [B, S], so that the j-th
    output channel corresponds to the j-th target channel.
    """
    # preds: [B, S, T]; perm: [B, S] (indices along speaker dim)
    B, S, T = preds.shape
    # expand perm to index along time dimension
    idx = perm.unsqueeze(-1).expand(-1, -1, T)  # [B, S, T]
    return torch.gather(preds, dim=1, index=idx)
