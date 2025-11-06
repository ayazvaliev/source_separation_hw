import torch


class CMVN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def __call__(self, mel: torch.Tensor):
        mean = mel.mean(dim=-1, keepdim=True)
        std = mel.std(dim=-1, keepdim=True)
        return (mel - mean) / (std + 1e-6)
