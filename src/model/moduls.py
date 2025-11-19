from torch import nn
import torch

EPS = 1e-8

class GLN(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.mean = nn.Parameter(torch.ones(input_channel), requires_grad=True)
        self.var = nn.Parameter(torch.zeros(input_channel), requires_grad=True)
    
    def forward(self, x: torch.Tensor):
        # x [B, F, T]
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        x_normalized = (x - mean) / (var + EPS).sqrt()
        return (self.var * x_normalized.transpose(1, 2) + self.mean).transpose(1, 2)
