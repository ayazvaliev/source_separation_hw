from torch import nn
import torch


class GLN(nn.Module):
    def __init__(self, n_channels, eps = 1e-5):
        super().__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_channels,1))
        self.betta = nn.Parameter(torch.zeros(n_channels,1))

    def forward(self, x):

        E_f = torch.mean(x)
        var_f = torch.var(x, unbiased=True)
        norm = (x - E_f)/torch.sqrt(var_f + self.eps) * self.gamma + self.betta
        return norm

