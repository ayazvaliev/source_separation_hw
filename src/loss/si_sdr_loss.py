import torch
import torch.nn as nn
from asteroid.losses import PITLossWrapper, pairwise_neg_snr


class SISDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss= PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")
    
    def forward(self, audio_s1, audio_s2, logits, **batch):
        target_audio = torch.concat([audio_s1, audio_s2], dim=1)
        return {"loss": self.loss(logits, target_audio)}