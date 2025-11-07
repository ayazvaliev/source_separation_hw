import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class ExampleLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = ScaleInvariantSignalNoiseRatio()

    def forward(self, 
                audio_s1, 
                audio_s2,
                audio_mix_length,
                logits,  
                **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        sl_snr = torch.tensor(0.0)
        batch_size = logits.size(0)
        target_s1 = audio_s1.squeeze(-1)
        target_s2 = audio_s2.squeeze(-1)

        print("mix length shape: ", audio_mix_length.shape)
        print("target s1 shape: ", target_s1.shape)
        print("target s2 shape: ", target_s2.shape)

        for i in range(batch_size):
            target_s1 = target_s1[i, :audio_mix_length[i]] # (L,)
            target_s2 = target_s2[i, :audio_mix_length[i]] # (L,)

            cur_logits = logits[i, :, :audio_mix_length[i]] #(C, L)

            sl_snr += - torch.max(self.loss(target_s1, cur_logits[0]) + self.loss(target_s2,  cur_logits[1]), 
                self.loss(target_s2,  cur_logits[0]) + self.loss(target_s1,  cur_logits[1]) 
            )
        sl_snr /= batch_size

        return {"loss": sl_snr}
