import os
from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, batch_idx: int, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        if self.is_train:
            mixed_precision = self.mixed_precision
        else:
            mixed_precision = torch.float32

        with torch.autocast(
            self.device, dtype=mixed_precision, enabled=mixed_precision is not torch.float32
        ):
            outputs = self.model(batch["audio_mix"])
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)
            if self.is_train:
                batch["loss"] /= self.iters_to_accumulate

        if self.is_train:
            self.grad_scaler.scale(batch["loss"]).backward()
            if ((batch_idx + 1) % self.iters_to_accumulate == 0) or (
                (batch_idx + 1) == self.epoch_len
            ):
                self.grad_scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                metrics.update("grad_norm", self._get_grad_norm())
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None and self.scheduler_config is None:
                    self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            saved_loss = batch[loss_name].item()
            if self.is_train:
                saved_loss *= self.iters_to_accumulate
            metrics.update(loss_name, saved_loss)

        with torch.no_grad():
            for met in metric_funcs:
                metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            pass
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, **batch):
        spectrogram_keys = [k for k in batch.keys() if k.startswith("spectrogram")]
        spectrograms_for_plot = [batch[k][0].squeeze(0).detach().cpu() for k in spectrogram_keys]
        images = [
            plot_spectrogram(spectrogram, desc)
            for spectrogram, desc in zip(
                spectrograms_for_plot, [k.split("_")[-1] for k in spectrogram_keys]
            )
        ]
        if len(images) > 0:
            self.writer.add_images("spectrograms", images)

    def log_predictions(
        self,
        **batch,
    ):
        # TBD
        pass
