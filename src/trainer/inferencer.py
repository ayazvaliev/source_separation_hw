from pathlib import Path

import torch
import torchaudio
from torch_audiomentations import PeakNormalization
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model_ = model
        self.batch_transforms = batch_transforms
        self.torchscript = self.cfg_trainer.get("ts_compile", False)

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = Path(save_path)

        # normalizer

        self.peak_normalizer = PeakNormalization(
            apply_to="only_too_loud_sounds", sample_rate=16_000, output_type="tensor", p=1.0
        )

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

        if self.torchscript:
            self.model = torch.jit.script(self.model_)
        else:
            self.model = self.model_

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part_save_path):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(batch["audio_mix"])
        batch.update(outputs)

        batch["logits"] = self.peak_normalizer(batch["logits"])

        if "audio_concat" in batch and metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if part_save_path is not None:
            batch_size = batch["logits"].size(0)
            for i in range(batch_size):
                audio_mix_name = Path(batch["audio_mix_path"][i]).name
                for speaker_id, speaker_dir in enumerate(["s1", "s2"]):
                    save_name = part_save_path / speaker_dir / audio_mix_name
                    torchaudio.save(
                        save_name,
                        batch["logits"][i, speaker_id : speaker_id + 1].cpu(),
                        sample_rate=16_000,
                        format=audio_mix_name.split(".")[-1],
                    )

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            part_save_path = self.save_path / part
            part_save_path.mkdir(exist_ok=True, parents=True)
            (part_save_path / "s1").mkdir(exist_ok=True)
            (part_save_path / "s2").mkdir(exist_ok=True)
        else:
            part_save_path = None

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    metrics=self.evaluation_metrics,
                    part_save_path=part_save_path,
                )

        ret_none = self.evaluation_metrics is None or self.evaluation_metrics.empty
        return self.evaluation_metrics.result() if not ret_none else None
