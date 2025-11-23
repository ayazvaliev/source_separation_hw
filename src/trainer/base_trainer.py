import gc
from abc import abstractmethod
from math import ceil

import torch
from hydra.utils import get_class, instantiate
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.trainer.trainer_utils import get_optimizer_grouped_parameters, has_param
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        metrics,
        config,
        project_config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            text_encoder (CTCTextEncoder): text encoder.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.project_config = project_config

        self.model_ = model

        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = self.cfg_trainer.get("skip_oom", True)

        self.logger = logger

        self.checkpoint_dir = ROOT_PATH / self.cfg_trainer.save_dir / config.writer.run_name

        self.criterion = instantiate(config.loss_function).to(self.device)

        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]

        if self.cfg_trainer.gradient_accumulation is None:
            gradient_accumulation = self.train_dataloader.batch_size
        else:
            gradient_accumulation = self.cfg_trainer.gradient_accumulation
        self.iters_to_accumulate = gradient_accumulation // self.train_dataloader.batch_size

        epoch_len = self.cfg_trainer.get("epoch_len")
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.log_step = self.cfg_trainer.get("log_step", 50) * self.iters_to_accumulate

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = self.cfg_trainer.save_period  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get("monitor", "off")  # format: "mnt_mode mnt_metric"

        self.val_step = self.cfg_trainer.get("val_step", 1)

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=writer,
        )

        mixed_precision = self.cfg_trainer.get("mixed_precision", "float32")
        if mixed_precision != "float32":
            self.torchscript = False
            if mixed_precision == "float16":
                self.mixed_precision = torch.float16
            elif mixed_precision == "bfloat16":
                self.mixed_precision = torch.bfloat16
            else:
                raise NotImplementedError()
        else:
            self.mixed_precision = torch.float32
        
        self.grad_scaler = torch.amp.GradScaler(
            self.device, enabled=self.mixed_precision is not torch.float32
        )

        self.torchscript = self.cfg_trainer.get("ts_compile", False)

        optimizer_sd, lr_scheduler_sd = None, None
        # define checkpoint dir and init everything if required
        if self.cfg_trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / self.cfg_trainer.resume_from
            optimizer_sd, lr_scheduler_sd = self._resume_checkpoint(resume_path)
        elif self.cfg_trainer.get("from_pretrained") is not None:
            self._from_pretrained(self.cfg_trainer.get("from_pretrained"))
            if self.torchscript:
                self.model = torch.jit.script(self.model_)
            else:
                self.model = self.model_
        else:
            if self.torchscript:
                self.model = torch.jit.script(self.model_)
            else:
                self.model = self.model_

        self.scheduler_config = self.config.get("scheduler_config", None)
        self._initialize_optimizer(optimizer_sd, lr_scheduler_sd)

    def _initialize_optimizer(self, optimizer_sd, lr_scheduler_sd):
        grouped_trainable_params = get_optimizer_grouped_parameters(
            self.model, self.config.optimizer.weight_decay
        )
        optimizer_cls = get_class(self.config.optimizer.cls)
        self.optimizer = optimizer_cls(
            grouped_trainable_params, **self.project_config["optimizer"]["optimizer_config"]
        )
        total_steps = (
            ceil(self.epoch_len / self.iters_to_accumulate) * self.cfg_trainer.n_epochs
        )
        lr_scheduler_cls = get_class(self.config.lr_scheduler._target_)
        if has_param(lr_scheduler_cls, "total_steps"):
            self.lr_scheduler = instantiate(
                self.config.lr_scheduler,
                optimizer=self.optimizer,
                total_steps=total_steps,
            )
        else:
            self.lr_scheduler = instantiate(
                self.config.lr_scheduler,
                optimizer=self.optimizer
            )
        if optimizer_sd is not None:
            self.optimizer.load_state_dict(optimizer_sd)
        if lr_scheduler_sd is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_sd)
        
        if self.scheduler_config is not None and self.scheduler_config.get('set_lr', None) is not None:
            new_lr = self.scheduler_config.set_lr
            for pg in self.optimizer.param_groups:
                pg['lr'] = new_lr
            if hasattr(self.lr_scheduler, 'base_lrs'):
                self.lr_scheduler.base_lrs = [new_lr for _ in self.lr_scheduler.base_lrs]
            if hasattr(self.lr_scheduler, '_last_lr'):
                self.lr_scheduler._last_lr = [new_lr for _ in self.lr_scheduler._last_lr]

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, only_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best

            stop_process = False
            if epoch % self.val_step == 0:
                best, stop_process, not_improved_count = self._monitor_performance(
                    logs, not_improved_count
                )

                if epoch % self.save_period == 0 or best:
                    self._save_checkpoint(epoch, save_best=best, only_best=best)

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        logs = {}

        if self.writer is not None:
            self.writer.set_step((epoch - 1) * self.epoch_len)
            self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            # self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                    last_lr = self.lr_scheduler.get_last_lr()[0]
                    self.writer.add_scalar("learning rate", last_lr)
                    self._log_scalars(self.train_metrics)
                    self._log_batch(batch_idx, batch)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.logger.debug(
                    f"Current LR: {last_lr}"
                )
                self._check_model_for_nans()
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

            logs = last_train_metrics

        # Run val/test
        if epoch % self.val_step == 0:
            for part, dataloader in self.evaluation_dataloaders.items():
                val_logs = self._evaluation_epoch(epoch, part, dataloader)
                if self.scheduler_config is not None and self.scheduler_config.monitor_part == part:
                    self.lr_scheduler.step(val_logs[self.scheduler_config.monitor_metric])
                logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                try:
                    batch = self.process_batch(
                        batch,
                        batch_idx,
                        metrics=self.evaluation_metrics,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    if self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        torch.cuda.empty_cache()  # free some memory
                        continue
                    else:
                        raise e
            if self.writer is not None:
                self.writer.set_step(epoch * self.epoch_len, part)
                self._log_scalars(self.evaluation_metrics)
                self._log_batch(batch_idx, batch, part)  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += self.val_step

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop * self.val_step)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            if tensor_for_device in batch:
                if tensor_for_device == "audio_mix" and "get_mix" in self.batch_transforms:
                    continue
                batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is None:
            batch["audio_concat"] = torch.concat([batch["audio_s1"], batch["audio_s2"]], dim=1)
            return batch
        
        used_transforms = set()
        for transform_name in transforms.keys():
            if not transform_name.startswith("get_"):
                if transform_name in batch:
                    batch[transform_name] = transforms[transform_name](batch[transform_name])
                    used_transforms.add(transform_name)
    
        if "get_mix" in transforms:
            batch["audio_mix"], batch["audio_s1"], batch["audio_s2"] = transforms["get_mix"](**batch)

            if "audio_mix" in transforms:
                batch["audio_mix"] = transforms["audio_mix"](batch["audio_mix"])
                used_transforms.add("audio_mix")

        batch["audio_concat"] = torch.concat([batch["audio_s1"], batch["audio_s2"]], dim=1)

        if "get_spectrogram" in transforms:
            batch["spectrogram_mix"] = transforms["get_spectrogram"](batch["audio_mix"])

        for transform_name in transforms.keys():
            if not transform_name.startswith("get_") and transform_name not in used_transforms:
                batch[transform_name] = transforms[transform_name](batch[transform_name])

        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        max_grad_norm = self.cfg_trainer.get("max_grad_norm", None)
        if max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), max_grad_norm)

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) > 0:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
                norm_type,
            )
        else:
            return -1
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        if self.torchscript:
            model_state_dict = self.model_.state_dict()
        else:
            model_state_dict = self.model._orig_mod.state_dict() if getattr(self.model_, "_orig_mod", None) is not None else self.model.state_dict()
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if only_best:
            filename = str(self.checkpoint_dir / "model_best.pth")
            if save_best:
                best_path = filename
                torch.save(state, best_path)
                if self.writer is not None and self.config.writer.log_checkpoints:
                    self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
                self.logger.info("Saving current best: model_best.pth ...")
        else:
            filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
            torch.save(state, filename)
            if self.writer is not None and self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        return filename

    def _check_model_for_nans(self):
        has_nan = any(torch.isnan(p).any() for p in self.model_.parameters())
        has_inf = any(torch.isinf(p).any() for p in self.model_.parameters())
        if has_nan or has_inf:
            self.logger.debug(f"Model weights: have_nan={has_nan}, have_inf={has_inf}")
            for name, p in self.model_.named_parameters():
                n_nans = torch.isnan(p).sum().item()
                n_infs = torch.isinf(p).sum().item()
                self.logger.debug(f"{name}: NaNs={n_nans}, Infs={n_infs}, shape={tuple(p.shape)}")

    def _check_sd_for_nans(self, key, sd):
        bad_tensors = []
        if not isinstance(sd, dict) and torch.is_tensor(sd):
            n_nans = int(torch.isnan(sd).sum().item())
            n_infs = int(torch.isinf(sd).sum().item())
            bad_tensors.append((key, "-", sd.shape, sd.dtype, n_nans, n_infs))
        elif isinstance(sd, dict):
            total = 0
            total_nans = 0
            total_infs = 0
            for k, t in sd.items():
                if not torch.is_tensor(t):
                    continue
                n_nans = int(torch.isnan(t).sum().item())
                n_infs = int(torch.isinf(t).sum().item())
                total += t.numel()
                total_nans += n_nans
                total_infs += n_infs
                if n_nans or n_infs:
                    bad_tensors.append((key, k, t.shape, t.dtype, n_nans, n_infs))
        
        return bad_tensors

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, weights_only=False, map_location=self.device)

        bad_tensors = []
        for sd_k in checkpoint:
            bad_tensors.extend(self._check_sd_for_nans(sd_k, checkpoint[sd_k]))

        if len(bad_tensors) == 0:
            self.logger.debug("All tensors were loaded without an issue")
        else:
            self.logger.debug("Bad tensors (state dict name, name, shape, dtype, #NaNs, #Infs):")
            for row in bad_tensors:
                self.logger.debug(row)

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        else:
            if getattr(self.model_, "_orig_mod", None) is not None:
                self.model_._orig_mod.load_state_dict(checkpoint["state_dict"])
            else:
                self.model_.load_state_dict(checkpoint["state_dict"])
            self._check_model_for_nans()
            if self.torchscript:
                self.model = torch.jit.script(self.model_)
            else:
                self.model = self.model_

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"] != self.config["optimizer"]:
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

        return checkpoint["optimizer"], checkpoint["lr_scheduler"]

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        self.model_.to("cpu")

        if checkpoint.get("state_dict") is not None:
            self.model_.load_state_dict(checkpoint["state_dict"])
        else:
            self.model_.load_state_dict(checkpoint)
        self.model_.to(self.device)
