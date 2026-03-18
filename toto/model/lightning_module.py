# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm
from torch.optim import AdamW

from ..data.util.dataset import CausalMaskedTimeseries, MaskedTimeseries
from ..model.backbone import TotoBackbone, TotoOutput
from .losses import CombinedLoss
from .scheduler import WarmupStableDecayLR


class TotoForFinetuning(L.LightningModule):
    """
    PyTorch LightningModule for fine-tuning Toto model.

    This class orchestrates all training and validation logic, optimizer and learning rate scheduler
    setup, as well as advanced logging (e.g., gradient norm logging). It is specifically designed to be
    used in Toto model fine-tuning scenarios, integrating robust time series loss functions and
    supporting advanced optimizer options.

    Parameters
    ----------
    val_prediction_len : int, default=96
        Number of timesteps to produce during validation (affects mask and metrics).
    stable_steps : int, default=1000
        Number of steps for stable learning rate.
    decay_steps : int, default=1000
        Number of steps for decay learning rate.
    warmup_steps : int, default=200
        Number of steps for warmup learning rate.
    lr : float, default=1e-4
        Initial learning rate.
    min_lr : float, default=1e-5
        Minimum learning rate after warmup.
    betas : tuple[float, float], default=(0.9, 0.999)
        Adam/AdamW optimizer beta parameters.
    weight_decay : float, default=0.01
        Optimizer weight decay.
    pretrained_backbone : TotoBackbone | None, default=None
        Optionally take an existing backbone; otherwise, constructed from kwargs.
    add_exogenous_features : bool, default=False
        Whether to add exogenous features to the model.
    **model_kwargs : Any
        Extra model-building args if not using a pretrained backbone.

    Methods
    -------
    on_before_optimizer_step(_):
        Optionally logs gradient norm(s) after optimizer step.
    configure_optimizers():
        Sets up the optimizer(s) and LR scheduler using configured settings.
    _prediction_mask(padding_mask):
        Builds the mask selecting which future timesteps to predict during validation.
    _get_inputs(x):
        Prepares series tensor and masks for the model's forward pass.
    forward(x):
        Gets model outputs (distribution, location, scale) for input batch.
    _train_or_val_step(batch, is_train):
        Shared logic for training and validation steps: computes loss, reduction, logging.
    training_step(batch, _batch_idx):
        PyTorch Lightning hook for backprop step.
    validation_step(batch, _batch_idx):
        PyTorch Lightning hook for evaluation/metrics.
    """

    def __init__(
        self,
        # training/validation hyperparameters
        val_prediction_len: int = 96,
        stable_steps: int = 1000,
        decay_steps: int = 1000,
        warmup_steps: int = 200,
        lr: float = 1e-4,
        min_lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        causal_robust_lambda: float = 0.0,
        causal_robust_alpha: float = 1e-4,
        causal_robust_eps: float = 1e-6,
        causal_robust_max_penalty: float = 20.0,
        # backbone construction
        pretrained_backbone: TotoBackbone | None = None,
        add_exogenous_features: bool = False,
        **model_kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_backbone"])

        # Build public backbone (requires explicit kwargs)
        if pretrained_backbone is not None:
            self.model = pretrained_backbone
        else:
            self.model = TotoBackbone(**model_kwargs)

        if add_exogenous_features:
            self.model.enable_variate_labels()

        # Training configuration setup
        self.lr = lr
        self.min_lr = min_lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.val_prediction_len = val_prediction_len
        self.causal_robust_lambda = causal_robust_lambda

        if self.causal_robust_lambda > 0:
            self.model.configure_causal_robustness(
                enabled=True,
                alpha=causal_robust_alpha,
                eps=causal_robust_eps,
                max_penalty=causal_robust_max_penalty,
            )

        # Loss setup (currently hard-wired to CombinedLoss)
        self.combined_loss = CombinedLoss()

    def configure_optimizers(self):
        """
        Sets up optimizer and learning rate scheduler.

        Returns
        -------
        dict
            PyTorch Lightning optimizer and scheduler configuration
        """
        decay_params = [param for param in self.parameters() if param.requires_grad]

        # Instantiate optimizer based on selection
        optimizer = AdamW(
            decay_params,
            lr=self.lr,
            betas=self.betas,
            eps=1e-7,
            weight_decay=self.weight_decay,
        )

        # Set up learning rate scheduler
        lr_scheduler = WarmupStableDecayLR(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            stable_steps=self.stable_steps,
            decay_steps=self.decay_steps,
            min_lr=self.min_lr,
            base_lr=self.lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _prediction_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Build prediction mask for validation: selects the most recent val_prediction_len timesteps
        in each sample, only where padding_mask is not zero.


        Parameters
        ----------
        padding_mask : torch.Tensor
            Binary mask with shape (..., seq_len)

        Returns
        -------
        torch.Tensor
            Mask of prediction timesteps for current batch/sample
        """
        pred_len = min(self.val_prediction_len, padding_mask.shape[-1])
        mask = torch.zeros_like(padding_mask, dtype=torch.bool)
        if pred_len > 0:
            mask[..., -pred_len:] = True
        return mask & padding_mask.bool()

    def _get_inputs(
        self, x: CausalMaskedTimeseries | MaskedTimeseries
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepares input slices and masks to feed to the backbone.

        Parameters
        ----------
        x : CausalMaskedTimeseries or MaskedTimeseries
            Input object containing series, padding, and id masks.

        Returns
        -------
        tuple
            Tuple of inputs (series, padding_mask, id_mask, num_exogenous_variables) for the model.
        """
        if isinstance(x, CausalMaskedTimeseries):
            input_slice = x.input_slice
            return (
                x.series[..., input_slice],
                x.padding_mask[..., input_slice],
                x.id_mask[..., input_slice],
                x.num_exogenous_variables,
            )
        else:
            return x.series, x.padding_mask, x.id_mask, x.num_exogenous_variables

    def forward(self, x: MaskedTimeseries | CausalMaskedTimeseries) -> TotoOutput:
        """
        Performs a forward pass of the model.

        Parameters
        ----------
        x : MaskedTimeseries or CausalMaskedTimeseries
            Input batch object (batched time series with masks).

        Returns
        -------
        TotoOutput
            Model output, including distribution, loc, and scale tensors.
        """
        inputs, input_padding_mask, id_mask, num_exogenous_variables = self._get_inputs(x)
        return self.model(
            inputs,
            input_padding_mask,
            id_mask,
            num_exogenous_variables=num_exogenous_variables,
        )

    def _train_or_val_step(self, batch: CausalMaskedTimeseries, is_train: bool) -> torch.Tensor:
        """
        Common step logic for both training and validation.

        Computes loss, applies correct mask (training vs validation),
        and logs mean loss for summary/reduction.

        Parameters
        ----------
        batch : CausalMaskedTimeseries
            Batch data (input series, masks, slices)
        is_train : bool
            Flag for train (True) vs validation (False)

        Returns
        -------
        torch.Tensor
            Mean loss for this batch/sample (for backward or logging)
        """
        eps = torch.finfo(batch.series.dtype).eps
        target_slice = batch.target_slice
        targets = batch.series[..., target_slice]
        targets_padding_mask = batch.padding_mask[..., target_slice]

        out = self(batch)
        distr, loc, scale = out.distribution, out.loc, out.scale
        # Normalize targets by model loc and scale for distributional prediction
        scaled_targets = (targets - loc) / (scale + eps)

        # Select target mask: during training we use padding mask, for validation use prediction mask (only focuses on val_prediction_len timesteps)
        if not is_train:
            mask = self._prediction_mask(targets_padding_mask)
        else:
            mask = targets_padding_mask

        # Mask out the exogenous variables from the loss computation
        if batch.num_exogenous_variables > 0:
            mask[..., -batch.num_exogenous_variables :] = False

        # Compute loss only on masked (valid) target positions
        # Loss is computed using the same parameters used during pretraining of Toto
        loss = self.combined_loss(distr, scaled_targets) * mask

        # Aggregate loss for distributed-safe reduction
        valid_count = mask.sum()
        total_sum = loss.sum()

        # Multi-GPU distributed reduction for proper mean/logging support
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(valid_count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_sum, op=torch.distributed.ReduceOp.SUM)

        mean_total = total_sum / (valid_count + eps)
        prefix = "train" if is_train else "val"

        if self.causal_robust_lambda > 0 and self.model.latest_causal_robustness_penalty is not None:
            causal_penalty = self.model.latest_causal_robustness_penalty
            weighted_causal_penalty = self.causal_robust_lambda * causal_penalty
            mean_total = mean_total + weighted_causal_penalty
            self.log(
                f"{prefix}_causal_robustness_penalty",
                causal_penalty.detach(),
                prog_bar=False,
                on_step=is_train,
                on_epoch=True,
                batch_size=batch.series.shape[0],
            )
            self.log(
                f"{prefix}_causal_robustness_penalty_weighted",
                weighted_causal_penalty.detach(),
                prog_bar=False,
                on_step=is_train,
                on_epoch=True,
                batch_size=batch.series.shape[0],
            )

        # Log loss for lightning progress bar and metrics
        self.log(
            f"{prefix}_loss",
            mean_total,
            prog_bar=True,
            on_step=is_train,
            on_epoch=True,
            batch_size=batch.series.shape[0],
        )
        return mean_total

    def training_step(self, batch: CausalMaskedTimeseries, _batch_idx: int):
        """
        Lightning hook for training step.
        """
        return self._train_or_val_step(batch, is_train=True)

    def validation_step(self, batch: CausalMaskedTimeseries, _batch_idx: int):
        """
        Lightning hook for validation step.
        """
        return self._train_or_val_step(batch, is_train=False)
