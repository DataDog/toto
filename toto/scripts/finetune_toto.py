# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import argparse
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from datasets import load_dataset
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from toto.data.datamodule.finetune_datamodule import FinetuneDataModule
from toto.model.lightning_module import TotoForFinetuning
from toto.model.toto import Toto

DEFAULT_MAX_ROWS = 100


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def init_lightning(config: Dict[str, Any]) -> Tuple[TotoForFinetuning, int]:
    # Seed
    seed = int(config.get("seed", 42))
    seed_everything(seed, workers=True)

    # Backbone
    model_id = config.get("pretrained_model", "Datadog/Toto-Open-Base-1.0")
    pretrained_backbone = Toto.from_pretrained(model_id).model
    patch_size = getattr(pretrained_backbone.patch_embed, "patch_size", 16)

    # LightningModule params
    mcfg = config.get("model", {})
    dcfg = config.get("data", {})

    lightning_module = TotoForFinetuning(
        pretrained_backbone=pretrained_backbone,
        val_prediction_len=int(mcfg.get("val_prediction_len", 96)),
        stable_steps=int(mcfg.get("stable_steps", 1000)),
        decay_steps=int(mcfg.get("decay_steps", 1000)),
        warmup_steps=int(mcfg.get("warmup_steps", 200)),
        lr=float(mcfg.get("lr", 1e-4)),
        min_lr=float(mcfg.get("min_lr", 1e-5)),
        causal_robust_lambda=float(mcfg.get("causal_robust_lambda", 0.0)),
        causal_robust_alpha=float(mcfg.get("causal_robust_alpha", 1e-4)),
        causal_robust_eps=float(mcfg.get("causal_robust_eps", 1e-6)),
        causal_robust_max_penalty=float(mcfg.get("causal_robust_max_penalty", 20.0)),
        add_exogenous_features=bool(dcfg.get("add_exogenous_features", False)),
    )

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    lightning_module.to(device)
    return lightning_module, patch_size


def get_datamodule(
    config: Dict[str, Any],
    patch_size: int,
    custom_dataset: dict,
    setup: bool = False,
) -> FinetuneDataModule:
    """
    Construct and optionally initialize a `FinetuneDataModule` for model finetuning.

    This helper resolves datamodule hyperparameters from a configuration dictionary,
    infers the context length when not explicitly provided, and wires together dataset
    components (targets, transforms, exogenous features) into a fully specified
    `FinetuneDataModule` instance. It is typically used during finetuning or adaptation
    stages where a model requires sliding-window construction, patching, and optional
    exogenous conditioning.

    Parameters
    ----------
    config : Dict[str, Any]
        The full experiment or training configuration. The function expects a `"data"`
        sub-dictionary containing datamodule parameters such as:
          - ``context_length`` (int): Optional fixed context length.
          - ``context_factor`` (int): If no context length is provided, defines
            ``context_length = context_factor * patch_size`` (default: 16×).
          - ``train_batch_size`` (int): Batch size for training (default: 4).
          - ``val_batch_size`` (int): Batch size for validation (default: 1).
          - ``num_workers`` (int): Number of dataloader workers (default: 0).
          - ``num_train_samples`` (int): Number of training windows to sample
            (default: 1).
          - ``add_exogenous_features`` (bool): Whether to include exogenous variables
            in the model inputs.
          - ``prediction_horizon`` (int): Prediction horizon for the model.
          - ``max_rows`` (int): Maximum number of rows to sample from the dataset.

    patch_size : int
        Patch size used by the model's patching/embedding mechanism. When
        ``context_length`` is not explicitly defined, it is inferred as
        ``context_factor * patch_size``.

    custom_dataset : dict
        A dictionary describing the dataset and its associated fields and transforms.
        It must contain:
          - ``"dataset"``: The underlying dataset object.
          - ``"target_fields"``: Names of target variables to extract.
          - ``"target_transform_fns"``: Preprocessing or normalization functions for
            each target field.
          - ``"ev_fields"``: Names of exogenous variables (if any).
          - ``"ev_transform_fns"``: Transform functions applied to exogenous variables.

    setup : bool, optional
        If ``True``, immediately invokes ``dm.setup(None)`` so that the datamodule
        is fully initialized at creation time. If ``False`` (default), initialization
        is deferred to PyTorch Lightning's lifecycle (e.g., inside each worker/node
        during distributed finetuning).

    Returns
    -------
    FinetuneDataModule
        A configured datamodule instance ready for use in finetuning, optionally
        pre-initialized depending on ``setup``.

    Notes
    -----
    - This function abstracts the common logic for computing context lengths and
      connecting dataset components into the finetuning pipeline.
    - It is designed to support Lightning-based multi-node finetuning workflows,
      where initialization may need to occur independently on each node.
    """
    dcfg = config.get("data", {})

    # If context_length not provided, use factor * patch_size (default 16x)
    if "max_context_length" in dcfg:
        max_context_length = int(dcfg["max_context_length"])
    else:
        context_factor = int(dcfg.get("context_factor", 8))
        max_context_length = context_factor * patch_size

    train_batch_size = int(dcfg.get("train_batch_size", 4))
    val_batch_size = int(dcfg.get("val_batch_size", 1))
    num_workers = int(dcfg.get("num_workers", 0))
    num_train_samples = int(dcfg.get("num_train_samples", 1))
    add_exogenous_features = bool(dcfg.get("add_exogenous_features", False))
    prediction_horizon = int(dcfg.get("prediction_horizon", 64))

    dataset = custom_dataset["dataset"]
    target_fields = custom_dataset["target_fields"]
    target_transform_fns = custom_dataset["target_transform_fns"]
    ev_fields = custom_dataset["ev_fields"]
    ev_transform_fns = custom_dataset["ev_transform_fns"]

    max_rows = int(dcfg.get("max_rows", DEFAULT_MAX_ROWS))

    dm = FinetuneDataModule(
        dataset=dataset,
        max_context_length=max_context_length,
        patch_size=patch_size,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_windows=None,
        num_workers=num_workers,
        num_train_samples=num_train_samples,
        add_exogenous_features=add_exogenous_features,
        target_fields=target_fields,
        target_transform_fns=target_transform_fns,
        ev_fields=ev_fields,
        ev_transform_fns=ev_transform_fns,
        prediction_horizon=prediction_horizon,
        max_rows=max_rows,
    )
    if setup:
        dm.setup(None)
    return dm


def train(
    lightning_module: TotoForFinetuning, datamodule: FinetuneDataModule, config: Dict[str, Any]
) -> Tuple[TotoForFinetuning, str | None, float | None]:
    tcfg = config.get("trainer", {})
    lcfg = config.get("logging", {})

    # -------------------
    # Progress bar callback
    # -------------------
    callbacks: list[Callback] = [TQDMProgressBar(refresh_rate=int(tcfg.get("refresh_rate", 1)))]

    # -------------------
    # Checkpoint callback (optional, controlled via config)
    # -------------------
    cckpt = config.get("checkpoint", {})

    # Only add ModelCheckpoint if any of the relevant options are present
    use_checkpoint = "dirpath" in cckpt.keys()
    checkpoint_callback: ModelCheckpoint | None = None

    if use_checkpoint:
        # check if checkpoint directory already exists if yes then add a suffix to the directory name
        if os.path.exists(cckpt.get("dirpath", "checkpoints")):
            cckpt["dirpath"] = str(cckpt["dirpath"]) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        monitor = cckpt.get("monitor")
        mode = cckpt.get("mode")
        save_top_k_cfg = cckpt.get("save_top_k")

        dirpath = str(cckpt.get("dirpath", "checkpoints"))
        filename = str(cckpt.get("filename", "{epoch}-{step}-{val_loss:.4f}"))

        # Decide checkpointing schedule
        every_n_train_steps = cckpt.get("every_n_train_steps", None)
        every_n_epochs = cckpt.get("every_n_epochs", None)
        train_time_interval_minutes = cckpt.get("train_time_interval_minutes", None)

        if every_n_train_steps is None and every_n_epochs is None and train_time_interval_minutes is None:
            # Set the checkpoint saving after each validation check
            save_on_train_epoch_end = False

        # Convert to proper types or None
        every_n_train_steps = int(every_n_train_steps) if every_n_train_steps is not None else None
        every_n_epochs = int(every_n_epochs) if every_n_epochs is not None else None
        train_time_interval = (
            timedelta(minutes=float(train_time_interval_minutes)) if train_time_interval_minutes is not None else None
        )

        # --- Default behavior when nothing is specified: save ALL checkpoints ---
        if monitor is None and mode is None and save_top_k_cfg is None:
            # "Just save everything"
            monitor_arg = None  # no ranking metric
            mode_arg = "min"  # ignored when monitor=None
            save_top_k_arg = -1  # -1 = save all checkpoints
        else:
            # User configured something -> respect it, with sensible defaults
            monitor_arg = monitor or "val_loss"
            mode_arg = mode or "min"
            save_top_k_arg = int(save_top_k_cfg) if save_top_k_cfg is not None else 1

        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,  # e.g. "{epoch}-{step}-{val_loss:.4f}"
            monitor=monitor_arg,
            mode=mode_arg,
            save_top_k=save_top_k_arg,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=every_n_epochs,
            train_time_interval=train_time_interval,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        callbacks.append(checkpoint_callback)

    # -------------------
    # TensorBoard logger
    # -------------------
    tb_logger = TensorBoardLogger(
        save_dir=str(lcfg.get("save_dir", "lightning_logs")),
        name=str(lcfg.get("name", "toto_finetuning")),
    )

    # -------------------
    # Trainer kwargs (including validation scheduling)
    # -------------------
    trainer_kwargs: Dict[str, Any] = dict(
        max_steps=int(tcfg.get("max_steps", 3000)),
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 1)),
        num_sanity_val_steps=int(tcfg.get("num_sanity_val_steps", 0)),
        enable_progress_bar=bool(tcfg.get("enable_progress_bar", True)),
        val_check_interval=tcfg.get("val_check_interval", None),
        check_val_every_n_epoch=tcfg.get("check_val_every_n_epoch", None),
        callbacks=callbacks,
        logger=tb_logger,
    )

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule=datamodule)

    # Extract best checkpoint info from the ModelCheckpoint callback
    best_ckpt_path: str | None = None
    best_score: float | None = None
    if checkpoint_callback is not None:
        best_ckpt_path = checkpoint_callback.best_model_path or None
        best_score = (
            float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None
        )

    return lightning_module, best_ckpt_path, best_score


def load_finetuned_toto(
    model_id: str,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> TotoForFinetuning:
    """
    Load a finetuned Toto model from a checkpoint file.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the pretrained Toto backbone
        (e.g., "Datadog/Toto-Open-Base-1.0").
    checkpoint_path : str
        Path to the Lightning checkpoint file (.ckpt).
    map_location : str | torch.device, default="cpu"
        Device to map the checkpoint tensors to.

    Returns
    -------
    TotoForFinetuning
        The loaded and eval-ready finetuned model.
    """
    # Load base Toto backbone from HuggingFace
    pretrained_backbone = Toto.from_pretrained(model_id).model

    # Load Lightning module from checkpoint
    model = TotoForFinetuning.load_from_checkpoint(  # type: ignore[operator]
        checkpoint_path=checkpoint_path,
        pretrained_backbone=pretrained_backbone,
        map_location=map_location,
    )
    model.eval()
    return model
