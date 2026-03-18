# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import torch
import pytest

try:
    from toto.data.util.dataset import CausalMaskedTimeseries
    from toto.model.backbone import TotoBackbone
    from toto.model.lightning_module import TotoForFinetuning
except Exception as exc:  # pragma: no cover - environment-specific import guard
    pytest.skip(f"Skipping causal robustness tests due unavailable dependencies: {exc}", allow_module_level=True)


def _make_backbone() -> TotoBackbone:
    return TotoBackbone(
        patch_size=4,
        stride=4,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_hidden_dim=64,
        dropout=0.0,
        spacewise_every_n_layers=2,
        spacewise_first=True,
        use_memory_efficient_attention=False,
        scaler_cls="<class 'model.scaler.CausalPatchStdMeanScaler'>",
        output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"],
    )


def _make_inputs(batch: int = 2, variates: int = 3, timesteps: int = 16):
    series = torch.randn(batch, variates, timesteps)
    padding_mask = torch.ones_like(series, dtype=torch.bool)
    id_mask = torch.zeros_like(series, dtype=torch.long)
    return series, padding_mask, id_mask


def _make_causal_batch(batch: int = 2, variates: int = 3, timesteps: int = 16) -> CausalMaskedTimeseries:
    series = torch.randn(batch, variates, timesteps)
    padding_mask = torch.ones_like(series, dtype=torch.bool)
    id_mask = torch.zeros_like(series, dtype=torch.long)
    timestamp_seconds = torch.arange(timesteps).view(1, 1, timesteps).expand(batch, variates, timesteps).long()
    time_interval_seconds = torch.ones(batch, variates, dtype=torch.long)
    # Input and target lengths are both 12 so that model outputs and targets align.
    input_slice = slice(0, timesteps - 4)
    target_slice = slice(4, timesteps)
    return CausalMaskedTimeseries(
        series=series,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
        input_slice=input_slice,
        target_slice=target_slice,
        num_exogenous_variables=0,
    )


def test_causal_robustness_penalty_disabled_by_default():
    model = _make_backbone().train()
    series, padding_mask, id_mask = _make_inputs()

    _ = model(series, padding_mask, id_mask)
    assert model.latest_causal_robustness_penalty is None


def test_causal_robustness_penalty_enabled_and_finite():
    model = _make_backbone().train()
    model.configure_causal_robustness(
        enabled=True,
        alpha=1e-4,
        eps=1e-6,
        max_penalty=20.0,
    )
    series, padding_mask, id_mask = _make_inputs()

    _ = model(series, padding_mask, id_mask)
    penalty = model.latest_causal_robustness_penalty

    assert penalty is not None
    assert torch.isfinite(penalty)
    assert penalty.item() >= 0.0


def test_lightning_step_adds_weighted_causal_robustness_penalty():
    module = TotoForFinetuning(
        pretrained_backbone=_make_backbone(),
        causal_robust_lambda=1.0,
        causal_robust_alpha=1e-4,
        causal_robust_eps=1e-6,
        causal_robust_max_penalty=20.0,
    ).train()
    batch = _make_causal_batch()

    with torch.no_grad():
        loss_with_penalty = module._train_or_val_step(batch, is_train=True)
        penalty = module.model.latest_causal_robustness_penalty
        module.causal_robust_lambda = 0.0
        loss_without_penalty = module._train_or_val_step(batch, is_train=True)

    assert penalty is not None
    assert penalty.item() >= 0.0
    assert loss_with_penalty >= loss_without_penalty
