# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""MuP-aware optimizer with FSDP2 support.

Mirrors ``unit_scaling.optim`` but adds metadata caching by parameter name
so MuP fan-in/fan-out values survive FSDP2 sharding (which replaces parameter
tensors with DTensors).
"""

from typing import Any

import torch
from unit_scaling.optim import _get_fan_in as _get_fan_in_base
from unit_scaling.optim import lr_scale_for_depth, scaled_parameters

# Cache for MuP metadata by parameter NAME (survives FSDP2 sharding).
# This is critical for FSDP2 compatibility — data_ptr() changes after sharding.
_UMUP_METADATA_BY_NAME: dict[str, dict[str, Any]] = {}


def cache_fan_values(named_parameters) -> None:
    """Cache MuP metadata by parameter name before FSDP wrapping.

    This allows metadata to survive FSDP2 sharding which creates new DTensor wrappers.
    Keyed by name instead of data_ptr() because FSDP2 replaces parameter tensors.

    Must be called before FSDP wrapping when param shapes are still correct.
    """
    _UMUP_METADATA_BY_NAME.clear()
    for name, param in named_parameters:
        # Skip parameters without MuP metadata
        if not hasattr(param, "mup_type"):
            continue
        fan_in = _get_fan_in(name, param)
        fan_out = param.shape[0] if param.ndim >= 2 else 1
        _UMUP_METADATA_BY_NAME[name] = {
            "mup_type": param.mup_type,
            "mup_scaling_depth": getattr(param, "mup_scaling_depth", None),
            "fan_in": fan_in,
            "fan_out": fan_out,
        }


def get_cached_metadata(param_name: str) -> dict[str, Any]:
    """Get cached MuP metadata for a parameter by name."""
    return _UMUP_METADATA_BY_NAME.get(param_name, {})


def _get_fan_in(param_name: str, param: torch.Tensor) -> int:
    """Get fan-in, checking cache first then falling back to unit_scaling.

    Args:
        param_name: Parameter name for cache lookup
        param: Parameter tensor (used as fallback if not in cache)

    Returns:
        fan_in value from cache if available, otherwise computed from param shape
    """
    metadata = get_cached_metadata(param_name)
    if "fan_in" in metadata:
        return metadata["fan_in"]
    # Fallback to unit_scaling's _get_fan_in (only safe before FSDP wrapping)
    return _get_fan_in_base(param) if param.ndim >= 2 else 1


def _lr_scale_func_adam(param: torch.Tensor) -> float:
    """Calculate the LR scaling factor for AdamW with FSDP2 support.

    Uses _original_fan_in attribute for correct scaling with FSDP2 sharded DTensors.
    Falls back to computing fan_in from param shape if the cached value is unavailable.

    LR scaling rules (per u-MuP):
    - bias/norm: 1.0
    - weight: 1/√fan_in (standard μP for hidden layers)
    - output: 1.0 (readout layers)
    """
    if not hasattr(param, "mup_type"):
        return 1.0

    mup_type = param.mup_type
    scale = lr_scale_for_depth(param)

    if mup_type in ("bias", "norm", "output"):
        return scale
    elif mup_type == "weight":
        fan_in = getattr(param, "_original_fan_in", None)
        if fan_in is None:
            fan_in = _get_fan_in_base(param) if param.ndim >= 2 else 1
        return scale * fan_in**-0.5
    else:
        return scale


def _lr_scale_func_muon(param: torch.Tensor) -> float:
    """LR scaling for Muon-family optimizers (Muon, NorMuon, Dion2).

    Depth scaling only — no 1/√fan_in. Muon's spectral norm adjustment
    (adjust_lr="spectral_norm") already provides the correct MuP width
    transfer for orthogonal optimizers. Adding 1/√fan_in double-counts,
    causing the update spectral norm to vanish as O(1/√d_model).
    """
    if not hasattr(param, "mup_type"):
        return 1.0
    return lr_scale_for_depth(param)


class Dion2:
    """Dion2 optimizer with u-MuP LR scaling and FSDP2 support.

    Dion2 uses submatrix selection instead of power iteration (faster per-step).
    This wraps the dion.Dion2 optimizer to apply per-parameter learning rate scaling
    following the μP/u-μP parameterization.

    Reference: https://arxiv.org/abs/2512.16928

    Learning rate scaling (per u-MuP):
    - All parameters: lr = base_lr × depth_scale (no 1/√fan_in for orthogonal optimizers)
    - Spectral norm adjustment provides width transfer automatically

    Args:
        params: Model parameters or parameter groups
        lr: Base learning rate (will be scaled per-parameter)
        fraction: Fraction of submatrix to use for orthogonalization
        ef_decay: Error feedback decay rate
        betas: Tuple of (beta1, beta2) for adaptive updates
        weight_decay: Weight decay coefficient
        epsilon: Small value to avoid division by zero
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training
        independent_weight_decay: If True, weight decay is independent of LR
        allow_non_unit_scaling_params: If True, allows parameters without mup_type
        adjust_lr: LR adjustment strategy ("spectral_norm", "rms_norm", or None).
                  Default is "spectral_norm" for u-MuP width transfer.
        use_triton: If True, use Triton-accelerated kernels. Default is True.

    Example:
        optimizer = Dion2(model.parameters(), lr=0.02, weight_decay=0.01)
    """

    def __new__(
        cls,
        params,
        *,
        lr: float = 0.02,
        fraction: float = 0.5,
        ef_decay: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        epsilon: float = 1e-7,
        distributed_mesh=None,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        adjust_lr: str | None = "spectral_norm",
        use_triton: bool = True,
    ):
        from dion import Dion2 as _Dion2

        params = scaled_parameters(
            params,
            _lr_scale_func_muon,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )

        return _Dion2(
            params,
            distributed_mesh=distributed_mesh,
            lr=lr,
            fraction=fraction,
            ef_decay=ef_decay,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            adjust_lr=adjust_lr,
            use_triton=use_triton,
        )


class NorMuon:
    """NorMuon optimizer with u-MuP LR scaling and FSDP2 support.

    NorMuon combines orthogonalization with neuron-wise adaptive learning rates.
    This wraps the dion.NorMuon optimizer to apply per-parameter learning rate scaling
    following the μP/u-μP parameterization.

    Reference: https://arxiv.org/abs/2510.05491

    Learning rate scaling (per u-MuP):
    - All parameters: lr = base_lr × depth_scale (no 1/√fan_in for orthogonal optimizers)
    - Spectral norm adjustment provides width transfer automatically

    Args:
        params: Model parameters or parameter groups
        lr: Base learning rate (will be scaled per-parameter)
        mu: Momentum factor for NorMuon algorithm
        muon_beta2: Second beta parameter for NorMuon's adaptive updates
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms
        weight_decay: Weight decay coefficient
        epsilon: Small value to avoid division by zero
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training
        independent_weight_decay: If True, weight decay is independent of LR
        allow_non_unit_scaling_params: If True, allows parameters without mup_type
        nesterov: Use Nesterov momentum (recommended for sign gradients)
        cautious_wd: Apply weight decay only where update and parameter signs align
                    (recommended for sign gradients like pinball loss)
        adjust_lr: LR adjustment strategy ("spectral_norm", "rms_norm", or None).
                  Default is "spectral_norm" for u-MuP width transfer.
        use_polar_express: If True, use Polar Express orthogonalization instead
                          of Newton-Schulz. Default is True (enabled upstream)
        use_triton: If True, use Triton-accelerated kernels for orthogonalization.
                   Default is True for better performance

    Example:
        optimizer = NorMuon(model.parameters(), lr=0.02, weight_decay=0.01)
    """

    def __new__(
        cls,
        params,
        *,
        lr: float = 0.02,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        epsilon: float = 1e-7,
        distributed_mesh=None,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        nesterov: bool = True,
        cautious_wd: bool = True,
        adjust_lr: str | None = "spectral_norm",
        use_polar_express: bool = True,
        use_triton: bool = True,
    ):
        from dion import NorMuon as _NorMuon

        # Depth-only LR scaling; spectral norm adjustment handles width transfer.
        # Adam params should be routed through a separate uu.AdamW, not passed here.
        params = scaled_parameters(
            params,
            _lr_scale_func_muon,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )

        return _NorMuon(
            params,
            distributed_mesh=distributed_mesh,
            lr=lr,
            mu=mu,
            muon_beta2=muon_beta2,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov,
            cautious_wd=cautious_wd,
            adjust_lr=adjust_lr,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
        )


class AdamW(torch.optim.AdamW):
    """World-size-aware AdamW optimizer with u-MuP support.

    This wraps torch.optim.AdamW to apply per-parameter learning rate scaling
    following the μP/u-μP parameterization.

    Learning rate scaling (per u-MuP):
    - bias/norm parameters: lr = base_lr × 1.0
    - weight parameters: lr = base_lr × 1/√fan_in
    - output parameters: lr = base_lr × 1.0

    Args:
        params: Model parameters or parameter groups
        lr: Base learning rate (will be scaled per-parameter)
        weight_decay: Weight decay coefficient
        independent_weight_decay: If True, weight decay is independent of LR
        allow_non_mup_params: If True, allows parameters without mup_type
        **kwargs: Additional arguments passed to torch.optim.AdamW

    Example:
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        *args,
        weight_decay: float = 0.0,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        **kwargs,
    ) -> None:
        params = scaled_parameters(
            params,
            _lr_scale_func_adam,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )
        # No need to forward {lr, weight_decay}, as each group has these specified
        super().__init__(params, *args, **kwargs)
