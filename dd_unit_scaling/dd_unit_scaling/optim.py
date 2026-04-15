# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

"""MuP-aware optimizer with FSDP2 support.

Mirrors ``unit_scaling.optim`` but adds metadata caching by parameter name
so MuP fan-in/fan-out values survive FSDP2 sharding (which replaces parameter
tensors with DTensors).
"""

from typing import Any

import torch


try:
    from dion.newton_schulz_triton import ns_line_1, ns_line_2
except ImportError:
    ns_line_1 = ns_line_2 = None

from unit_scaling.optim import (
    _get_fan_in as _get_fan_in_base,
    lr_scale_for_depth,
    scaled_parameters,
)


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


def create_dion2(
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
):
    """Factory that returns a Dion2 optimizer with u-MuP LR scaling.

    Dion2 uses submatrix selection instead of power iteration (faster per-step).
    Reference: https://arxiv.org/abs/2512.16928
    """
    from dion import Dion2

    params = scaled_parameters(
        params,
        _lr_scale_func_muon,
        lr=lr,
        weight_decay=weight_decay,
        independent_weight_decay=independent_weight_decay,
        allow_non_unit_scaling_params=allow_non_unit_scaling_params,
    )

    return Dion2(
        params,
        distributed_mesh=distributed_mesh,
        lr=lr,
        fraction=fraction,
        ef_decay=ef_decay,
        betas=betas,
        weight_decay=weight_decay,
        epsilon=epsilon,
        adjust_lr="spectral_norm",
        use_triton=True,
    )


def create_normuon(
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
    use_polar_express: bool = False,
):
    """Factory that returns a NorMuon optimizer with u-MuP LR scaling.

    Uses the optimized mega-batched NorMuon implementation for better performance.

    NorMuon combines orthogonalization with neuron-wise adaptive learning rates.
    Reference: https://arxiv.org/abs/2510.05491

    Args:
        nesterov: Use Nesterov momentum (recommended for sign gradients).
        cautious_wd: Apply weight decay only where update and parameter signs
                     align (recommended for sign gradients like pinball loss).
        use_polar_express: If True, use Triton-accelerated Polar Express for
                          orthogonalization instead of Newton-Schulz.
                          Comparable speed with better numerical accuracy.
    """
    from dion import NorMuon

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

    # Select orthogonalization function
    newton_schulz_func = None
    use_triton = False

    if use_polar_express:
        # Use Triton-accelerated Polar Express orthogonalization
        newton_schulz_func = polar_express_triton
        use_triton = False  # We're providing custom func, don't use default Triton
    else:
        # Use default Newton-Schulz with Triton
        use_triton = True

    return NorMuon(
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
        adjust_lr="spectral_norm",
        use_triton=use_triton,
        newton_schulz_func=newton_schulz_func,
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


# ── Polar Express orthogonalization ─────────────────────────────────────────
# Alternative to Newton-Schulz for NorMuon optimizer with better numerical accuracy.
# Reference: https://arxiv.org/abs/2505.16932
# Used in: https://github.com/karpathy/nanochat

# Polar Express coefficients (higher order polynomial approximation)
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def polar_express_triton(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """
    Triton-accelerated Polar Express orthogonalization (5 iterations).

    Uses the same symmetric matrix multiplication kernels as Newton-Schulz,
    but with Polar Express coefficients for better accuracy.

    Args:
        G: Input matrix to orthogonalize, shape [..., M, N]
        epsilon: Small value to avoid division by zero

    Returns:
        Orthogonalized matrix with same shape as G

    Example:
        from dion.normuon import NorMuon
        optimizer = NorMuon(..., newton_schulz_func=polar_express_triton)
    """
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize to ensure spectral norm is at most 1
    # Using 1.01 multiplier for slightly more aggressive normalization
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + epsilon)

    # Allocate buffers for intermediate results
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Select batched or non-batched matmul
    line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform Polar Express iterations using Triton-accelerated kernels
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        ns_line_1(X, out=A)  # A = X @ X.mT (symmetric, exploits symmetry in Triton)
        ns_line_2(
            A, alpha=c, beta=b, out=B
        )  # B = b * A + c * A @ A (symmetric quadratic)
        line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X (standard matmul)
        X, C = C, X  # Swap references to avoid copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(fullgraph=True)
def polar_express(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """
    Reference Polar Express implementation without Triton (for comparison/fallback).

    This is the pure PyTorch version that can run without Triton.
    For production use, prefer polar_express_triton() which is 2-3x faster.

    Args:
        G: Input matrix to orthogonalize, shape [..., M, N]
        epsilon: Small value to avoid division by zero

    Returns:
        Orthogonalized matrix with same shape as G
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + epsilon)

    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
