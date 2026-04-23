# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Compile-friendly reimplementation of unit_scaling's scale_fwd / scale_bwd.

The original unit_scaling._ScaledGrad.forward uses
    isinstance(bwd_scale, fx.proxy.Proxy)
checks and saves context inside forward(), which breaks torch.compile
(dynamo cannot trace through isinstance checks on proxy objects).

We rewrite it using the setup_context pattern (PyTorch 2.x), which separates
context setup from forward so torch.compile can trace the graph cleanly.
"""

from typing import Any, Tuple

import torch


class _ScaledGrad(torch.autograd.Function):
    """Apply different scales in forward and backward passes (compile-friendly)."""

    @staticmethod
    def forward(X: torch.Tensor, fwd_scale: float, bwd_scale: float) -> torch.Tensor:
        return fwd_scale * X  # type: ignore[return-value]

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple, output: torch.Tensor) -> None:
        _, _, bwd_scale = inputs
        ctx.bwd_scale = bwd_scale

    @staticmethod
    def backward(ctx: Any, grad_Y: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        return ctx.bwd_scale * grad_Y, None, None


def scale_fwd(input: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale a tensor in the forward pass only (gradient is unchanged)."""
    return _ScaledGrad.apply(input, scale, 1.0)  # type: ignore[return-value]


def scale_bwd(input: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale a tensor's gradient in the backward pass only (forward is identity)."""
    return _ScaledGrad.apply(input, 1.0, scale)  # type: ignore[return-value]
