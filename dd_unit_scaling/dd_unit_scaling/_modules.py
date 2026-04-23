# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""World-size-aware unit-scaled nn.Module wrappers.

Mirrors ``unit_scaling._modules`` but uses our compile-friendly functional
implementations and accounts for DDP/FSDP world size.
"""

import math
from typing import Any, Optional, Tuple

import torch
import unit_scaling as uu

from .functional import linear, per_dim_scale, rms_norm, softplus
from .scale import scale_bwd


class Linear(torch.nn.Linear):
    """World-size-aware unit-scaled Linear layer.

    Configurable via:
    - scale_power: (output, grad_input, grad_weight) scaling exponents
      - (0.5, 0.5, 0.5): standard hidden layer (default)
      - (1.0, 0.5, 0.5): readout layer (1/fan_in output scaling)
    - constraint: "to_output_scale" matches grad_input_scale to output_scale
    - weight_mup_type: "weight", or "output" for LR scaling

    Common configurations:
    - Hidden: Linear() - default, constraint ties grad_input to output
    - Output: Linear(scale_power=(1.0, 0.5, 0.5), constraint=None, weight_mup_type="output")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = "to_output_scale",
        weight_mup_type: str = "weight",
        scale_power: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.constraint = constraint
        self.scale_power = scale_power
        self.weight = uu.Parameter(self.weight.data, mup_type=weight_mup_type)
        if self.bias is not None:
            self.bias = uu.Parameter(self.bias.data, mup_type="bias")

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear(input, self.weight, self.bias, self.constraint, self.scale_power)


class LinearReadout(Linear):
    """World-size-aware unit-scaled LinearReadout layer (final output projection).

    Uses scale_power=(1.0, 0.5, 0.5) per u-μP:
    - Forward: output_scale = 1/fan_in (not 1/√fan_in)
    - Backward: grad_input_scale = 1/√fan_out (independent)
    - MuP: mup_type="output" gives lr_factor=1.0

    Only use for final output projections, not hidden layers or skip connections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = None,
        weight_mup_type: str = "output",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
            constraint=constraint,
            weight_mup_type=weight_mup_type,
            scale_power=(1.0, 0.5, 0.5),  # Readout layer scaling
        )

    # Uses parent's forward() which handles scale_power


class RMSNorm(torch.nn.RMSNorm):
    """World-size-aware unit-scaled RMSNorm.

    Inherits from torch.nn.RMSNorm so that FSDP2 leaf-module detection via
    isinstance(m, torch.nn.RMSNorm) works correctly for sharding.

    Args:
        normalized_shape: The normalization dimension
        eps: Epsilon for numerical stability
        elementwise_affine: Legacy kwarg for compatibility (use include_weight instead)
        include_weight: Controls weight parameters:
            - False: no weight parameters
            - True: full dimensional weight
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = False,
        include_weight: Optional[bool] = None,
    ) -> None:
        weight_mode = (
            include_weight if include_weight is not None else elementwise_affine
        )

        super().__init__(normalized_shape, eps=eps, elementwise_affine=bool(weight_mode))
        self.dim = (
            normalized_shape
            if isinstance(normalized_shape, int)
            else normalized_shape[0]
        )

        # Override self.weight set by parent with uu.Parameter to carry mup_type metadata.
        if weight_mode:
            self.weight = uu.Parameter(torch.ones(self.dim), mup_type="norm")
        else:
            self.weight = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rms_norm(
            input,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            eps=self.eps,
        )


class PerDimScale(torch.nn.Module):
    """Learned per-dimension scaling with unit-scaled gradients.

    Each dimension gets an independent positive scale factor parametrized
    via softplus so the factor is always > 0.  At init (params=0), all
    factors = 1.0 (identity).

    Gradient of the learned parameter is scaled by sqrt(D / global_numel)
    (same pattern as RMSNorm weight) so that parameter updates remain
    O(1) regardless of batch size.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.per_dim_scale = uu.Parameter(torch.zeros(dim), mup_type="norm")

    # Compensates for softplus backward (4.8x) and /log(2) (1.44x) chain
    # amplifying param gradients by 3.46x at init (params=0).
    # = log(2) / (softplus_grad_input_scale * sigmoid(0))
    # = log(2) / ((1/0.20833) * 0.5)
    _param_grad_compensation: float = math.log(2.0) / (0.5 / 0.20833444)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        p = scale_bwd(self.per_dim_scale, self._param_grad_compensation)
        r = (softplus(p) / math.log(2.0)).to(input.dtype)
        return per_dim_scale(input, r)
