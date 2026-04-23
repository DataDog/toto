# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Toto2ModelConfig:
    patch_size: int
    d_model: int
    num_heads: int
    num_layers: int
    layer_group_size: int
    num_variate_layers_per_group: int
    variate_layer_first: bool
    dropout_p: float = 0.0
    norm_eps: float = 5e-5
    attn_bias: bool = False
    mlp_bias: bool = False
    num_output_patches: int = 1
    pre_norm: bool = True
    d_ff: Optional[int] = None
    qk_dim: Optional[int] = None
    v_dim: Optional[int] = None
    num_groups: Optional[int] = None
    heads_per_group: Optional[int] = None
    residual_mult: float = 1.0
    residual_attn_ratio: Optional[float] = None
    qk_norm: bool = True
    norm_include_weight: bool = False
    qk_norm_include_weight: Optional[bool] = None
    per_dim_scale: bool = False
    use_xpos: bool = False

    @staticmethod
    def compute_residual_attn_ratio(context_length: int, patch_size: int) -> float:
        """sqrt(S / log(S)) where S = context_length / patch_size.

        Restores attn/MLP variance balance lost by using unscaled F.sdpa
        instead of unit-scaled sdpa.
        """
        s = context_length / patch_size
        return math.sqrt(s / math.log(s))

    def __post_init__(self):
        if self.dropout_p != 0.0:
            raise ValueError("Non-zero dropout_p is a bad choice here: it causes long-term training instability.")
        if self.d_ff is None:
            self.d_ff = (int(4 * self.d_model * 2 / 3) + 7) // 8 * 8
        if self.qk_norm_include_weight is None:
            self.qk_norm_include_weight = self.norm_include_weight
        if self.residual_attn_ratio is None:
            raise ValueError(
                "residual_attn_ratio must be set explicitly. Use "
                "Toto2ModelConfig.compute_residual_attn_ratio(context_length, patch_size) "
                "to compute it."
            )
        self.num_groups = self.num_groups or self.num_heads
        self.qk_dim = self.qk_dim or self.d_model // self.num_heads
        self.v_dim = self.v_dim or self.qk_dim
        self.heads_per_group = self.num_heads // self.num_groups

        assert self.num_layers % self.layer_group_size == 0, (
            f"num_layers must be divisible by layer_group_size"
            f"got num_layers={self.num_layers} and layer_group_size={self.layer_group_size}"
        )
        assert self.num_heads > 0 and self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert (self.num_heads % self.num_groups == 0) and (self.num_heads >= self.num_groups), (
            f"num_heads ({self.num_heads}) must be divisible by num_groups ({self.num_groups}) and greater than or equal to num_groups ({self.num_groups})"
        )

    # @property
    # def heads_per_group(self) -> int:
    #     return self.num_heads // self.num_groups


@dataclass
class Toto2GluonTSModelConfig:
    prediction_length: int
    context_length: int
    target_dim: int
    past_feat_dynamic_real_dim: int = 0
    feat_dynamic_real_dim: int = 0
    decode_block_size: Optional[int] = None
    has_missing_values: bool = True
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
