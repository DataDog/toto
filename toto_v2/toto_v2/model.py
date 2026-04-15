# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

"""TotoV2: Time series foundation model with MuP scaling.

Standalone, inference-only implementation. All model components in a single file.
"""

import abc
import dataclasses
import functools as ft
import json
import math
import warnings
from pathlib import Path
from typing import Any, Callable, NamedTuple, NotRequired, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    ExpandDimArray,
    Transformation,
    TestSplitSampler,
)
from gluonts.transform.split import InstanceSplitter, TFTInstanceSplitter
from huggingface_hub import load_torch_model, PyTorchModelHubMixin
from jaxtyping import Bool, Float, Int
from safetensors.torch import load_file as load_safetensors_file

import dd_unit_scaling as uu
from dd_unit_scaling import functional as U

from .configuration import TotoV2GluonTSModelConfig, TotoV2ModelConfig


__all__ = [
    "TotoV2Model",
    "TotoV2GluonTSModel",
]




# =====================================================================
# Scaler
# =====================================================================


class PatchedCausalStdScaler(nn.Module):
    """Causal standard-deviation scaler with patch-aware statistics."""

    def __init__(
        self,
        patch_size: int,
        correction: int | float = 1,
        minimum_scale: float = 1e-6,
        # Accepted for API compat, ignored in inference-only build
        stabilize_with_global: bool = False,
        online: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.correction = correction
        self.minimum_scale = minimum_scale

    def forward(
        self,
        data: Float[torch.Tensor, "..."],
        mask: Optional[Bool[torch.Tensor, "..."]] = None,
    ) -> tuple[
        Float[torch.Tensor, "..."],
        Float[torch.Tensor, "..."],
        Float[torch.Tensor, "..."],
    ]:
        try:
            hp = data.to(torch.float64)
        except TypeError:
            warnings.warn(
                f"Float64 not supported on {data.device}, using float32 for scaler.",
                RuntimeWarning,
            )
            hp = data.to(torch.float32)

        if mask is None:
            mask = torch.ones_like(data, dtype=torch.bool)

        loc, scale = self._compute_loc_scale(hp, mask)
        loc, scale = loc.to(data.dtype), scale.to(data.dtype)
        return torch.where(mask, (data - loc) / scale, 0), loc, scale

    def _compute_loc_scale(
        self, data: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Causal (cumulative) mean
        cum_data = (data * mask).cumsum(dim=-1)
        denominator = mask.cumsum(dim=-1).clamp_min(1)
        causal_loc = cum_data / denominator

        # Welford-style causal variance
        prev_loc = torch.cat(
            [torch.zeros_like(causal_loc[..., :1]), causal_loc[..., :-1]], dim=-1
        )
        delta = data - prev_loc
        increment = delta * (data - causal_loc) * mask
        m_2 = torch.cumsum(increment, dim=-1)
        causal_var = m_2 / (denominator - self.correction).clamp(min=1)
        causal_scale = torch.sqrt(causal_var).clamp(min=self.minimum_scale)

        # Patch-aware: use last value in each patch, repeat across patch
        loc = repeat(
            rearrange(
                causal_loc, "... (seq patch) -> ... seq patch", patch=self.patch_size
            )[..., -1],
            "... seq -> ... (seq patch)",
            patch=self.patch_size,
        )
        scale = repeat(
            rearrange(
                causal_scale, "... (seq patch) -> ... seq patch", patch=self.patch_size
            )[..., -1],
            "... seq -> ... (seq patch)",
            patch=self.patch_size,
        )
        return loc, scale


# =====================================================================
# Positional Encodings
# =====================================================================


class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int):
        super().__init__()
        self.proj_width = proj_width

    @abc.abstractmethod
    def forward(
        self,
        x: Float[torch.Tensor, "*shape heads seq dim"],
        seq_ids: Optional[Int[torch.Tensor, "*shape seq"]],
    ) -> Float[torch.Tensor, "*shape heads seq dim"]: ...


class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__(proj_width)
        assert self.proj_width % 2 == 0, f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0 / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float) / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(max_len, device=self.theta.device, dtype=self.theta.dtype)
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)

    def forward(
        self,
        x: Float[torch.Tensor, "*shape heads seq dim"],
        seq_ids: Optional[Int[torch.Tensor, "*shape seq"]] = None,
    ) -> Float[torch.Tensor, "*shape heads seq dim"]:
        if seq_ids is None:
            seq_ids = torch.arange(x.shape[-2], device=x.device, dtype=torch.int32)
        else:
            seq_ids = rearrange(seq_ids, "... seq -> ... 1 seq")

        rot_cos = self.cos[seq_ids].to(x.dtype)
        rot_sin = self.sin[seq_ids].to(x.dtype)
        return rot_cos * x + rot_sin * self._rotate(x)


class ExtrapolatableRotaryProjection(RotaryProjection):
    """RoPE with xPos scaling for length extrapolation (Sun et al., 2022)."""

    def __init__(
        self,
        *,
        proj_width: int,
        max_len: int = 512,
        base: int = 10000,
        xpos_scale_base: int = 256,
        xpos_scale_exponent: float = 1.0,
    ):
        super().__init__(proj_width=proj_width, max_len=max_len, base=base)
        self.xpos_scale_base = xpos_scale_base
        self.xpos_scale_exponent = xpos_scale_exponent

        xpos_base_scale = (
            torch.arange(0, self.proj_width, 2).float() + 0.4 * self.proj_width
        ) / (1.4 * self.proj_width)
        self.register_buffer("xpos_base_scale", xpos_base_scale, persistent=False)

    def _get_xpos_scale(
        self,
        seq_ids: Int[torch.Tensor, "*shape heads seq"],
    ) -> Float[torch.Tensor, "*shape heads seq dim"]:
        max_pos = seq_ids.max()
        center = torch.div(max_pos + 1, 2, rounding_mode='floor')
        power = (seq_ids.float() - center) / self.xpos_scale_base
        scale = self.xpos_base_scale ** power.unsqueeze(-1)
        scale = repeat(scale, "... d -> ... (d 2)")
        return scale ** self.xpos_scale_exponent

    def forward(
        self,
        x: Float[torch.Tensor, "*shape heads seq dim"],
        seq_ids: Optional[Int[torch.Tensor, "*shape seq"]] = None,
    ) -> Float[torch.Tensor, "*shape heads seq dim"]:
        if seq_ids is None:
            prepared_seq_ids = torch.arange(x.size(-2), device=x.device, dtype=torch.int32)
        else:
            prepared_seq_ids = rearrange(seq_ids, "... seq -> ... 1 seq")

        return super().forward(x, seq_ids) * self._get_xpos_scale(prepared_seq_ids).to(x.dtype)


class QueryKeyProjection(nn.Module):
    def __init__(
        self,
        head_dim: int,
        proj_layer: type[Projection],
        kwargs: Optional[dict[str, Any]] = None,
        key_proj_layer: Optional[type[Projection]] = None,
        key_kwargs: Optional[dict[str, Any]] = None,
        partial_factor: Optional[tuple[float, float]] = None,
    ):
        super().__init__()
        if partial_factor is not None:
            assert 0.0 <= partial_factor[0] < partial_factor[1] <= 1.0

        self.head_dim = head_dim
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(proj_width=self.proj_width, **(kwargs or {}))
        if key_proj_layer is None:
            self.key_proj = self.query_proj
        else:
            self.key_proj = key_proj_layer(proj_width=self.proj_width, **(key_kwargs or kwargs or {}))

    @ft.cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @ft.cached_property
    def split_sizes(self) -> tuple[int, int, int]:
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(
        self,
        query: Float[torch.Tensor, "*shape q_heads q_len dim"],
        key: Float[torch.Tensor, "*shape kv_heads kv_len dim"],
        query_ids: Optional[Int[torch.Tensor, "*shape q_len"]] = None,
        kv_ids: Optional[Int[torch.Tensor, "*shape kv_len"]] = None,
    ) -> tuple[
        Float[torch.Tensor, "*shape q_heads q_len dim"],
        Float[torch.Tensor, "*shape kv_heads kv_len dim"],
    ]:
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_ids=query_ids)
            keys[1] = self.key_proj(keys[1], seq_ids=kv_ids)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_ids=query_ids)
            key = self.key_proj(key, seq_ids=kv_ids)
        return query, key


# =====================================================================
# Self-Attention
# =====================================================================


class SelfAttention(nn.Module):
    """Multi-head self-attention with MuP scaling.

    Layout: HSD (batch, heads, seq, dim) — zero transposes.
    """

    def __init__(
        self,
        config: TotoV2ModelConfig,
        qk_proj_layer: Optional[Callable[[int], QueryKeyProjection]] = None,
        is_variate_layer: bool = False,
    ):
        super().__init__()
        self.config = config
        self.is_variate_layer = is_variate_layer

        if config.qk_norm:
            self.q_norm = uu.RMSNorm(
                config.qk_dim, eps=config.norm_eps, include_weight=config.qk_norm_include_weight,
            )
            self.k_norm = uu.RMSNorm(
                config.qk_dim, eps=config.norm_eps, include_weight=config.qk_norm_include_weight,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.in_proj = uu.Linear(
            config.d_model,
            config.qk_dim * config.num_heads
            + config.qk_dim * config.num_groups
            + config.v_dim * config.num_groups,
            bias=config.attn_bias,
        )
        self.qk_proj = qk_proj_layer(config.qk_dim) if qk_proj_layer is not None else None
        self.out_proj = uu.Linear(
            config.v_dim * config.num_heads, config.d_model, bias=config.attn_bias,
        )

        self._split_sizes = [
            config.qk_dim * config.num_heads,
            config.qk_dim * config.num_groups,
            config.v_dim * config.num_groups,
        ]
        self._Hq = config.num_heads
        self._Hkv = config.num_groups
        # MuP: 1/d_k (not 1/√d_k) to prevent logit explosion as width grows.
        self.mult = 1.0 / config.qk_dim
        self._pds = uu.PerDimScale(config.qk_dim) if config.per_dim_scale else None

    def forward(
        self,
        state: Float[torch.Tensor, "batch seq_len dim"],
        seq_ids: Optional[Int[torch.Tensor, "... seq_len"]] = None,
        **kwargs,
    ) -> Float[torch.Tensor, "batch seq_len dim"]:
        Hq, Hkv = self._Hq, self._Hkv
        q, k, v = torch.split(self.in_proj(state), self._split_sizes, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=Hq)
        k = rearrange(k, "b s (h d) -> b h s d", h=Hkv)
        v = rearrange(v, "b s (h d) -> b h s d", h=Hkv)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self._pds is not None:
            q = self._pds(q)
        if self.qk_proj is not None:
            seq = seq_ids[..., -q.size(-2):] if seq_ids is not None else None
            q, k = self.qk_proj(q, k, query_ids=seq, kv_ids=seq)

        attn_mask = kwargs.get("attn_mask")
        is_causal = not self.is_variate_layer if attn_mask is None else False
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.mult,
            enable_gqa=self.config.heads_per_group > 1,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


# =====================================================================
# Feed-Forward Networks
# =====================================================================


class GatedLinearUnitFeedForwardNetwork(nn.Module):
    """SwiGLU-based FFN with MuP scaling."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or self.adjust_hidden_dim(4 * in_dim)
        out_dim = out_dim or in_dim
        self.fc1 = uu.Linear(in_dim, 2 * hidden_dim, bias=bias, constraint=None)
        self.fc2 = uu.Linear(hidden_dim, out_dim, bias=bias, constraint=None)
        self.dropout1 = uu.Dropout(ffn_dropout_p)
        self.dropout2 = uu.Dropout(ffn_dropout_p)

    @staticmethod
    def adjust_hidden_dim(dim) -> int:
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        fc1_out = self.fc1(x)
        gate, x = fc1_out.chunk(2, dim=-1)
        x = self.dropout1(gate * F.silu(x))
        return self.dropout2(self.fc2(x))


class ResidualMLP(nn.Module):
    """Residual MLP with MuP scaling using τ-rule residual connections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        tau: float = 1.0,
        layer_type: str = "hidden",
    ):
        super().__init__()
        self.tau = tau

        if layer_type == "input":
            self.linear1 = uu.Linear(in_dim, hidden_dim, bias=bias, constraint=None)
            self.linear2 = uu.Linear(hidden_dim, out_dim, bias=bias)
            self.skip_proj = uu.Linear(in_dim, out_dim, bias=bias, constraint=None)
        elif layer_type == "output":
            self.linear1 = uu.Linear(in_dim, hidden_dim, bias=bias)
            self.linear2 = uu.LinearReadout(hidden_dim, out_dim, bias=bias)
            self.skip_proj = uu.LinearReadout(in_dim, out_dim, bias=bias)
        else:
            self.linear1 = uu.Linear(in_dim, hidden_dim, bias=bias)
            self.linear2 = uu.Linear(hidden_dim, out_dim, bias=bias)
            self.skip_proj = uu.Linear(in_dim, out_dim, bias=bias)

        self.dropout = uu.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main, x_skip = U.residual_split(x, self.tau)
        h = U.silu(self.linear1(x_main))
        h = self.dropout(self.linear2(h))
        skip = self.skip_proj(x_skip)
        return U.residual_add(h, skip, self.tau)


class InputResidualMLP(ResidualMLP):
    def __init__(self, *args, **kwargs):
        kwargs["layer_type"] = "input"
        super().__init__(*args, **kwargs)


class OutputResidualMLP(ResidualMLP):
    def __init__(self, *args, **kwargs):
        kwargs["layer_type"] = "output"
        super().__init__(*args, **kwargs)


# =====================================================================
# Transformer
# =====================================================================


class SelfAttentionTransformerLayer(nn.Module):
    """Transformer layer with MuP scaling and τ-rule residual scaling."""

    def __init__(
        self,
        config: TotoV2ModelConfig,
        attn: SelfAttention,
        layer_idx: int = 0,
        num_layers: int = 1,
        residual_mult: float = 1.0,
        residual_attn_ratio: float = 0.25,
    ):
        super().__init__()
        self.config = config
        self.attn = attn
        self._layer_idx = layer_idx
        self.ffn = GatedLinearUnitFeedForwardNetwork(
            in_dim=config.d_model, hidden_dim=config.d_ff,
            out_dim=None, bias=config.mlp_bias, ffn_dropout_p=config.dropout_p,
        )
        self.norm1 = uu.RMSNorm(config.d_model, eps=config.norm_eps, include_weight=config.norm_include_weight)
        self.norm2 = uu.RMSNorm(config.d_model, eps=config.norm_eps, include_weight=config.norm_include_weight)

        total_depth = 2 * num_layers
        tau_rule = uu.transformer_residual_scaling_rule(
            residual_mult=residual_mult, residual_attn_ratio=residual_attn_ratio,
        )
        self.register_buffer("attn_tau", torch.tensor(tau_rule(2 * layer_idx, total_depth)))
        self.register_buffer("mlp_tau", torch.tensor(tau_rule(2 * layer_idx + 1, total_depth)))

        self.attn_resid_dropout = uu.Dropout(config.dropout_p)
        self.mlp_resid_dropout = uu.Dropout(config.dropout_p)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len dim"],
        seq_ids: Optional[Int[torch.Tensor, "... seq"]] = None,
        **kwargs,
    ) -> Float[torch.Tensor, "... seq_len dim"]:
        x, skip = U.residual_split(x, self.attn_tau)
        x = self.attn(self.norm1(x), seq_ids, **kwargs)
        x = U.residual_add(self.attn_resid_dropout(x), skip, self.attn_tau)

        x, skip = U.residual_split(x, self.mlp_tau)
        x = self.ffn(self.norm2(x))
        return U.residual_add(self.mlp_resid_dropout(x), skip, self.mlp_tau)


class VariateTimeTransformerDecoder(nn.Module):
    """Transformer decoder with alternating time/variate attention layers."""

    def __init__(
        self,
        config: TotoV2ModelConfig,
        residual_mult: float = 1.0,
        residual_attn_ratio: float = 0.25,
    ):
        super().__init__()
        self.config = config

        if config.use_xpos:
            query_proj_layer = ft.partial(ExtrapolatableRotaryProjection, xpos_scale_exponent=1.0)
            key_proj_layer = ft.partial(ExtrapolatableRotaryProjection, xpos_scale_exponent=-1.0)
            qk_proj_layer = ft.partial(
                QueryKeyProjection,
                proj_layer=query_proj_layer, key_proj_layer=key_proj_layer,
                kwargs={"max_len": 8192}, partial_factor=(0.0, 0.5),
            )
        else:
            qk_proj_layer = ft.partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection, kwargs={"max_len": 8192},
                partial_factor=(0.0, 0.5),
            )

        layers = []
        for idx in range(config.num_layers):
            variate_layer = self._if_variate_layer(idx)
            layers.append(
                SelfAttentionTransformerLayer(
                    config,
                    attn=SelfAttention(
                        config,
                        qk_proj_layer=qk_proj_layer if not variate_layer else None,
                        is_variate_layer=variate_layer,
                    ),
                    layer_idx=idx, num_layers=config.num_layers,
                    residual_mult=residual_mult, residual_attn_ratio=residual_attn_ratio,
                )
            )

        self.layers = uu.DepthModuleList(layers)
        self.out_norm = uu.RMSNorm(config.d_model, eps=config.norm_eps, include_weight=config.norm_include_weight)

    def _if_variate_layer(self, layer_idx: int) -> bool:
        if self.config.variate_layer_first:
            return layer_idx % self.config.layer_group_size < self.config.num_variate_layers_per_group
        return (
            layer_idx % self.config.layer_group_size
            >= self.config.layer_group_size - self.config.num_variate_layers_per_group
        )

    def _sdpa_kwargs(
        self,
        state: Float[torch.Tensor, "*shape num_series q_len dim"],
        time_ids: Optional[Int[torch.Tensor, "*shape #num_series seq_len"]],
        group_ids: Optional[Int[torch.Tensor, "*shape #num_series #seq_len"]],
    ) -> tuple[dict, dict]:
        q_len = state.shape[-2]
        kv_len = q_len

        # Time attention mask (causal)
        time_attn_mask = torch.where(
            torch.tril(torch.ones(q_len, kv_len, dtype=bool, device=state.device), diagonal=kv_len - q_len),
            torch.zeros(1, dtype=state.dtype, device=state.device),
            torch.full((1,), -torch.inf, dtype=state.dtype, device=state.device),
        )
        if group_ids is not None and group_ids.shape[-1] > 1:
            time_attn_mask = time_attn_mask + torch.where(
                group_ids[..., -q_len:, None] == group_ids[..., None, :],
                torch.zeros(1, dtype=state.dtype, device=state.device),
                torch.full((1,), -torch.inf, dtype=state.dtype, device=state.device),
            )
        elif time_ids is not None:
            time_attn_mask = time_attn_mask + torch.where(
                rearrange(time_ids == -1, "... kv_len -> ... 1 kv_len"),
                torch.full((1,), -torch.inf, dtype=state.dtype, device=state.device),
                torch.zeros(1, dtype=state.dtype, device=state.device),
            )
        time_attn_mask = rearrange(time_attn_mask, "... s1 s2 -> (...) 1 s1 s2").contiguous()
        time_layer_kwargs = {"attn_mask": time_attn_mask}

        # Variate attention mask
        if group_ids is not None and group_ids.shape[-2] > 1:
            var_attn_mask = torch.where(
                rearrange(group_ids[..., -q_len:], "... n s -> ... s 1 n 1")
                == rearrange(group_ids[..., -q_len:], "... n s -> ... s 1 1 n"),
                torch.zeros(1, dtype=state.dtype, device=state.device),
                torch.full((1,), -torch.inf, dtype=state.dtype, device=state.device),
            )
            var_attn_mask = rearrange(var_attn_mask, "... 1 n1 n2 -> (...) 1 n1 n2").contiguous()
            var_layer_kwargs: dict = {"attn_mask": var_attn_mask}
        else:
            var_layer_kwargs = {}

        return time_layer_kwargs, var_layer_kwargs

    def forward(
        self,
        state: Float[torch.Tensor, "*shape num_series q_len dim"],
        time_ids: Optional[Int[torch.Tensor, "*shape #num_series seq_len"]] = None,
        group_ids: Optional[Int[torch.Tensor, "*shape #num_series #seq_len"]] = None,
        **kwargs,
    ) -> Float[torch.Tensor, "*shape num_series q_len dim"]:
        if time_ids is None:
            time_ids = torch.arange(state.shape[-2], device=state.device, dtype=torch.int32)

        time_layer_kwargs, var_layer_kwargs = self._sdpa_kwargs(state, time_ids, group_ids)
        num_series, seq_len = state.shape[-3], state.shape[-2]

        if time_ids is not None and time_ids.dim() > 1:
            flat_time_ids = time_ids.expand(*state.shape[:-1]).flatten(0, -2)
        else:
            flat_time_ids = time_ids

        leading = state.shape[:-2]
        state = rearrange(state, "... seq_len dim -> (...) seq_len dim")

        for idx, layer in enumerate(self.layers):
            if self._if_variate_layer(idx):
                state = rearrange(
                    state, "(b n) s d -> (b s) n d", n=num_series,
                )
                state = layer(state, **var_layer_kwargs)
                state = rearrange(
                    state, "(b s) n d -> (b n) s d", s=seq_len,
                )
            else:
                state = layer(state, seq_ids=flat_time_ids, **time_layer_kwargs)

        state = state.unflatten(0, leading)
        return self.out_norm(state)


# =====================================================================
# Output Head
# =====================================================================


class FusedPatchedParamProjection(nn.Module):
    """Single fused projection for patched outputs."""

    def __init__(
        self,
        embeds_dim: int,
        param_shapes: tuple[int, ...],
        get_proj_fn: Callable[[int, int], nn.Module],
        patch_size: int,
    ):
        super().__init__()
        self.output_shape = (patch_size, *param_shapes)
        self.proj = get_proj_fn(embeds_dim, math.prod(self.output_shape))

    def forward(
        self, inputs: Float[torch.Tensor, "*batch_shape embed_dim"]
    ) -> torch.Tensor:
        return self.proj(inputs).unflatten(-1, self.output_shape)


class QuantileKnotsOutputHead(nn.Module):
    """Output head predicting quantiles at fixed knot positions."""

    def __init__(
        self,
        knots: list[float],
        embeds_dim: int,
        param_projection_factory: Optional[Callable],
    ):
        super().__init__()
        self.knots = knots
        self.param_projection = (
            param_projection_factory(embeds_dim, (len(knots),))
            if param_projection_factory is not None
            else None
        )

    def forward(
        self,
        embeddings: Float[torch.Tensor, "*shape embed_dim"],
        q: None,
    ) -> Float[torch.Tensor, "q ..."]:
        return rearrange(self.param_projection(embeddings), "... q -> q ...")




# =====================================================================
# TotoV2 Model
# =====================================================================


class TotoV2ModelInputs(TypedDict):
    target: Float[torch.Tensor, "*batch n_var time"]
    target_mask: Bool[torch.Tensor, "*batch n_var time"]
    series_ids: Int[torch.Tensor, "*batch n_var"]
    num_return_steps: NotRequired[Optional[slice]]


class TotoV2ModelOutputs(NamedTuple):
    quantiles: torch.Tensor
    loc: Float[torch.Tensor, "*batch n_var seq 1"]
    scale: Float[torch.Tensor, "*batch n_var seq 1"]


class TotoV2ForecastInputs(TypedDict):
    target: Float[torch.Tensor, "*batch n_var ctx"]
    target_mask: Bool[torch.Tensor, "*batch n_var ctx"]
    series_ids: Int[torch.Tensor, "*batch n_var"]
    known_dynamic: NotRequired[Float[torch.Tensor, "*batch n_exog ctx+horizon"]]
    known_dynamic_mask: NotRequired[Bool[torch.Tensor, "*batch n_exog ctx+horizon"]]
    known_dynamic_series_ids: NotRequired[Int[torch.Tensor, "*batch n_exog"]]


class TotoV2Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: TotoV2ModelConfig):
        super().__init__()
        self.config = config
        self.scaler = PatchedCausalStdScaler(
            patch_size=config.patch_size, stabilize_with_global=False, online=False,
        )
        self.patch_proj = InputResidualMLP(
            in_dim=2 * config.patch_size, hidden_dim=4 * config.d_model,
            out_dim=config.d_model, dropout_p=config.dropout_p, bias=True,
        )
        self.transformer = VariateTimeTransformerDecoder(
            config, residual_mult=config.residual_mult, residual_attn_ratio=config.residual_attn_ratio,
        )

        def res_mlp_proj_fn(in_dim: int, out_dim: int) -> nn.Module:
            return OutputResidualMLP(
                in_dim=in_dim, hidden_dim=4 * config.d_model,
                out_dim=out_dim, dropout_p=config.dropout_p, bias=True,
            )

        self.output_head = QuantileKnotsOutputHead(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            embeds_dim=config.d_model,
            param_projection_factory=ft.partial(
                FusedPatchedParamProjection,
                get_proj_fn=res_mlp_proj_fn,
                patch_size=config.patch_size * config.num_output_patches,
            ),
        )

    def forward(
        self,
        target: Float[torch.Tensor, "*batch n_var time"],
        target_mask: Bool[torch.Tensor, "*batch n_var time"],
        cpm_mask: Optional[Bool[torch.Tensor, "*batch n_var time"]],
        series_ids: Int[torch.Tensor, "*batch n_var"],
        num_return_steps: Optional[int] = None,
    ) -> TotoV2ModelOutputs:
        scaled_series, loc, scale = self.scaler(target, target_mask & cpm_mask)
        scaled_series = scaled_series.asinh()
        x = self.patch_proj(
            torch.cat(
                [
                    rearrange(scaled_series, "... (seq patch) -> ... seq patch", patch=self.config.patch_size),
                    rearrange(
                        (~(target_mask & cpm_mask)).to(target.dtype),
                        "... (seq patch) -> ... seq patch", patch=self.config.patch_size,
                    ),
                ],
                dim=-1,
            )
        )

        group_ids = repeat(series_ids, "... n_var -> ... n_var seq", seq=x.shape[-2]).clone()

        group_ids[
            (reduce(target_mask, "... (seq patch) -> ... seq", "sum", patch=self.config.patch_size) == 0)
            & (reduce(cpm_mask, "... (seq patch) -> ... seq", "prod", patch=self.config.patch_size) == 1)
        ] = -1

        x = self.transformer(x, group_ids=group_ids)

        if num_return_steps is not None:
            x = x[..., -num_return_steps:, :]
            loc = loc[..., -num_return_steps * self.config.patch_size:]
            scale = scale[..., -num_return_steps * self.config.patch_size:]

        quantiles = self.output_head(x, q=None)
        return TotoV2ModelOutputs(quantiles, loc, scale)

    @torch.no_grad()
    def forecast(self, inputs, horizon, **kwargs):
        num_patches = math.ceil(horizon / self.config.patch_size)
        target = torch.cat(
            [
                inputs["target"],
                torch.zeros(
                    inputs["target"].shape[:-1] + ((num_patches - 1) * self.config.patch_size,),
                    device=inputs["target"].device, dtype=inputs["target"].dtype,
                ),
            ],
            dim=-1,
        )
        target_mask = torch.cat(
            [
                inputs["target_mask"][..., :-self.config.patch_size],
                torch.ones(
                    inputs["target_mask"].shape[:-1] + (num_patches * self.config.patch_size,),
                    device=inputs["target_mask"].device, dtype=inputs["target_mask"].dtype,
                ),
            ],
            dim=-1,
        )
        series_ids = inputs["series_ids"]
        cpm_mask = torch.cat(
            [
                torch.ones_like(inputs["target_mask"]),
                torch.zeros(
                    inputs["target_mask"].shape[:-1] + ((num_patches - 1) * self.config.patch_size,),
                    device=inputs["target_mask"].device, dtype=inputs["target_mask"].dtype,
                ),
            ],
            dim=-1,
        )
        n_var = series_ids.shape[-1]

        if "known_dynamic" in inputs:
            required_padding = (
                math.ceil(horizon / self.config.patch_size) * self.config.patch_size - horizon
            )
            known_dynamic = torch.cat(
                [
                    inputs["known_dynamic"][..., self.config.patch_size:],
                    torch.zeros(
                        inputs["known_dynamic"].shape[:-1] + (required_padding,),
                        device=inputs["known_dynamic"].device, dtype=inputs["known_dynamic"].dtype,
                    ),
                ],
                dim=-1,
            )
            known_dynamic_mask = torch.cat(
                [
                    inputs["known_dynamic_mask"][..., self.config.patch_size:],
                    torch.zeros(
                        inputs["known_dynamic_mask"].shape[:-1] + (required_padding,),
                        device=inputs["known_dynamic_mask"].device, dtype=inputs["known_dynamic_mask"].dtype,
                    ),
                ],
                dim=-1,
            )
            target = torch.cat([target, known_dynamic], dim=-2)
            target_mask = torch.cat([target_mask, known_dynamic_mask], dim=-2)
            series_ids = torch.cat([series_ids, inputs["known_dynamic_series_ids"]], dim=-1)
            cpm_mask = torch.cat([cpm_mask, torch.ones_like(known_dynamic_mask)], dim=-2)

        quantiles, loc, scale = self.forward(
            target, target_mask, cpm_mask=cpm_mask, series_ids=series_ids,
            num_return_steps=num_patches,
        )

        loc = repeat(
            loc, "... (seq patch) -> ... seq (n patch)",
            patch=self.config.patch_size, n=self.config.num_output_patches,
        )
        scale = repeat(
            scale, "... (seq patch) -> ... seq (n patch)",
            patch=self.config.patch_size, n=self.config.num_output_patches,
        )
        quantiles = quantiles.sinh() * scale + loc
        quantiles = rearrange(
            quantiles[..., ::self.config.num_output_patches, :],
            "... seq patch -> ... (seq patch)",
        )[..., :n_var, :horizon]

        def clamp_nonfinite(vals: torch.Tensor) -> torch.Tensor:
            return torch.where(
                vals == float("inf"),
                torch.where(vals.isfinite(), vals, -float("inf")).amax(dim=-1, keepdim=True),
                torch.where(
                    vals == -float("inf"),
                    torch.where(vals.isfinite(), vals, float("inf")).amin(dim=-1, keepdim=True),
                    vals,
                ),
            )

        quantiles = clamp_nonfinite(quantiles)
        quantiles = quantiles.sort(dim=0).values

        return quantiles

    @classmethod
    def _from_pretrained(cls, *, model_id, map_location="cpu", strict=True, **kwargs):
        model_dir = Path(model_id)
        if model_dir.is_dir() and (model_dir / "config.json").exists():
            raw = json.loads((model_dir / "config.json").read_text())
            known = {f.name for f in dataclasses.fields(TotoV2ModelConfig)}
            model = cls(TotoV2ModelConfig(**{k: v for k, v in raw.items() if k in known}))

            index_file = model_dir / "model.safetensors.index.json"
            if index_file.exists():
                index = json.loads(index_file.read_text())
                state_dict = {}
                for shard_file in set(index["weight_map"].values()):
                    state_dict.update(
                        load_safetensors_file(str(model_dir / shard_file), device=str(map_location))
                    )
                model.load_state_dict(state_dict, strict=strict)
            else:
                load_torch_model(model, model_dir, strict=strict, map_location=map_location)

            return model.eval()

        return super()._from_pretrained(model_id=model_id, map_location=map_location, strict=strict, **kwargs)


# =====================================================================
# GluonTS Integration
# =====================================================================


class TotoV2GluonTSModel(nn.Module):
    def __init__(self, model: TotoV2Model, config: TotoV2GluonTSModelConfig):
        super().__init__()
        self.prediction_length = config.prediction_length
        self.model = model
        self.config = config

    def forward(
        self,
        past_target: Float[torch.Tensor, "*batch n_var ctx"],
        past_observed_target: Bool[torch.Tensor, "*batch n_var ctx"],
        past_is_pad: Bool[torch.Tensor, "*batch n_var ctx"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "*batch n_var ctx+horizon"]] = None,
        observed_feat_dynamic_real: Optional[Bool[torch.Tensor, "*batch n_var ctx+horizon"]] = None,
        past_feat_dynamic_real: Optional[Float[torch.Tensor, "*batch n_var ctx"]] = None,
        past_observed_feat_dynamic_real: Optional[Bool[torch.Tensor, "*batch n_var ctx"]] = None,
    ) -> Float[torch.Tensor, "*batch q seq n_var"]:
        inputs = {
            "target": past_target,
            "target_mask": past_observed_target * ~past_is_pad.bool().unsqueeze(-2),
        }

        if past_feat_dynamic_real is not None:
            inputs["target"] = torch.cat([inputs["target"], past_feat_dynamic_real], dim=-2)
            inputs["target_mask"] = torch.cat(
                [
                    inputs["target_mask"],
                    past_observed_feat_dynamic_real * ~past_is_pad.bool().unsqueeze(-2),
                ],
                dim=-2,
            )

        inputs["series_ids"] = torch.zeros_like(inputs["target"][..., 0], dtype=torch.long)

        if feat_dynamic_real is not None:
            inputs |= {
                "known_dynamic": feat_dynamic_real,
                "known_dynamic_mask": observed_feat_dynamic_real,
                "known_dynamic_series_ids": torch.zeros_like(feat_dynamic_real[..., 0], dtype=torch.long),
            }

        quantiles = self.model.forecast(inputs, self.prediction_length)
        outputs = rearrange(
            quantiles[:, :, :past_target.shape[-2], :],
            "q b var seq -> b var seq q",
        ).squeeze(1)
        return (outputs,), None, None

    @property
    def input_transform(self) -> Transformation:
        transform = AsNumpyArray(
            field="target",
            expected_ndim=1 if self.config.target_dim == 1 else 2,
            dtype=np.float32,
        )
        if self.config.target_dim == 1:
            transform += ExpandDimArray(field="target", axis=0)
        transform += AddObservedValuesIndicator(
            target_field="target", output_field="observed_target", dtype=bool,
        )
        if self.config.feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(field="feat_dynamic_real", expected_ndim=2, dtype=np.float32)
            transform += AddObservedValuesIndicator(
                target_field="feat_dynamic_real", output_field="observed_feat_dynamic_real", dtype=bool,
            )
        if self.config.past_feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(field="past_feat_dynamic_real", expected_ndim=2, dtype=np.float32)
            transform += AddObservedValuesIndicator(
                target_field="past_feat_dynamic_real", output_field="past_observed_feat_dynamic_real", dtype=bool,
            )
        return transform

    @property
    def instance_splitter(self) -> InstanceSplitter:
        context_length = (
            self.config.context_length
            - (math.ceil(self.prediction_length / self.model.config.patch_size) - 1)
            * self.model.config.patch_size
        )
        past_length = max(
            self.model.config.patch_size,
            math.floor(context_length / self.model.config.patch_size) * self.model.config.patch_size,
        )
        return TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=past_length,
            future_length=math.ceil(self.prediction_length / self.model.config.patch_size) * self.model.config.patch_size,
            observed_value_field="observed_target",
            time_series_fields=(
                ["feat_dynamic_real", "observed_feat_dynamic_real"]
                if self.config.feat_dynamic_real_dim > 0 else []
            ),
            past_time_series_fields=(
                ["past_feat_dynamic_real", "past_observed_feat_dynamic_real"]
                if self.config.past_feat_dynamic_real_dim > 0 else []
            ),
            output_NTC=False,
        )

    @property
    def input_names(self) -> list[str]:
        return (
            ["past_target", "past_observed_target", "past_is_pad"]
            + (["feat_dynamic_real", "observed_feat_dynamic_real"] if self.config.feat_dynamic_real_dim > 0 else [])
            + (["past_feat_dynamic_real", "past_observed_feat_dynamic_real"] if self.config.past_feat_dynamic_real_dim > 0 else [])
        )

    @property
    def forecast_generator(self) -> QuantileForecastGenerator:
        return QuantileForecastGenerator(list(self.config.quantiles))

    def create_predictor(self, batch_size: int, device: str = "auto") -> PyTorchPredictor:
        return PyTorchPredictor(
            input_names=self.input_names,
            prediction_net=self,
            batch_size=batch_size,
            prediction_length=self.prediction_length,
            input_transform=self.input_transform + self.instance_splitter,
            forecast_generator=self.forecast_generator,
            device=device,
        )
