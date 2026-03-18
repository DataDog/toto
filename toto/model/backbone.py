# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from math import ceil
from typing import NamedTuple, Optional, Type, cast

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int

from ..model.distribution import DISTRIBUTION_CLASSES_LOOKUP, DistributionOutput
from ..model.embedding import PatchEmbedding
from ..model.scaler import scaler_types
from ..model.transformer import Transformer
from ..model.util import KVCache
from .fusion import Fusion


class TotoOutput(NamedTuple):
    """
    Output of the Toto model. Contains the output distribution, the location parameters,
    and the scale parameters.
    """

    distribution: torch.distributions.Distribution
    loc: Float[torch.Tensor, "batch variate"]
    scale: Float[torch.Tensor, "batch variate"]


class TotoBackbone(torch.nn.Module):
    """
    Toto (Timeseries-Optimized Transformer for Observability) is a transformer-based model for multivariate
    time series forecasting. It applies a patch embedding to the input data, followed by a transformer
    that alternates between time-wise and space-wise attention. The transformer is followed by a linear projection
    that maps the transformer output to the output distribution.

    The output distribution can be a single distribution (e.g. Gaussian) or a mixture of distributions.
    If a mixture of distributions is used, the model will learn to predict the mixture weights
    as well as the parameters of the individual distributions.

    Parameters
    ----------
    patch_size
        Size of the patch to use for the patch embedding.
    stride
        Stride to use for the patch embedding.
    embed_dim
        Dimension of the model's latent space.
    num_layers
        Number of transformer layers to use.
    num_heads
        Number of attention heads to use in each self-attention layer.
    mlp_hidden_dim
        Dimension of the hidden layer in the feedforward network.
    dropout
        Dropout rate to use in the model.
    spacewise_every_n_layers
        How many time-wise transformer layers to apply between each space-wise transformer layer.
    spacewise_first
        Whether to apply space-wise attention before time-wise attention.
    scaler_cls
        Class to use for scaling the input data.
    output_distribution_classes
        List of classes to use for the output distribution. If a single class is provided, the model
        will output a single distribution. If multiple classes are provided, the model will output a
        learned mixture of distributions.
    output_distribution_kwargs
        Keyword arguments to pass to the output distribution class. Note: this currently only works
        with a single output distribution class.
    use_memory_efficient_attention:
        Whether to use memory-efficient attention. If True, the model will use the memory-efficient from xFormers.
    stabilize_with_global:
        Whether to use global statistics to stabilize causal statistics by clamping extreme values. Only applies to causal scalers.
    scale_factor_exponent:
        Exponent that controls the allowed range of deviation from global scale for causal scalers.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        spacewise_every_n_layers: int,
        scaler_cls: str,
        output_distribution_classes: list[str],
        spacewise_first: bool = True,
        output_distribution_kwargs: dict | None = None,
        use_memory_efficient_attention: bool = True,
        stabilize_with_global: bool = True,
        scale_factor_exponent: float = 10.0,
        enable_causal_robustness: bool = False,
        causal_robustness_alpha: float = 1e-4,
        causal_robustness_eps: float = 1e-6,
        causal_robustness_max_penalty: float = 20.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latest_causal_robustness_penalty: Optional[torch.Tensor] = None
        # Attributes for variate-label fusion (initialized when enable_variate_labels is called)
        self.fusion: Optional[Fusion] = None
        self.num_prepended_tokens: int = 0
        self.target_variate_label: Optional[torch.nn.Parameter] = None
        self.exogenous_variate_label: Optional[torch.nn.Parameter] = None
        # strings are used when loading a safetensors checkpoint
        # Initialize patch-based scalers with the correct patch_size
        if scaler_cls == "<class 'model.scaler.CausalPatchStdMeanScaler'>":
            self.scaler = scaler_types[scaler_cls](
                patch_size=patch_size,
                stabilize_with_global=stabilize_with_global,
                scale_factor_exponent=scale_factor_exponent,
            )
        else:
            self.scaler = scaler_types[scaler_cls]()

        self.patch_embed = PatchEmbedding(patch_size, stride, embed_dim)
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.transformer = Transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            spacewise_every_n_layers=spacewise_every_n_layers,
            spacewise_first=spacewise_first,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            enable_causal_robustness=enable_causal_robustness,
            causal_robustness_alpha=causal_robustness_alpha,
            causal_robustness_eps=causal_robustness_eps,
            causal_robustness_max_penalty=causal_robustness_max_penalty,
            fusion=self.fusion,
        )
        self.unembed = torch.nn.Linear(embed_dim, embed_dim * patch_size)

        # TODO[BEN] this doesn't need to be a list
        output_distribution_classes_ = [DISTRIBUTION_CLASSES_LOOKUP[c] for c in output_distribution_classes]
        self.output_distribution = output_distribution_classes_[0](embed_dim, **(output_distribution_kwargs or {}))

    def allocate_kv_cache(
        self,
        batch_size: int,
        num_variates: int,
        max_time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        return KVCache(
            batch_size=batch_size,
            num_variates=num_variates,
            transformer_layers=list(self.transformer.layers),
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=cast(int, self.transformer.layers[0].num_heads),
            max_seq_len=ceil(max_time_steps / self.patch_embed.stride),
            device=device,
            dtype=dtype,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
        )

    def backbone(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"],
        id_mask: Float[torch.Tensor, "batch #variate time_steps"],
        kv_cache: Optional[KVCache] = None,
        scaling_prefix_length: Optional[int] = None,
        num_exogenous_variables: int = 0,
    ) -> tuple[
        Float[torch.Tensor, "batch variates time_steps embed_dim"],
        Float[torch.Tensor, "batch variates time_steps"],
        Float[torch.Tensor, "batch variates time_steps"],
    ]:
        scaled_inputs: Float[torch.Tensor, "batch variate time_steps"]
        loc: Float[torch.Tensor, "batch variate time_steps"]
        scale: Float[torch.Tensor, "batch variate time_steps"]

        # Standard scaling operation, same API but without ID mask.
        scaled_inputs, loc, scale = self.scaler(
            inputs,
            weights=torch.ones_like(inputs, device=inputs.device),
            padding_mask=input_padding_mask,
            prefix_length=scaling_prefix_length,
        )

        if kv_cache is not None:
            # Account for prepended condition tokens when using KV cache.
            # Cached length counts prepended tokens; do not overcount when computing time-series prefix.
            kv_cache_len_tensor = kv_cache.current_len(0)
            kv_cache_len = (
                int(kv_cache_len_tensor) if isinstance(kv_cache_len_tensor, torch.Tensor) else kv_cache_len_tensor
            )
            prefix_len = max(0, self.patch_embed.stride * (kv_cache_len - self.num_prepended_tokens))

            # Truncate inputs so that the transformer only processes
            # the last patch in the sequence. We'll use the KVCache
            # for the earlier patches.
            scaled_inputs = scaled_inputs[:, :, prefix_len:]

            # As a simplification, when using kv cache we only allow decoding
            # one step at a time after the initial forward pass.
            assert (prefix_len == 0) or (
                scaled_inputs.shape[-1] == self.patch_embed.stride
            ), "Must decode one step at a time."

            input_padding_mask = input_padding_mask[:, :, prefix_len:]
            id_mask = id_mask[:, :, prefix_len:]

        embeddings: Float[torch.Tensor, "batch variate seq_len embed_dim"]
        reduced_id_mask: Float[torch.Tensor, "batch variate seq_len"]

        embeddings, reduced_id_mask = self.patch_embed(scaled_inputs, id_mask)

        # Build variate label embeddings (one per variate) if enabled
        variate_label_embeds = self.build_variate_label_embeds(num_exogenous_variables, embeddings)

        # Apply the transformer on the embeddings (fusion handles prepending at layer 0)
        original_seq_len = embeddings.shape[2]
        transformed: Float[torch.Tensor, "batch variates seq_len embed_dim"] = self.transformer(  # type: ignore[assignment]
            embeddings, reduced_id_mask, kv_cache, variate_label_embeds=variate_label_embeds
        )
        self.latest_causal_robustness_penalty = self.transformer.latest_causal_robustness_penalty
        # Crop out the prepended tokens before unembedding
        added_tokens = transformed.shape[2] - original_seq_len
        if added_tokens > 0:
            transformed = transformed[:, :, added_tokens:]

        # Unembed and flatten the sequence
        flattened: Float[torch.Tensor, "batch variates new_seq_len embed_dim"] = rearrange(
            self.unembed(transformed),
            "batch variates seq_len (patch_size embed_dim) -> batch variates (seq_len patch_size) embed_dim",
            embed_dim=self.embed_dim,
        )
        return flattened, loc, scale

    def forward(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"],
        id_mask: Float[torch.Tensor, "batch #variate time_steps"],
        kv_cache: Optional[KVCache] = None,
        scaling_prefix_length: Optional[int] = None,
        num_exogenous_variables: int = 0,
    ) -> TotoOutput:
        flattened, loc, scale = self.backbone(
            inputs,
            input_padding_mask,
            id_mask,
            kv_cache,
            scaling_prefix_length,
            num_exogenous_variables,
        )

        return TotoOutput(self.output_distribution(flattened), loc, scale)

    @property
    def device(self):
        return next(self.parameters()).device

    def enable_variate_labels(self) -> None:
        """
        Enable variate labels for exogenous feature differentiation.
        Called automatically when using exogenous features during finetuning.
        - Creates trainable label parameters for target and exogenous variates
        - Enables fusion by installing a Fusion module
        """
        self.fusion = Fusion()
        self.num_prepended_tokens = 1
        self.target_variate_label = torch.nn.Parameter(torch.randn(self.embed_dim))
        self.exogenous_variate_label = torch.nn.Parameter(torch.randn(self.embed_dim))
        # If transformer already exists (e.g., loaded from checkpoint), update it as well
        if hasattr(self, "transformer") and self.transformer is not None:
            self.transformer.fusion = self.fusion

    def build_variate_label_embeds(
        self,
        num_exogenous_variables: int,
        embeddings: Float[torch.Tensor, "batch variate seq_len embed_dim"],
    ) -> Optional[Float[torch.Tensor, "batch variate 1 embed_dim"]]:
        """
        Build per-variate label embeddings for fusion.
        The last num_exogenous_variables variates are treated as exogenous and receive the exogenous label.
        Returns None when variate labels are not enabled.
        """
        if self.fusion is None:
            return None

        assert self.target_variate_label is not None
        assert self.exogenous_variate_label is not None

        batch_size, num_variates, _, _ = embeddings.shape

        target_variate_label = repeat(self.target_variate_label, "d -> b v 1 d", b=batch_size, v=num_variates).to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        exogenous_variate_label = repeat(self.exogenous_variate_label, "d -> b v 1 d", b=batch_size, v=num_variates).to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        # Build exog_mask from num_exogenous_variables: last num_exogenous_variables variates are exogenous
        exog_mask = torch.zeros(1, num_variates, 1, 1, dtype=torch.bool, device=embeddings.device)
        if num_exogenous_variables > 0:
            exog_mask[:, -num_exogenous_variables:] = True
        # Select per-variate label: target label for genuine targets, exogenous label for EV channels
        return torch.where(exog_mask, exogenous_variate_label, target_variate_label)  # (B, V, 1, D)

    def configure_causal_robustness(
        self,
        *,
        enabled: bool,
        alpha: float = 1e-4,
        eps: float = 1e-6,
        max_penalty: float = 20.0,
    ) -> None:
        self.transformer.configure_causal_robustness(
            enabled=enabled,
            alpha=alpha,
            eps=eps,
            max_penalty=max_penalty,
        )
