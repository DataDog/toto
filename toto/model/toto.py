# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors.torch as safetorch
import torch
from huggingface_hub import ModelHubMixin, constants, hf_hub_download

from ..model.attention import XFORMERS_AVAILABLE
from ..model.backbone import TotoBackbone
from ..model.transformer import XFORMERS_SWIGLU_AVAILABLE


class Toto(torch.nn.Module, ModelHubMixin):
    """
    PyTorch module for Toto (Timeseries-Optimized Transformer for Observability).

    Parameters
    ----------
    **model_kwargs
        Additional keyword arguments to pass to the TotoModule constructor.
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
        **model_kwargs,
    ):
        super().__init__()
        self.model = TotoBackbone(
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            spacewise_every_n_layers=spacewise_every_n_layers,
            scaler_cls=scaler_cls,
            output_distribution_classes=output_distribution_classes,
            spacewise_first=spacewise_first,
            output_distribution_kwargs=output_distribution_kwargs,
            use_memory_efficient_attention=use_memory_efficient_attention,
            stabilize_with_global=stabilize_with_global,
            scale_factor_exponent=scale_factor_exponent,
            **model_kwargs,
        )
        self.model_kwargs = model_kwargs

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location: str = "cpu",
        strict=True,
        **model_kwargs,
    ):
        """
        Custom checkpoint loading. Used to load a local
        safetensors checkpoint with an optional config.json file.
        """
        if os.path.isdir(checkpoint_path):
            safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
        else:
            safetensors_file = checkpoint_path

        if os.path.exists(safetensors_file):
            model_state = safetorch.load_file(safetensors_file, device=map_location)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {safetensors_file}")

        # Load configuration from config.json if it exists.
        config_file = os.path.join(checkpoint_path, "config.json")
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)

        # Merge any extra kwargs into the configuration.
        config.update(model_kwargs)

        remapped_state_dict = cls._map_state_dict_keys(
            model_state, XFORMERS_SWIGLU_AVAILABLE and not config.get("pre_xformers_checkpoint", False)
        )

        if not XFORMERS_AVAILABLE and config.get("use_memory_efficient_attention", True):
            config["use_memory_efficient_attention"] = False

        instance = cls(**config)
        instance.to(map_location)

        # Filter out unexpected keys
        filtered_remapped_state_dict = {
            k: v
            for k, v in remapped_state_dict.items()
            if k in instance.state_dict() and not k.endswith("rotary_emb.freqs")
        }

        instance.load_state_dict(filtered_remapped_state_dict, strict=strict)
        return instance

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            return cls.load_from_checkpoint(model_file, map_location, strict, **model_kwargs)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=constants.SAFETENSORS_SINGLE_FILE,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
            return cls.load_from_checkpoint(model_file, map_location, strict, **model_kwargs)

    @staticmethod
    def _map_state_dict_keys(state_dict, use_fused_swiglu):
        """
        Maps the keys of a state_dict to match the current model's state_dict.
        Currently this is only used to convert between fused and unfused SwiGLU implementations.
        """
        if use_fused_swiglu:
            remap_keys = {
                "mlp.0.weight": "mlp.0.w12.weight",
                "mlp.0.bias": "mlp.0.w12.bias",
                "mlp.2.weight": "mlp.0.w3.weight",
                "mlp.2.bias": "mlp.0.w3.bias",
            }
        else:
            remap_keys = {
                "mlp.0.w12.weight": "mlp.0.weight",
                "mlp.0.w12.bias": "mlp.0.bias",
                "mlp.0.w3.weight": "mlp.2.weight",
                "mlp.0.w3.bias": "mlp.2.bias",
            }

        def replace_key(text):
            for pattern, replacement in remap_keys.items():
                text = re.sub(pattern, replacement, text)
            return text

        return {replace_key(k): v for k, v in state_dict.items()}

    @property
    def device(self):
        return next(self.model.parameters()).device
