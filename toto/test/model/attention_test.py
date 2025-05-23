# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import os
import sys

import pytest
import torch
from beartype import beartype

from ..helper_functions import set_default_dtype, skip_if_no_xformers

from toto.model.transformer import Transformer
from toto.model.util import KVCache

skip_if_no_xformers()
set_default_dtype()

DEVICE = torch.get_default_device()
DTYPE = torch.get_default_dtype()

# Test parameters
BATCH = 2
VARIATE = 3
SEQ_LEN = 10
EMBED_DIM = 64
NUM_HEADS = 8
HEAD_DIM = EMBED_DIM // NUM_HEADS
DROPOUT = 0.0  # Fixed dropout, but model is always in eval mode.


def generate_id_mask(batch, variate, seq_len):
    # Generate random lengths that sum up to variate
    ids = torch.arange(0, variate, device=DEVICE, dtype=torch.int)
    ids = (ids // 2).clamp(max=variate - 1)
    return ids.unsqueeze(0).unsqueeze(-1).expand(batch, -1, seq_len)


@pytest.fixture(params=[(use_kv_cache) for use_kv_cache in [True, False]])
@beartype
def mock_inputs(request):
    """Create mock input data."""
    use_kv_cache = request.param

    inputs = torch.randn(BATCH, VARIATE, SEQ_LEN, EMBED_DIM, device=DEVICE, dtype=DTYPE)

    # Initialize Transformer
    transformer = Transformer(
        num_layers=6,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=128,
        dropout=DROPOUT,
        spacewise_every_n_layers=3,
        spacewise_first=True,
    ).eval()

    # Generate id_mask
    id_mask = generate_id_mask(BATCH, VARIATE, SEQ_LEN)

    # Conditional generation of timewise_attention_mask
    timewise_attention_mask = None

    # Generate tensor-based spacewise_attention_mask_tensor (train mode)
    spacewise_attention_mask_tensor = transformer._get_mask(
        num_heads=NUM_HEADS,
        dtype=DTYPE,
        id_mask=id_mask,  # Provide the id_mask for spacewise masks
    ).contiguous()

    spacewise_attention_mask_blockdiag = None
    kv_cache = None
    if use_kv_cache:
        kv_cache = KVCache(
            batch_size=BATCH,
            num_variates=VARIATE,
            transformer_layers=transformer.layers,
            num_layers=6,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            max_seq_len=SEQ_LEN,
            device=DEVICE,
            dtype=DTYPE,
        )

    return (
        inputs,
        timewise_attention_mask,
        spacewise_attention_mask_tensor,
        spacewise_attention_mask_blockdiag,
        kv_cache,
    )
