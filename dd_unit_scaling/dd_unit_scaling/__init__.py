# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

"""Compile-friendly, world-size-aware unit scaling library.

All model code should import from this package — never from ``unit_scaling``
directly. The upstream library is re-exported in full here, and our
compile-friendly / world-size-aware replacements override the relevant names.

Anything available in ``unit_scaling`` is available here. Our overrides:
- ``scale_fwd``, ``scale_bwd`` — compile-friendly (setup_context pattern)
- ``linear``, ``rms_norm`` — world-size-aware
- ``residual_split``, ``residual_add`` — use our compile-friendly scale ops
- ``Linear``,  ``LinearReadout``, ``RMSNorm`` — world-size-aware nn.Module wrappers
- ``AdamW`` — MuP-aware with FSDP2 metadata caching
"""

# --- Pull in everything from upstream unit_scaling ---
from unit_scaling import *  # noqa: F401,F403
from unit_scaling import functional, optim  # noqa: F401 — submodules
from unit_scaling.functional import *  # noqa: F401,F403

# modules (world-size-aware)
from ._modules import (  # noqa: F811
    Linear as Linear,
    LinearReadout as LinearReadout,
    PerDimScale as PerDimScale,
    RMSNorm as RMSNorm,
)

# functional (world-size-aware + compile-friendly)
from .functional import (  # noqa: F811
    _get_effective_batch_multiplier as _get_effective_batch_multiplier,
    GRAD_ACCUMULATION_STEPS as GRAD_ACCUMULATION_STEPS,
    init_world_size_cache as init_world_size_cache,
    linear as linear,
    per_dim_scale as per_dim_scale,
    residual_add as residual_add,
    residual_split as residual_split,
    rms_norm as rms_norm,
    set_grad_accumulation_steps as set_grad_accumulation_steps,
    silu_glu as silu_glu,
    softplus as softplus,
)

# optim (MuP + FSDP2)
from .optim import (  # noqa: F811
    AdamW as AdamW,
    cache_fan_values as cache_fan_values,
    CombinedOptimizer as CombinedOptimizer,
    create_dion2 as create_dion2,
    create_normuon as create_normuon,
    get_cached_metadata as get_cached_metadata,
    polar_express as polar_express,
    polar_express_triton as polar_express_triton,
)

# --- Override with our compile-friendly / world-size-aware versions ---
# scale (compile-friendly)
from .scale import scale_bwd as scale_bwd, scale_fwd as scale_fwd  # noqa: F811
