# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Compile-friendly, world-size-aware unit scaling library.

All model code should import from this package — never from ``unit_scaling``
directly. The upstream library is re-exported in full here, and our
compile-friendly / world-size-aware replacements override the relevant names.

Anything available in ``unit_scaling`` is available here. Our overrides:
- ``scale_fwd``, ``scale_bwd`` — compile-friendly (setup_context pattern)
- ``linear``, ``rms_norm`` — world-size-aware
- ``residual_split``, ``residual_add`` — use our compile-friendly scale ops
- ``Linear``,  ``LinearReadout``, ``RMSNorm`` — world-size-aware nn.Module wrappers
- ``AdamW``, ``NorMuon``, ``Dion2`` — MuP-aware optimizers with FSDP2 support
"""

# --- Pull in everything from upstream unit_scaling ---
from unit_scaling import *  # noqa: F401,F403
from unit_scaling import functional, optim  # noqa: F401 — submodules
from unit_scaling.functional import *  # noqa: F401,F403

# modules (world-size-aware)
from ._modules import (  # noqa: F811
    Linear,
    LinearReadout,
    PerDimScale,
    RMSNorm,
)

# functional (world-size-aware + compile-friendly)
from .functional import (  # noqa: F811
    GRAD_ACCUMULATION_STEPS,
    _get_effective_batch_multiplier,
    init_world_size_cache,
    linear,
    per_dim_scale,
    residual_add,
    residual_split,
    rms_norm,
    set_grad_accumulation_steps,
    silu_glu,
    softplus,
)

# optim (MuP + FSDP2)
from .optim import (  # noqa: F811
    AdamW,
    Dion2,
    NorMuon,
    cache_fan_values,
    get_cached_metadata,
)

# scale (compile-friendly)
from .scale import (  # noqa: F811
    scale_bwd,
    scale_fwd,
)
