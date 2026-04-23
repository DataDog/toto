# dd-unit-scaling

A production-ready, thin wrapper around [graphcore-research/unit-scaling](https://github.com/graphcore-research/unit-scaling) that makes u-μP work reliably with `torch.compile`, FSDP2, and distributed training at scale. Used to train [Toto 2.0](../README.md#toto-20).

## Background

### Why u-μP?

Toto 2.0 uses [u-μP](https://arxiv.org/abs/2407.17465) (Unit-Scaled Maximal Update Parameterization) rather than standard [μP](https://arxiv.org/abs/2203.03466). Standard μP requires running a base model to compute reference scales, then transferring those scales to the target model size. u-μP eliminates this — scaling factors are derived directly from layer fan-in/fan-out, so there is no base model metadata to manage. u-μP has also been [shown to converge better on decoder-only transformers](https://arxiv.org/abs/2407.17465).

### Why a thin wrapper?

The upstream `unit_scaling` library implements u-μP correctly in principle, but has accumulated small issues that prevent it from working with `torch.compile` and distributed training at scale. Rather than fork the entire library, `dd-unit-scaling` re-exports everything from upstream and overrides only the broken pieces. This keeps the surface area minimal and makes it easy to drop if upstream fixes land.

## Installation

```bash
pip install "dd-unit-scaling @ git+https://github.com/DataDog/toto.git#subdirectory=dd_unit_scaling"
```

For Muon-family optimizers, also install [dion](https://github.com/microsoft/dion):

```bash
pip install git+https://github.com/microsoft/dion.git
```

## Usage

Drop-in replacement for `unit_scaling`:

```python
import dd_unit_scaling as uu
from dd_unit_scaling import functional as U

out = U.linear(x, weight, bias)
normed = U.rms_norm(x, normalized_shape=(dim,), weight=w)
```

Optimizers:

```python
# AdamW for bias/norm/output params
opt_adam = uu.AdamW(bias_params, lr=1e-3)

# Muon-family for weight params
opt_muon = uu.Muon(weight_params, lr=0.02)
opt_dion2 = uu.Dion2(weight_params, lr=0.02)
opt_normuon = uu.NorMuon(weight_params, lr=0.02, use_polar_express=True)
```

## Features

### `torch.compile` fixes

Upstream `unit_scaling` has a collection of issues that break `torch.compile` — `isinstance` checks on `fx.proxy.Proxy`, ints where dynamo expects floats, and other tracing-unfriendly patterns. This package fixes all of them. Core primitives (`scale_fwd`, `scale_bwd`) are rewritten using PyTorch 2.x's `setup_context` pattern, and residual ops and activations are reimplemented on top of them.

### World-size-aware scaling

In DDP/FSDP, gradients are averaged across workers, so batch-dependent scale factors need to account for the full effective batch: `local_batch × world_size × grad_accumulation_steps`. Call `init_world_size_cache()` before `torch.compile` to cache the world size as a plain int (avoiding graph breaks from process group calls).

### u-μP optimizers with FSDP2 support

Provides `AdamW`, `Muon`, `Dion2`, and `NorMuon` wrappers that apply per-parameter u-μP LR scaling. Metadata (fan-in, fan-out, mup_type) is cached by parameter name before FSDP wrapping, since FSDP2 replaces parameter tensors with DTensors. Includes [Polar Express](https://arxiv.org/abs/2505.16932) orthogonalization as an alternative to Newton-Schulz for NorMuon (also used in [Karpathy's nanochat](https://github.com/karpathy/nanochat/discussions/481)).

## Distributed training setup

When training with DDP/FSDP, three caching calls must happen in a specific order:

```python
import dd_unit_scaling as uu

# 1. Build model (param shapes are still intact here)
model = MyModel()

# 2. Cache μP metadata BEFORE FSDP wrapping
uu.cache_fan_values(model.named_parameters())

# 3. Apply FSDP/DDP wrapping (replaces param tensors with DTensors)
fully_shard(model)

# 4. Set world size and grad accumulation BEFORE torch.compile
#    (these become plain ints that dynamo can trace without graph breaks)
uu.init_world_size_cache(world_size=dist.get_world_size())
uu.set_grad_accumulation_steps(accum_steps)

# 5. Compile
model = torch.compile(model)

# 6. Create optimizers (they read the cached metadata)
opt = uu.AdamW(bias_params, lr=1e-3)
opt_muon = uu.Dion2(weight_params, lr=0.02)
```

- `cache_fan_values` must happen **before** FSDP wrapping because FSDP replaces parameter tensors with DTensors, changing their shapes. The cache stores fan-in/fan-out/mup_type by parameter name so the optimizers can look them up later.
- `init_world_size_cache` and `set_grad_accumulation_steps` must happen **before** `torch.compile` so the values are baked in as plain ints. If called after compile, they cause graph breaks.

### Setting `world_size` by parallelism strategy

The goal is to make each GPU's local `input.numel()` reflect the global batch — as if all data were on a single device. The effective global batch is `local_batch × world_size × grad_accumulation_steps`.

`world_size` should be the product of all parallelism dimensions where ranks process **different data**:

| Dimension | Different data? | Counts toward `world_size`? |
|---|---|---|
| DP (data parallel, incl. FSDP) | Yes — each rank sees a different batch | Yes |
| CP (context parallel) | Yes — each rank sees different sequence chunks | Yes |
| TP (tensor parallel) | No — ranks split the same input across heads/features | No |
| SP (sequence parallel) | No — activation sharding within TP groups | No |
| PP (pipeline parallel) | No — ranks process different stages of the same micro-batch | No |
| EP (expert parallel) | No — same batch, different experts | No |

**`world_size = dp × cp`** (where `dp` is the total data-parallel degree, whether DDP, FSDP, or HSDP).

```python
# Examples
uu.init_world_size_cache(dp)                # DDP or FSDP only
uu.init_world_size_cache(dp * cp)           # DP + CP
uu.init_world_size_cache(dp * cp)           # DP + CP + TP + PP (TP/PP don't count)
# Single GPU — no call needed (defaults to 1)
```

If you use gradient accumulation, also set:

```python
uu.set_grad_accumulation_steps(accum_steps)  # defaults to 1
```

## Design note: sequence-length-independent scaling in Toto 2.0

This isn't a feature of `dd-unit-scaling` itself, but a design choice in Toto 2.0 worth calling out. Toto 2.0 uses unscaled `F.scaled_dot_product_attention` (PyTorch native SDPA) instead of unit-scaled SDPA, so no scale factors depend on sequence length — making the model compatible with KV-cache inference. The resulting attn/MLP variance imbalance is compensated via `residual_attn_ratio = sqrt(S / log(S))` (where `S = context_length / patch_size`), which adjusts the residual tau values so that attention branches get proportionally more weight. `residual_mult` is set to `0.75` (the `unit_scaling` default is `1.0`).

## Requirements

- Python >= 3.12
- PyTorch >= 2.4.0
- [unit-scaling](https://github.com/graphcore-research/unit-scaling) >= 0.2.0
- [dion](https://github.com/microsoft/dion) (optional, for Muon/Dion2/NorMuon)
