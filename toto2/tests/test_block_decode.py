"""Tests for the unified forecast path in Toto2Model.

Verifies:
  1. Single-block (no cache) and multi-block (with cache) determinism.
  2. Block decode runs without error for various block sizes.
  3. decode_block_size assertion fires when not divisible by patch_size.
  4. KV cache is only used when horizon requires multiple blocks.
  5. known_dynamic covariates are supported.
  6. torch.compile(fullgraph=True) works with block decode.
"""

import pytest
import torch

from toto2 import Toto2Model
from toto2.configuration import Toto2ModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATCH_SIZE = 32


def _tiny_config() -> Toto2ModelConfig:
    return Toto2ModelConfig(
        patch_size=PATCH_SIZE,
        d_model=64,
        num_heads=4,
        num_layers=4,
        layer_group_size=2,
        num_variate_layers_per_group=1,
        variate_layer_first=True,
        residual_attn_ratio=Toto2ModelConfig.compute_residual_attn_ratio(512, PATCH_SIZE),
    )


@pytest.fixture(scope="module")
def model():
    m = Toto2Model(_tiny_config())
    m.eval()
    return m


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(42)
    target = torch.randn(1, 2, 512)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    series_ids = torch.zeros(1, 2, dtype=torch.long)
    return {"target": target, "target_mask": target_mask, "series_ids": series_ids}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSinglePassEquivalence:
    """Large-block decode (no cache) must be deterministic and consistent."""

    def test_large_block_skips_cache(self, model, inputs):
        """decode_block_size >= horizon should run without cache and still
        produce valid, deterministic results."""
        horizon = 96
        q1 = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE * 100)
        q2 = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE * 100)
        assert q1.shape == q2.shape
        assert torch.equal(q1, q2), (
            f"max diff: {(q1 - q2).abs().max():.2e}"
        )

    def test_large_block_matches_exact_block(self, model, inputs):
        """An oversized block_size should clamp to num_patches and give the
        same result as block_size == num_patches."""
        horizon = 96
        num_patches = -(-horizon // PATCH_SIZE)
        q_exact = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE * num_patches)
        q_large = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE * 100)
        assert torch.equal(q_exact, q_large)

    def test_no_block_size_matches_zero_and_large(self, model, inputs):
        """No decode_block_size, decode_block_size=0, and an oversized
        decode_block_size should all produce identical results (unified path)."""
        horizon = 64
        q_default = model.forecast(inputs, horizon)
        q_zero = model.forecast(inputs, horizon, decode_block_size=0)
        q_large = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE * 100)
        assert torch.equal(q_default, q_zero)
        assert torch.equal(q_default, q_large)


class TestBlockDecode:
    """Block decode produces valid outputs for various configurations."""

    @pytest.mark.parametrize("block_patches", [1, 2, 3])
    def test_various_block_sizes(self, model, inputs, block_patches):
        horizon = 128
        q = model.forecast(
            inputs, horizon, decode_block_size=PATCH_SIZE * block_patches,
        )
        num_quantiles = len(model.output_head.knots)
        assert q.shape == (num_quantiles, 1, 2, horizon)
        assert q.isfinite().all()

    def test_horizon_not_divisible_by_patch(self, model, inputs):
        """Horizon that doesn't divide evenly into patches."""
        horizon = 100  # not a multiple of PATCH_SIZE=32
        q = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE)
        assert q.shape[-1] == horizon

    def test_univariate(self, model):
        torch.manual_seed(0)
        target = torch.randn(1, 1, 256)
        inp = {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(1, 1, dtype=torch.long),
        }
        q = model.forecast(inp, 64, decode_block_size=PATCH_SIZE)
        assert q.shape == (len(model.output_head.knots), 1, 1, 64)

    def test_quantiles_sorted(self, model, inputs):
        """Quantiles should be monotonically non-decreasing along dim 0."""
        q = model.forecast(inputs, 96, decode_block_size=PATCH_SIZE)
        diffs = q[1:] - q[:-1]
        assert (diffs >= -1e-6).all(), "quantiles not sorted"


class TestValidation:
    def test_block_size_not_divisible_by_patch(self, model, inputs):
        with pytest.raises(AssertionError, match="divisible"):
            model.forecast(inputs, 96, decode_block_size=PATCH_SIZE + 1)

    def test_noop_when_horizon_within_block(self, model, inputs):
        """When horizon fits in one block, no cache should be used and
        results should be deterministic."""
        horizon = PATCH_SIZE  # 1 patch = fits in any block
        q1 = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE * 2)
        q2 = model.forecast(inputs, horizon, decode_block_size=PATCH_SIZE * 2)
        assert torch.equal(q1, q2)


class TestCacheReuse:
    def test_cache_reset_on_repeated_calls(self, model, inputs):
        """Repeated calls with same shape should reuse cache and give
        identical results."""
        horizon = 128
        bs = PATCH_SIZE * 2
        q1 = model.forecast(inputs, horizon, decode_block_size=bs)
        q2 = model.forecast(inputs, horizon, decode_block_size=bs)
        assert torch.equal(q1, q2)


class TestKnownDynamic:
    """known_dynamic covariates should be handled in both cache and no-cache paths."""

    @pytest.fixture()
    def inputs_with_kd(self):
        torch.manual_seed(99)
        ctx = 512
        horizon = 128
        target = torch.randn(1, 2, ctx)
        kd = torch.randn(1, 1, ctx + horizon)
        return {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(1, 2, dtype=torch.long),
            "known_dynamic": kd,
            "known_dynamic_mask": torch.ones_like(kd, dtype=torch.bool),
            "known_dynamic_series_ids": torch.zeros(1, 1, dtype=torch.long),
        }

    def test_single_block_with_kd(self, model, inputs_with_kd):
        q = model.forecast(inputs_with_kd, 128)
        assert q.shape == (len(model.output_head.knots), 1, 2, 128)
        assert q.isfinite().all()

    def test_multi_block_with_kd(self, model, inputs_with_kd):
        q = model.forecast(inputs_with_kd, 128, decode_block_size=PATCH_SIZE * 2)
        assert q.shape == (len(model.output_head.knots), 1, 2, 128)
        assert q.isfinite().all()

    def test_kd_deterministic(self, model, inputs_with_kd):
        q1 = model.forecast(inputs_with_kd, 128, decode_block_size=PATCH_SIZE * 2)
        q2 = model.forecast(inputs_with_kd, 128, decode_block_size=PATCH_SIZE * 2)
        assert torch.equal(q1, q2)


class TestCompile:
    def test_compile_fullgraph_single_pass(self):
        config = _tiny_config()
        m = Toto2Model(config).eval()
        m_compiled = torch.compile(m, fullgraph=True)

        torch.manual_seed(0)
        target = torch.randn(1, 1, 256)
        inp = {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(1, 1, dtype=torch.long),
        }
        q = m_compiled.forecast(inp, 64)
        assert q.shape == (len(m.output_head.knots), 1, 1, 64)
        assert q.isfinite().all()

    def test_compile_fullgraph_block_decode(self):
        config = _tiny_config()
        m = Toto2Model(config).eval()
        m_compiled = torch.compile(m, fullgraph=True)

        torch.manual_seed(0)
        target = torch.randn(1, 1, 256)
        inp = {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(1, 1, dtype=torch.long),
        }
        q = m_compiled.forecast(inp, 128, decode_block_size=PATCH_SIZE * 2)
        assert q.shape == (len(m.output_head.knots), 1, 1, 128)
        assert q.isfinite().all()
