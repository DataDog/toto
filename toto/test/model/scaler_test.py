# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import os
import sys

import torch

from toto.model.scaler import CausalPatchStdMeanScaler, CausalStdMeanScaler

MIN_SCALE = 1e-6

# Test data fixture with 1 batch, 2 variates, and 6 time steps
# [batch, variates, time_steps]
TEST_DATA = torch.tensor(
    [
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # First variate (steady increase)
            [2.0, -5.0, 1.0, 6.0, -3.0, 8.0],  # Second variate (oscillating)
        ]
    ],
    dtype=torch.float32,
)


def compute_simple_scale(
    values: torch.Tensor, minimum_scale: float, use_bessel_correction: bool = True
) -> torch.Tensor:
    """
    Compute the scale for a given set of values.
    """
    variance = torch.var(values, correction=1 if use_bessel_correction else 0)
    return torch.sqrt(torch.nan_to_num(variance, 0) + minimum_scale)


def test_causal_patch_std_mean_scaler():
    """
    Test the CausalPatchStdMeanScaler with a simple example.
    We'll use a batch size of 1, 2 variates, and 6 time steps with
    patch_size=2. This will give us 3 patches, and we'll manually
    calculate the expected values.
    """
    # Set up a fixed random seed for reproducibility
    torch.manual_seed(42)

    # Create a simple example with 1 batch, 2 variates, and 6 time steps
    data = TEST_DATA.clone()

    # All values are valid, so use all ones for padding_mask and weights
    padding_mask = torch.ones_like(data)
    weights = torch.ones_like(data)

    # Create the scaler with patch_size=2 and a small minimum_scale
    patch_size = 2
    scaler = CausalPatchStdMeanScaler(
        dim=-1, patch_size=patch_size, minimum_scale=MIN_SCALE, use_bessel_correction=False
    )

    # Apply the scaler
    scaled_data, patch_means, patch_scales = scaler(data, padding_mask, weights)

    expected_means = torch.tensor(
        [
            [
                [
                    torch.mean(data[0, 0, :2]),
                    torch.mean(data[0, 0, :2]),
                    torch.mean(data[0, 0, :4]),
                    torch.mean(data[0, 0, :4]),
                    torch.mean(data[0, 0, :6]),
                    torch.mean(data[0, 0, :6]),
                ],  # First variate
                [
                    torch.mean(data[0, 1, :2]),
                    torch.mean(data[0, 1, :2]),
                    torch.mean(data[0, 1, :4]),
                    torch.mean(data[0, 1, :4]),
                    torch.mean(data[0, 1, :6]),
                    torch.mean(data[0, 1, :6]),
                ],  # Second variate
            ]
        ],
        dtype=torch.float32,
    )

    # Compute the causal scales using a naive (O(n^2)) approach
    # with torch.var as a baseline
    expected_scales = torch.tensor(
        [
            [
                [
                    compute_simple_scale(data[0, 0, :2], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :2], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :4], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :4], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :6], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :6], MIN_SCALE, False),
                ],  # First variate
                [
                    compute_simple_scale(data[0, 1, :2], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :2], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :4], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :4], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :6], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :6], MIN_SCALE, False),
                ],  # Second variate
            ]
        ],
        dtype=torch.float32,
    )

    # Step 4: Expected scaled_data
    # scaled_data = (data - patch_means) / patch_scales

    expected_scaled_data = (data - expected_means) / expected_scales

    # Check that our implementation matches the expected values
    torch.testing.assert_close(patch_means, expected_means, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(patch_scales, expected_scales, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(scaled_data, expected_scaled_data, rtol=1e-7, atol=1e-7)


def test_causal_patch_std_mean_scaler_with_bessel_correction():
    """
    Test the CausalPatchStdMeanScaler with Bessel's correction.
    """
    torch.manual_seed(42)

    # Create a simple example with 1 batch, 2 variates, and 6 time steps
    data = TEST_DATA.clone()

    # Create the scaler with patch_size=2 and Bessel's correction
    patch_size = 2
    scaler = CausalPatchStdMeanScaler(
        dim=-1, patch_size=patch_size, minimum_scale=MIN_SCALE, use_bessel_correction=True
    )

    # Apply the scaler with default padding mask and weights
    padding_mask = torch.ones_like(data)
    weights = torch.ones_like(data)
    scaled_data, patch_means, patch_scales = scaler(data, padding_mask, weights)

    expected_means = torch.tensor(
        [
            [
                [
                    torch.mean(data[0, 0, :2]),
                    torch.mean(data[0, 0, :2]),
                    torch.mean(data[0, 0, :4]),
                    torch.mean(data[0, 0, :4]),
                    torch.mean(data[0, 0, :6]),
                    torch.mean(data[0, 0, :6]),
                ],  # First variate
                [
                    torch.mean(data[0, 1, :2]),
                    torch.mean(data[0, 1, :2]),
                    torch.mean(data[0, 1, :4]),
                    torch.mean(data[0, 1, :4]),
                    torch.mean(data[0, 1, :6]),
                    torch.mean(data[0, 1, :6]),
                ],  # Second variate
            ],
        ],
        dtype=torch.float32,
    )

    # Compute the causal scales using a naive (O(n^2)) approach
    # with torch.var as a baseline
    expected_scales = torch.tensor(
        [
            [
                [
                    compute_simple_scale(data[0, 0, :2], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :2], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :4], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :4], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :6], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :6], MIN_SCALE, True),
                ],  # First variate
                [
                    compute_simple_scale(data[0, 1, :2], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :2], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :4], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :4], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :6], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :6], MIN_SCALE, True),
                ],  # Second variate
            ],
        ],
        dtype=torch.float32,
    )

    expected_scaled_data = (data - expected_means) / expected_scales

    # Check that our implementation matches the expected values
    torch.testing.assert_close(patch_means, expected_means, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(patch_scales, expected_scales, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(scaled_data, expected_scaled_data, rtol=1e-7, atol=1e-7)


def test_causal_patch_std_mean_scaler_with_padding():
    """
    Test the CausalPatchStdMeanScaler with padding and weights.
    We'll use a similar setup to the previous test but with some values
    padded out.
    """
    torch.manual_seed(42)

    # Create a simple example with 1 batch, 2 variates, and 6 time steps
    data = TEST_DATA.clone()

    # Set up padding_mask to mask out the last two time steps for the second variate
    padding_mask = torch.ones_like(data)
    # Mask out t=4,5 for the second variate
    padding_mask[0, 1, 4:] = 0.0

    # Also set up weights to give less weight to some values
    weights = torch.ones_like(data)
    weights[0, 0, 1] = 0.5  # Give half weight to t=1 for first variate

    # Create the scaler with patch_size=2
    patch_size = 2
    scaler = CausalPatchStdMeanScaler(dim=-1, patch_size=patch_size, minimum_scale=MIN_SCALE)

    # Apply the scaler
    scaled_data, patch_means, patch_scales = scaler(data, padding_mask, weights)

    # For this test, we won't calculate everything by hand,
    # but we'll check a few key properties:

    # 1. The patch_means and patch_scales should be constant within each patch
    for i in range(0, data.shape[-1], patch_size):
        end = min(i + patch_size, data.shape[-1])
        for b in range(data.shape[0]):
            for v in range(data.shape[1]):
                # Check means are constant within patch
                assert torch.allclose(
                    patch_means[b, v, i:end],
                    torch.full((end - i,), patch_means[b, v, i].item(), dtype=data.dtype),
                    rtol=1e-7,
                )
                # Check scales are constant within patch
                assert torch.allclose(
                    patch_scales[b, v, i:end],
                    torch.full((end - i,), patch_scales[b, v, i].item(), dtype=data.dtype),
                    rtol=1e-7,
                )

    # 2. The padded values should not affect the statistics
    # For the second variate (index 1), verify that the last patch statistics
    # are based on valid data only

    # Instead of calculating the expected value manually, extract the value
    # from the previous patch since we know that's based on valid data up to t=3
    expected_mean = patch_means[0, 1, 2].item()
    expected_scale = patch_scales[0, 1, 2].item()

    # The last patch should use statistics only from valid data
    assert abs(patch_means[0, 1, 4].item() - expected_mean) < 1e-3
    assert abs(patch_scales[0, 1, 4].item() - expected_scale) < 1e-3


def test_causal_std_mean_scaler():
    """
    Test the CausalStdMeanScaler with the same example as the patch version.
    Each time step gets its own causal statistics instead of using patches.
    """
    # Set up a fixed random seed for reproducibility
    torch.manual_seed(42)

    # Create a simple example with 1 batch, 2 variates, and 6 time steps
    data = TEST_DATA.clone()

    # All values are valid, so use all ones for padding_mask and weights
    padding_mask = torch.ones_like(data)
    weights = torch.ones_like(data)

    # Create the scaler with a small minimum_scale
    scaler = CausalStdMeanScaler(dim=-1, minimum_scale=MIN_SCALE, use_bessel_correction=False)

    # Apply the scaler
    scaled_data, causal_means, causal_scales = scaler(data, padding_mask, weights)

    # Create the expected outputs
    expected_means = torch.tensor(
        [
            [
                [
                    torch.mean(data[0, 0, :1]),
                    torch.mean(data[0, 0, :2]),
                    torch.mean(data[0, 0, :3]),
                    torch.mean(data[0, 0, :4]),
                    torch.mean(data[0, 0, :5]),
                    torch.mean(data[0, 0, :6]),
                ],  # First variate
                [
                    torch.mean(data[0, 1, :1]),
                    torch.mean(data[0, 1, :2]),
                    torch.mean(data[0, 1, :3]),
                    torch.mean(data[0, 1, :4]),
                    torch.mean(data[0, 1, :5]),
                    torch.mean(data[0, 1, :6]),
                ],  # Second variate
            ]
        ],
        dtype=torch.float32,
    )

    # Compute the causal scales using a naive (O(n^2)) approach
    # with torch.var as a baseline
    expected_scales = torch.tensor(
        [
            [
                [
                    compute_simple_scale(data[0, 0, :1], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :2], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :3], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :4], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :5], MIN_SCALE, False),
                    compute_simple_scale(data[0, 0, :6], MIN_SCALE, False),
                ],  # First variate
                [
                    compute_simple_scale(data[0, 1, :1], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :2], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :3], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :4], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :5], MIN_SCALE, False),
                    compute_simple_scale(data[0, 1, :6], MIN_SCALE, False),
                ],  # Second variate
            ]
        ],
        dtype=torch.float32,
    )

    # Expected scaled_data = (data - expected_means) / expected_scales
    expected_scaled_data = (data - expected_means) / expected_scales

    # Check that our implementation matches the expected values
    torch.testing.assert_close(causal_means, expected_means, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(causal_scales, expected_scales, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(scaled_data, expected_scaled_data, rtol=1e-7, atol=1e-7)

    # Also verify the key difference between CausalStdMeanScaler
    # and CausalPatchStdMeanScaler:
    # Each time step should have its own statistics (not constant within patches)
    for t in range(1, data.shape[-1]):
        # Check that adjacent time steps have different means/scales
        assert causal_means[0, 0, t] != causal_means[0, 0, t - 1]
        assert causal_scales[0, 0, t] != causal_scales[0, 0, t - 1]


def test_causal_std_mean_scaler_with_bessel_correction():
    """
    Test the CausalStdMeanScaler with Bessel correction.
    """
    torch.manual_seed(42)

    # Create a simple example with 1 batch, 2 variates, and 6 time steps
    data = TEST_DATA.clone()

    # Create padding mask and weights (all ones for this test)
    padding_mask = torch.ones_like(data)
    weights = torch.ones_like(data)

    # Create the scaler with Bessel correction (default is True)
    scaler = CausalStdMeanScaler(dim=-1, minimum_scale=MIN_SCALE, use_bessel_correction=True)

    # Apply the scaler
    scaled_data, causal_means, causal_scales = scaler(data, padding_mask, weights)

    # Create the expected outputs
    expected_means = torch.tensor(
        [
            [
                [
                    torch.mean(data[0, 0, :1]),
                    torch.mean(data[0, 0, :2]),
                    torch.mean(data[0, 0, :3]),
                    torch.mean(data[0, 0, :4]),
                    torch.mean(data[0, 0, :5]),
                    torch.mean(data[0, 0, :6]),
                ],  # First variate
                [
                    torch.mean(data[0, 1, :1]),
                    torch.mean(data[0, 1, :2]),
                    torch.mean(data[0, 1, :3]),
                    torch.mean(data[0, 1, :4]),
                    torch.mean(data[0, 1, :5]),
                    torch.mean(data[0, 1, :6]),
                ],  # Second variate
            ]
        ],
        dtype=torch.float32,
    )

    # Compute the causal scales using a naive (O(n^2)) approach
    # with torch.var as a baseline
    expected_scales = torch.tensor(
        [
            [
                [
                    compute_simple_scale(data[0, 0, :1], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :2], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :3], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :4], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :5], MIN_SCALE, True),
                    compute_simple_scale(data[0, 0, :6], MIN_SCALE, True),
                ],  # First variate
                [
                    compute_simple_scale(data[0, 1, :1], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :2], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :3], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :4], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :5], MIN_SCALE, True),
                    compute_simple_scale(data[0, 1, :6], MIN_SCALE, True),
                ],  # Second variate
            ]
        ],
        dtype=torch.float32,
    )

    # Expected scaled_data = (data - expected_means) / expected_scales
    expected_scaled_data = (data - expected_means) / expected_scales

    # Check that our implementation matches the expected values
    torch.testing.assert_close(causal_means, expected_means, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(causal_scales, expected_scales, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(scaled_data, expected_scaled_data, rtol=1e-7, atol=1e-7)


def test_causal_std_mean_scaler_with_padding():
    """
    Test the CausalStdMeanScaler with padding and weights.
    """
    torch.manual_seed(42)

    # Create a simple example with 1 batch, 2 variates, and 6 time steps
    data = TEST_DATA.clone()

    # Set up padding_mask to mask out the last two time steps for the second variate
    padding_mask = torch.ones_like(data)
    padding_mask[0, 1, 4:] = 0.0  # Mask out t=4,5 for the second variate

    # Also set up weights to give less weight to some values
    weights = torch.ones_like(data)
    weights[0, 0, 1] = 0.5  # Give half weight to t=1 for first variate

    # Create the scaler
    scaler = CausalStdMeanScaler(dim=-1, minimum_scale=MIN_SCALE)

    # Apply the scaler
    scaled_data, causal_means, causal_scales = scaler(data, padding_mask, weights)

    # For this test, we'll check that padded values don't affect the statistics

    # For the second variate (index 1), the statistics at t=4,5 should be
    # the same as at t=3 since t=4,5 are padded out
    assert torch.allclose(causal_means[0, 1, 4], causal_means[0, 1, 3], rtol=1e-7)
    assert torch.allclose(causal_means[0, 1, 5], causal_means[0, 1, 3], rtol=1e-7)
    assert torch.allclose(causal_scales[0, 1, 4], causal_scales[0, 1, 3], rtol=1e-7)
    assert torch.allclose(causal_scales[0, 1, 5], causal_scales[0, 1, 3], rtol=1e-7)

    # For the first variate (index 0), the statistics should continue to change
    # since all time steps are valid
    for t in range(1, data.shape[-1]):
        assert causal_means[0, 0, t] != causal_means[0, 0, t - 1]
