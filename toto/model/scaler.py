# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import warnings
from typing import Tuple, Type

import torch
from einops import reduce, repeat
from gluonts.core.component import validated
from gluonts.torch.scaler import Scaler


class StdMeanScaler(Scaler):
    """
    Scales data to have zero mean and unit variance along a given dimension.

    Parameters
    ----------
    dim
        dimension along which to compute the scale
    keepdim
        controls whether to retain dimension ``dim`` (of length 1) in the
        scale tensor, or suppress it.
    minimum_scale
        default scale that is used for elements that are constantly zero
        along dimension `dim`.
    """

    @validated()
    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = True,
        minimum_scale: float = 1e-3,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def __call__(
        self,
        data: torch.Tensor,
        padding_mask: torch.Tensor,
        weights: torch.Tensor,
        prefix_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert data.shape == weights.shape, "data and weights must have same shape"
        with torch.no_grad():
            if prefix_length is not None:
                # Create a prefix mask that is 1 for positions within prefix, 0 elsewhere
                prefix_mask = torch.zeros_like(weights)
                prefix_mask[..., :prefix_length] = 1.0
                # Apply prefix mask to weights instead of slicing
                weights = weights * prefix_mask

            weights = weights * padding_mask

            # We need to calculate the standard deviation using double-precision floats
            # to avoid overflow for extreme values.
            try:
                high_precision_data = data.to(torch.float64)
            except TypeError:
                # Certain backends (particularly MacOS/MPS) don't support float64.
                # In this case, we might be doing inference in float16
                # so it's still worthwhile to cast to float32 to avoid
                # some (but not all) overflow issues.
                warnings.warn(
                    f"Float64 is not supported by device {data.device}. "
                    "Using float32 instead for accumulating denominator in input scaler. "
                    "This may lead to overflow issues if the data contains extreme values.",
                    RuntimeWarning,
                )
                high_precision_data = data.to(torch.float32)

            denominator = weights.sum(self.dim, keepdim=self.keepdim).clamp_min(1.0).to(high_precision_data.dtype)
            means = (high_precision_data * weights).sum(self.dim, keepdim=self.keepdim) / denominator
            means = torch.nan_to_num(means)

            variance = (((high_precision_data - means) * weights) ** 2).sum(
                self.dim, keepdim=self.keepdim
            ) / denominator
            scale = torch.sqrt(variance + self.minimum_scale).to(data.dtype)
            loc = means.to(data.dtype)

            return (data - loc) / scale, loc, scale


def compute_causal_statistics(
    data: torch.Tensor,
    weights: torch.Tensor,
    padding_mask: torch.Tensor,
    dim: int,
    minimum_scale: float,
    use_bessel_correction: bool = True,
    stabilize_with_global: bool = False,
    scale_factor_exponent: float = 10.0,
    prefix_length: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute causal mean and scale statistics along a specified dimension using
    a vectorized implementation of Welford's algorithm for numerical stability.

    This implementation avoids explicit loops while maintaining the numerical stability
    of Welford's algorithm, achieving better performance with the same robustness
    against overflow issues.


    Can optionally use global statistics to stabilize causal statistics by clamping
    extreme values, preventing instability while preserving a relaxed version of the
    causal property. This allows a controlled amount of future information leakage,
    introducing an explicit tradeoff between causality and stability.
    extreme values, preventing instability while preserving the causal property.

    Parameters
    ----------
    data
        The input data tensor
    weights
        The weight tensor (same shape as data)
    padding_mask
        The padding mask tensor (same shape as data)
    dim
        The dimension along which to compute statistics (must be -1, the time dimension)
    minimum_scale
        Minimum scale value to use
    use_bessel_correction
        Whether to use Bessel's correction to get an unbiased estimator
    stabilize_with_global
        Whether to use global statistics to stabilize the causal statistics by clamping
        extreme values
    scale_factor_exponent
        Exponent that controls the allowed range of deviation from global scale.
        For example, with exponent=1.0, causal scale must be between 0.1x and 10x the global scale.
        With exponent=2.0, the range would be 0.01x to 100x.
    prefix_length
        If specified, the global statistics will be computed using only the prefix length
        requested. This is used for multistep decoding, where we only want to use the
        initial historical data to compute the global statistics. If stabilize_with_global
        is False, this parameter is ignored.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Causal mean and scale tensors, potentially stabilized with global statistics
    """
    # Assert that dim is -1 (last dimension)
    assert dim == -1, "compute_causal_statistics only supports dim=-1 (last dimension)"

    with torch.no_grad():
        # Apply padding mask to weights
        weights = weights * padding_mask

        # Try to use higher precision for numerical stability
        try:
            high_precision_data = data.to(torch.float64)
            high_precision_weights = weights.to(torch.float64)
        except TypeError:
            # Fallback for devices that don't support float64
            warnings.warn(
                f"Float64 is not supported by device {data.device}. "
                "Using float32 instead for causal scaler calculations. "
                "This may lead to numerical issues if the data contains extreme values.",
                RuntimeWarning,
            )
            high_precision_data = data.to(torch.float32)
            high_precision_weights = weights.to(torch.float32)

        # Check if deterministic algorithms are enabled and we're using CUDA.
        # Cumsum operations do not support deterministic mode in CUDA,
        # so we need to disable it for just this section.
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        if prev_deterministic and data.device.type == "cuda":
            # Disable deterministic algorithms for operations
            torch.use_deterministic_algorithms(False)

        try:
            # Create weighted data
            weighted_data = high_precision_weights * high_precision_data

            # Compute cumulative sum of weights and weighted data along time dimension
            cum_weights = torch.cumsum(high_precision_weights, dim=dim)
            cum_values = torch.cumsum(weighted_data, dim=dim)

            # Avoid division by zero for the first time step or when no valid values
            denominator = cum_weights.clamp_min(1.0)

            # Compute causal means at each time step
            causal_means = cum_values / denominator

            # For Welford's algorithm, we need to compute the correction term
            # using the difference between the current value and the current mean

            # Create shifted version of causal means to compute delta efficiently
            # First item in shifted_means will be zero
            shifted_means = torch.zeros_like(causal_means)
            shifted_means[..., 1:] = causal_means[..., :-1]

            # Compute delta between current data point and previous mean
            # For t=0, this is just the first data point
            delta = high_precision_data - shifted_means

            # Compute the increment term for Welford's algorithm.
            # This is defined as the product of the delta and the difference between the current data point and the causal mean.
            # This is where we avoid the traditional E[X²] - E[X]² computation
            increment = delta * (high_precision_data - causal_means) * high_precision_weights

            # The Welford algorithm uses the term m_2, which is the cumulative sum of the increment term.
            # This is an accumulator that helps us compute the second moment (hence m_2) of the distribution.
            # Compute cumulative sum of the increment term
            m_2 = torch.cumsum(increment, dim=dim)

            # Compute variance according to Welford's algorithm
            if use_bessel_correction:
                causal_variance = m_2 / torch.clamp(denominator - 1.0, min=1.0)
            else:
                causal_variance = m_2 / denominator

            # Add minimum scale but keep in high precision for now
            causal_scale = torch.sqrt(causal_variance + minimum_scale)

            # Apply stabilization with global statistics if requested
            if stabilize_with_global:
                if prefix_length is not None:
                    # Create a prefix mask for global statistics computation
                    prefix_mask = torch.zeros_like(weights)
                    prefix_mask[..., :prefix_length] = 1.0

                    # Apply prefix mask to restrict computation to prefix
                    weighted_data = weighted_data * prefix_mask
                    weights = weights * prefix_mask
                    padding_mask = padding_mask * prefix_mask

                # Calculate scale factors from the exponent
                scale_factor_min = 10.0 ** (-scale_factor_exponent)
                scale_factor_max = 10.0**scale_factor_exponent

                global_denominator = (weights * padding_mask).sum(dim, keepdim=True).clamp_min(1.0)
                global_means = (weighted_data).sum(dim, keepdim=True) / global_denominator
                global_means = torch.nan_to_num(global_means)

                global_variance = (((high_precision_data - global_means) * weights * padding_mask) ** 2).sum(
                    dim, keepdim=True
                ) / global_denominator
                global_scale = torch.sqrt(global_variance + minimum_scale)

                # Expand global statistics to match the time dimension
                expanded_global_scale = global_scale.expand_as(causal_scale)

                # Define bounds using scale factors
                min_allowed_scale = expanded_global_scale * scale_factor_min
                max_allowed_scale = expanded_global_scale * scale_factor_max

                # Clamp the causal scale between min_allowed_scale and max_allowed_scale
                causal_scale = torch.clamp(
                    causal_scale,
                    min=torch.max(torch.tensor(minimum_scale, device=causal_scale.device), min_allowed_scale),
                    max=max_allowed_scale,
                )

            # Now convert means and scale to original dtype after all numerical operations
            causal_means = causal_means.to(data.dtype)
            causal_scale = causal_scale.to(data.dtype)

        finally:
            # Restore original deterministic setting if it was changed
            if prev_deterministic and data.device.type == "cuda":
                torch.use_deterministic_algorithms(True)

        return causal_means, causal_scale


class CausalStdMeanScaler(Scaler):
    """
    Causally scales the data along dimension `dim` which is expected to be the
    time dimension. For each position t along this dimension, the mean and
    standard deviation are computed using only data from positions up to t.

    Can optionally stabilize causal statistics using global statistics to prevent
    extreme values, while preserving the causal property.

    Note: This scaler only works with dim=-1 (the last dimension).

    Parameters
    ----------
    dim
        dimension along which to compute the causal scale (must be -1, the last dimension)
    minimum_scale
        default scale that is used if the scale is below this threshold
        or for the first time step, since standard deviation cannot be
        computed with a single observation
    use_bessel_correction
        whether to use Bessel's correction to get an unbiased estimator
    stabilize_with_global
        whether to use global statistics to stabilize extreme causal statistics
    scale_factor_exponent
        exponent that controls the allowed range of deviation from global scale.
        For example, with exponent=1.0, causal scale must be between 0.1x and 10x the global scale.
        With exponent=2.0, the range would be 0.01x to 100x.
    """

    @validated()
    def __init__(
        self,
        dim: int = -1,
        minimum_scale: float = 0.1,
        use_bessel_correction: bool = True,
        stabilize_with_global: bool = False,
        scale_factor_exponent: float = 10.0,
    ) -> None:
        super().__init__()
        assert dim == -1, "CausalStdMeanScaler only supports dim=-1 (last dimension)"
        self.dim = dim
        self.minimum_scale = minimum_scale
        self.use_bessel_correction = use_bessel_correction
        self.stabilize_with_global = stabilize_with_global
        self.scale_factor_exponent = scale_factor_exponent

    def __call__(
        self,
        data: torch.Tensor,
        padding_mask: torch.Tensor,
        weights: torch.Tensor,
        prefix_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert data.shape == weights.shape, "data and weights must have same shape"
        assert len(data.shape) == 3, "Input data must have shape [batch, variates, time_steps]"

        # Compute causal statistics with optional stabilization
        causal_means, causal_scale = compute_causal_statistics(
            data,
            weights,
            padding_mask,
            self.dim,
            self.minimum_scale,
            self.use_bessel_correction,
            self.stabilize_with_global,
            self.scale_factor_exponent,
            prefix_length,
        )

        # Apply the normalization
        scaled_data = (data - causal_means) / causal_scale

        return scaled_data, causal_means, causal_scale


class CausalPatchStdMeanScaler(Scaler):
    """
    Causally scales data in patches, where each patch uses statistics computed
    from all data up to and including that patch. Within each patch, all timesteps
    use the same scaling values.

    This approach provides more stability than per-timestep causal scaling while
    still maintaining the causal property (not using future data).

    Can optionally stabilize causal statistics using global statistics to prevent
    extreme values, while preserving the causal property.

    The statistics are computed using Welford's algorithm, which provides better
    numerical stability compared to the direct computation of variance, especially
    when dealing with large values or a large number of data points.

    Note: This scaler only works with the following constraints:
    - The input must have shape [batch, variates, time_steps]
    - It only operates on the last dimension (-1)
    - The time_steps must be divisible by patch_size

    Parameters
    ----------
    dim
        dimension along which to compute the causal scale. Must be -1 (the last dimension).
    patch_size
        number of timesteps in each patch
    minimum_scale
        default scale that is used for elements that are constantly zero
        along dimension `dim` or for the first patch.
    use_bessel_correction
        whether to use Bessel's correction to get an unbiased estimator
    stabilize_with_global
        whether to use global statistics to stabilize extreme causal statistics
    scale_factor_exponent
        exponent that controls the allowed range of deviation from global scale.
        For example, with exponent=1.0, causal scale must be between 0.1x and 10x the global scale.
        With exponent=2.0, the range would be 0.01x to 100x.
    """

    @validated()
    def __init__(
        self,
        dim: int = -1,
        patch_size: int = 32,
        minimum_scale: float = 0.1,
        use_bessel_correction: bool = True,
        stabilize_with_global: bool = False,
        scale_factor_exponent: float = 10.0,
    ) -> None:
        super().__init__()
        assert dim == -1, "CausalPatchStdMeanScaler only supports dim=-1 (last dimension)"
        self.dim = dim
        self.patch_size = patch_size
        self.minimum_scale = minimum_scale
        self.use_bessel_correction = use_bessel_correction
        self.stabilize_with_global = stabilize_with_global
        self.scale_factor_exponent = scale_factor_exponent

    def __call__(
        self,
        data: torch.Tensor,
        padding_mask: torch.Tensor,
        weights: torch.Tensor,
        prefix_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert data.shape == weights.shape, "data and weights must have same shape"
        assert len(data.shape) == 3, "Input data must have shape [batch, variates, time_steps]"

        with torch.no_grad():
            # Get the number of time steps (last dimension)
            time_steps = data.shape[-1]

            # Assert that time_steps is divisible by patch_size
            assert (
                time_steps % self.patch_size == 0
            ), f"Time steps ({time_steps}) must be divisible by patch size ({self.patch_size})"

            # First compute causal statistics with optional stabilization
            causal_means, causal_scale = compute_causal_statistics(
                data,
                weights,
                padding_mask,
                -1,
                self.minimum_scale,
                self.use_bessel_correction,
                self.stabilize_with_global,
                self.scale_factor_exponent,
                prefix_length,
            )

            # Unfold the causal means and scales to get the patches
            means_unfolded = causal_means.unfold(-1, self.patch_size, self.patch_size)
            scales_unfolded = causal_scale.unfold(-1, self.patch_size, self.patch_size)

            # Get the last element of each patch (the most recent statistic)
            patch_stats_means = means_unfolded[..., -1]
            patch_stats_scales = scales_unfolded[..., -1]

            # Tile the patch statistics across time dimension using einops.repeat
            # With our fixed [batch, variates, num_patches] shape this is much simpler
            patch_means = repeat(patch_stats_means, "b v p -> b v (p s)", s=self.patch_size)
            patch_scales = repeat(patch_stats_scales, "b v p -> b v (p s)", s=self.patch_size)

            # Apply normalization
            scaled_data = (data - patch_means) / patch_scales

            return scaled_data, patch_means, patch_scales


# for deserialization of SafeTensors checkpoints
scaler_types = {
    "<class 'model.scaler.StdMeanScaler'>": StdMeanScaler,
    "<class 'model.scaler.CausalStdMeanScaler'>": CausalStdMeanScaler,
    "<class 'model.scaler.CausalPatchStdMeanScaler'>": CausalPatchStdMeanScaler,
}
