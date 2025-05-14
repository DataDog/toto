# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from functools import reduce
from typing import NamedTuple

import numpy as np
import torch
import torch.utils.data
from einops import repeat
from jaxtyping import Bool, Float, Int, Shaped


def pad_array(
    values: Shaped[torch.Tensor, "*batch variates series_len"],  # noqa: F722
    patch_stride: int,
) -> Shaped[torch.Tensor, "*batch variates padded_length"]:  # noqa: F722
    """
    Makes sure that the series length is divisible by the patch_stride
    by adding left-padding.
    """
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
    series_len = values.shape[-1]
    # left-pad the time series to make sure we can divide it into patches.
    padded_length = int(np.ceil(series_len / patch_stride) * patch_stride)
    if values.ndim == 2:  # variates series_len
        padded_values = torch.zeros((values.shape[0], padded_length), dtype=values.dtype, device=values.device)
    elif values.ndim == 3:  # batch variates series_len
        padded_values = torch.zeros(
            (values.shape[0], values.shape[1], padded_length),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        raise ValueError(f"Unsupported number of dimensions: {values.ndim}")
    padded_values[..., -series_len:] = values

    return padded_values


def pad_id_mask(
    id_mask: Int[torch.Tensor, "*batch variates series_len"],  # noqa: F722
    patch_stride: int,
) -> Int[torch.Tensor, "*batch variates padded_length"]:  # noqa: F722
    """
    Makes sure that the series length is divisible by the patch_stride
    by adding left-padding to the id mask. It does this by repeating
    the leftmost value of the id mask for each variate
    """
    series_len = id_mask.shape[-1]
    # left-pad the time series to make sure we can divide it into patches.
    padded_length = int(np.ceil(series_len / patch_stride) * patch_stride)
    padding_amount = padded_length - series_len
    left_edge: Int[torch.Tensor, "*batch variates"] = id_mask[..., 0]  # noqa: F722
    if id_mask.ndim == 2:  # variates series_len
        # repeat the left edge of the id mask for padding_amount
        padding = repeat(
            left_edge,
            "variates -> variates padding_amount",
            padding_amount=padding_amount,
        )
        id_mask = torch.cat([padding, id_mask], dim=1)
    elif id_mask.ndim == 3:  # batch variates series_len
        # repeat the left edge of the id mask for padding_amount
        padding = repeat(
            left_edge,
            "batch variates -> batch variates padding_amount",
            padding_amount=padding_amount,
        )
        id_mask = torch.cat([padding, id_mask], dim=2)
    else:
        raise ValueError(f"Unsupported number of dimensions: {id_mask.ndim}")

    return id_mask


class MaskedTimeseries(NamedTuple):
    # Note: "*batch" indicates the batch dimension is optional.
    series: Float[torch.Tensor, "*batch variates series_len"]  # noqa: F722
    """
    The time series data.
    """

    padding_mask: Bool[torch.Tensor, "*batch variates series_len"]  # noqa: F722
    """
    A mask that indicates which values are padding. If padding_mask[..., i] is True,
    then series[..., i] is _NOT_ padding; i.e., it's a valid value in the time series.
    """

    id_mask: Int[torch.Tensor, "*batch variates #series_len"]  # noqa: F722
    """
    A mask that indicates the group ID of each variate. Any
    variates with the same ID are considered to be part of the same multivariate
    time series, and can attend to each other.

    Note: the #series_len dimension can be 1 if the IDs should
    be broadcast across the time dimension.
    """

    timestamp_seconds: Int[torch.Tensor, "*batch variates series_len"]  # noqa: F722
    """
    A POSIX timestamp in seconds for each time step in the series.
    """

    time_interval_seconds: Int[torch.Tensor, "*batch variates"]  # noqa: F722
    """
    The time frequency of each variate in seconds
    """


def is_extreme_value(t: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(t):
        max_value = torch.finfo(t.dtype).max
    else:
        max_value = torch.iinfo(t.dtype).max

    return reduce(
        torch.logical_or,
        (
            torch.isinf(t),
            torch.isnan(t),
            t.abs() >= max_value / 2,
        ),
    )


def replace_extreme_values(t: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    return torch.where(is_extreme_value(t), torch.tensor(replacement, dtype=t.dtype, device=t.device), t)
