# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch
from beartype import beartype

from ..helper_functions import set_default_dtype, skip_if_no_xformers

skip_if_no_xformers()
set_default_dtype()


from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import Forecast, TotoForecaster
from toto.model.backbone import PatchEmbedding, TotoBackbone


@pytest.fixture
def mock_model():
    """Create a mock TotoModule with PatchEmbedding and realistic distribution handling."""
    patch_size = 4
    stride = 4
    embed_dim = 4

    # PatchEmbedding instance
    patch_embed = PatchEmbedding(patch_size=patch_size, stride=stride, embed_dim=embed_dim)

    # Mock TotoModule
    mock = MagicMock(spec=TotoBackbone)
    mock.patch_embed = patch_embed

    # Mock allocate_kv_cache
    mock.allocate_kv_cache.return_value = None

    def mock_forward(inputs, **kwargs):
        batch_size, num_variates, seq_len = inputs.shape
        loc = torch.zeros(batch_size, num_variates, seq_len)
        scale = torch.ones(batch_size, num_variates, seq_len)

        # Use torch.distributions.Uniform as the base distribution
        base_distr = torch.distributions.Uniform(
            low=torch.zeros(batch_size, num_variates, seq_len),
            high=torch.ones(batch_size, num_variates, seq_len),
        )
        return base_distr, loc, scale

    mock.side_effect = mock_forward

    return mock


@pytest.fixture
def mock_inputs():
    """Generate a mock MaskedTimeseries input."""
    batch_size = 2
    num_variates = 2
    time_steps = 4

    series = torch.randn(batch_size, num_variates, time_steps)
    padding_mask = torch.ones_like(series, dtype=torch.bool)
    time_interval_seconds = torch.randint(1, 10, (batch_size, num_variates))
    timestamp_seconds = torch.arange(time_steps).unsqueeze(0).unsqueeze(0)
    timestamp_seconds = timestamp_seconds.expand(
        batch_size, num_variates, time_steps
    ) * time_interval_seconds.unsqueeze(-1)
    id_mask = torch.zeros_like(series)

    return MaskedTimeseries(
        series=series,
        padding_mask=padding_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
        id_mask=id_mask,
    )


@beartype
def test_forecast_mean(mock_model, mock_inputs):
    """Test the forecast method when only mean predictions are generated."""
    forecaster = TotoForecaster(model=mock_model)
    prediction_length = 8

    forecast = forecaster.forecast(inputs=mock_inputs, prediction_length=prediction_length)

    assert isinstance(forecast, Forecast), "Forecast output is not of type Forecast."
    assert forecast.mean.shape == (
        mock_inputs.series.shape[0],
        mock_inputs.series.shape[1],
        prediction_length,
    ), "Mean forecast shape mismatch."
    assert forecast.samples is None, "Samples should be None when not requested."


@beartype
def test_forecast_samples(mock_model, mock_inputs):
    """Test the forecast method when samples are generated."""
    forecaster = TotoForecaster(model=mock_model)
    prediction_length = 8
    num_samples = 10

    forecast = forecaster.forecast(inputs=mock_inputs, prediction_length=prediction_length, num_samples=num_samples)

    assert isinstance(forecast, Forecast), "Forecast output is not of type Forecast."
    assert forecast.mean.shape == (
        mock_inputs.series.shape[0],
        mock_inputs.series.shape[1],
        prediction_length,
    ), "Mean forecast shape mismatch."
    assert forecast.samples.shape == (
        mock_inputs.series.shape[0],
        mock_inputs.series.shape[1],
        prediction_length,
        num_samples,
    ), "Samples shape mismatch."


@beartype
def test_generate_mean(mock_model, mock_inputs):
    """Test the generate_mean method directly."""
    forecaster = TotoForecaster(model=mock_model)
    prediction_length = 8

    mean = forecaster.generate_mean(
        inputs=mock_inputs.series,
        prediction_length=prediction_length,
        timestamp_seconds=mock_inputs.timestamp_seconds,
        time_interval_seconds=mock_inputs.time_interval_seconds,
        input_padding_mask=mock_inputs.padding_mask,
        id_mask=mock_inputs.id_mask,
    )

    assert mean.shape == (
        mock_inputs.series.shape[0],
        mock_inputs.series.shape[1],
        prediction_length,
    ), "Mean output shape mismatch."


@beartype
def test_generate_samples(mock_model, mock_inputs):
    """Test the generate_samples method directly."""
    forecaster = TotoForecaster(model=mock_model)
    prediction_length = 8
    num_samples = 10

    samples = forecaster.generate_samples(
        inputs=mock_inputs.series,
        prediction_length=prediction_length,
        num_samples=num_samples,
        timestamp_seconds=mock_inputs.timestamp_seconds,
        time_interval_seconds=mock_inputs.time_interval_seconds,
        input_padding_mask=mock_inputs.padding_mask,
        id_mask=mock_inputs.id_mask,
    )

    assert samples.shape == (
        mock_inputs.series.shape[0],
        mock_inputs.series.shape[1],
        prediction_length,
        num_samples,
    ), "Samples output shape mismatch."


@beartype
def test_forecast_kv_cache(mock_model, mock_inputs):
    """Test the forecast method with kv_cache enabled."""
    forecaster = TotoForecaster(model=mock_model)
    prediction_length = 8
    mock_model.allocate_kv_cache.return_value = MagicMock()  # Simulate kv_cache allocation

    forecast = forecaster.forecast(inputs=mock_inputs, prediction_length=prediction_length, use_kv_cache=True)

    assert forecast.mean.shape == (
        mock_inputs.series.shape[0],
        mock_inputs.series.shape[1],
        prediction_length,
    ), "Mean forecast shape mismatch with kv_cache enabled."
