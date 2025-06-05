# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from beartype import beartype
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.torch.batchify import batchify
from gluonts.transform import InstanceSplitter, TestSplitSampler

from ..helper_functions import set_default_dtype, skip_if_no_xformers

skip_if_no_xformers()
set_default_dtype()

from toto.inference.forecaster import TotoForecaster
from toto.inference.gluonts_predictor import (
    Multivariate,
    TotoPredictor,
    TotoSampleForecast,
)
from toto.model.toto import Toto

DEVICE = torch.get_default_device()

BASE_MODEL_KWARGS = {
    "patch_size": 8,
    "stride": 8,
    "embed_dim": 64,
    "num_layers": 2,
    "num_heads": 2,
    "mlp_hidden_dim": 128,
    "dropout": 0.1,
    "spacewise_every_n_layers": 2,
    "spacewise_first": True,
    "use_memory_efficient_attention": False,
    "scaler_cls": "<class 'model.scaler.CausalPatchStdMeanScaler'>",
    "output_distribution_classes": ["<class 'model.distribution.MixtureOfStudentTsOutput'>"],
    "output_distribution_kwargs": {
        "k_components": 1,
    },
}


@pytest.fixture
def real_model():
    """Fixture to create a small Toto model instance."""
    return Toto(**BASE_MODEL_KWARGS).to(DEVICE)


@pytest.fixture
def mock_forecaster(mock_model):
    """Mock the TotoForecaster."""

    mock = MagicMock(spec=TotoForecaster)
    mock.model = mock_model.to(DEVICE)
    mock.model.query_embed = None

    def mock_forecast(*args, **kwargs):
        batch_size = 1
        variates = 3
        time_steps = kwargs.get("prediction_length", 5)
        num_samples = kwargs.get("num_samples", 10)

        class MockOutputs:
            def __init__(self, samples, mean):
                self.samples = samples
                self.mean = mean

        # Create samples with shape (batch_size, variate, time_steps, samples)
        samples = torch.rand(batch_size, variates, time_steps, num_samples, device=DEVICE)
        mean = torch.mean(samples, dim=-1)  # Shape: (batch_size, variates, time_steps)

        return MockOutputs(samples=samples, mean=mean)

    mock.forecast.side_effect = mock_forecast

    return mock


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    data = [
        {
            FieldName.TARGET: torch.rand(3, 30).tolist(),  # 3 variates, 30 time steps
            FieldName.START: pd.Period("2021-01-01", freq="D"),
            FieldName.FORECAST_START: pd.Period("2021-01-31", freq="D"),
            FieldName.IS_PAD: torch.tensor([[False] * 30] * 3, dtype=torch.bool).tolist(),
        }
    ]
    return ListDataset(data, freq="D", one_dim_target=False)


@pytest.fixture
def mock_inference_loader(mock_dataset):
    print("Mock Dataset:")
    for entry in mock_dataset:
        print(entry)
    """Create a mock InferenceDataLoader with a fixed past_length and future_length."""
    transform = InstanceSplitter(
        target_field=FieldName.TARGET,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        is_pad_field=FieldName.IS_PAD,
        instance_sampler=TestSplitSampler(),
        past_length=20,  # Set past_length explicitly
        future_length=10,  # Set future_length explicitly
    )
    # Verify and enforce `past_is_pad` is Bool
    return InferenceDataLoader(
        dataset=mock_dataset,
        transform=transform,
        batch_size=1,  # Single batch
        stack_fn=lambda data: batchify(data, DEVICE),
    )


@pytest.fixture
def toto_forecaster(real_model):
    """Fixture to create a TotoForecaster instance."""
    from toto.inference.forecaster import TotoForecaster

    return TotoForecaster(model=real_model.model)


@beartype
def test_totosampleforecast():
    """Test TotoSampleForecast initialization."""
    samples = np.random.rand(10, 5, 3)
    mean = np.mean(samples, axis=0)
    start_date = pd.Period("2021-01-01", freq="D")
    forecast = TotoSampleForecast(samples, mean, start_date)

    assert forecast.samples.shape == samples.shape, "Sample shape mismatch"
    assert np.allclose(forecast.mean, mean), "Mean computation mismatch"
    assert forecast.start_date == start_date, "Start date mismatch"


@beartype
def test_custom_stack_fn(real_model, mock_dataset):
    """Test the custom_stack_fn for valid and problematic inputs."""
    # Create a TotoPredictor instance
    mode = Multivariate(batch_size=2)
    predictor = TotoPredictor.create_for_eval(
        model=real_model,
        prediction_length=3,
        context_length=4,
        mode=mode,
    )

    # Prepare mock data to simulate a DataLoader batch
    valid_batch = [
        {
            FieldName.TARGET: torch.rand(3, 30),
            FieldName.START: pd.Period("2021-01-01", freq="D"),
            FieldName.FORECAST_START: pd.Period("2021-01-31", freq="D"),
            FieldName.IS_PAD: torch.tensor([[False] * 30] * 3),
        },
        {
            FieldName.TARGET: torch.rand(3, 30),
            FieldName.START: pd.Period("2021-01-02", freq="D"),
            FieldName.FORECAST_START: pd.Period("2021-02-01", freq="D"),
            FieldName.IS_PAD: torch.tensor([[False] * 30] * 3),
        },
    ]

    # Test with valid input
    valid_output = predictor.custom_stack_fn(valid_batch, device=torch.device("cpu"))

    # Verify that tensor fields are lists of tensors
    tensor_fields = [FieldName.TARGET, FieldName.IS_PAD]
    for field in tensor_fields:
        assert isinstance(valid_output[field], list), f"{field} should be a list"
        assert all(
            isinstance(t, torch.Tensor) for t in valid_output[field]
        ), f"All elements in {field} should be tensors"
        assert all(
            t.shape == torch.Size([3, 30]) for t in valid_output[field]
        ), f"Shapes of tensors in {field} do not match"

    # Verify non-tensor fields remain lists
    non_tensor_fields = [FieldName.START, FieldName.FORECAST_START]
    for field in non_tensor_fields:
        assert isinstance(valid_output[field], list), f"{field} should remain a list"
        assert len(valid_output[field]) == len(valid_batch), f"{field} length mismatch"

    # Ensure the overall structure of the output matches the input structure
    assert len(valid_output[FieldName.TARGET]) == len(valid_batch), "Output batch size mismatch for target"
    assert len(valid_output[FieldName.IS_PAD]) == len(valid_batch), "Output batch size mismatch for is_pad"

    print("Custom stack function passed all tests.")


@beartype
def test_custom_stack_fn_consistency(real_model):
    """Test consistency of custom_stack_fn between batchify and fallback logic."""
    # Create a TotoPredictor instance
    mode = Multivariate(batch_size=2)
    predictor = TotoPredictor.create_for_eval(
        model=real_model,
        prediction_length=3,
        context_length=4,
        mode=mode,
    )

    # Prepare mock data
    # Generate the first item once
    first_item = {
        FieldName.TARGET: torch.rand(3, 30),
        FieldName.START: pd.Period("2021-01-01", freq="D"),
        FieldName.FORECAST_START: pd.Period("2021-01-31", freq="D"),
        FieldName.IS_PAD: torch.tensor([[False] * 30] * 3),
    }

    # Second item for consistent batch
    second_item_consistent = {
        FieldName.TARGET: torch.rand(3, 30),
        FieldName.START: pd.Period("2021-01-02", freq="D"),
        FieldName.FORECAST_START: pd.Period("2021-02-01", freq="D"),
        FieldName.IS_PAD: torch.tensor([[False] * 30] * 3),
    }

    # Second item for problematic batch (shorter sequence)
    second_item_problematic = {
        FieldName.TARGET: torch.rand(3, 25),
        FieldName.START: pd.Period("2021-01-02", freq="D"),
        FieldName.FORECAST_START: pd.Period("2021-02-01", freq="D"),
        FieldName.IS_PAD: torch.tensor([[False] * 25] * 3),
    }

    # Define the batches
    consistent_batch = [first_item, second_item_consistent]
    problematic_batch = [first_item, second_item_problematic]

    # Test with the consistent batch (should pass batchify)
    batchify_output = predictor.custom_stack_fn(consistent_batch, device=torch.device("cpu"))

    # Test with the problematic batch (should fall back to custom logic)
    fallback_output = predictor.custom_stack_fn(problematic_batch, device=torch.device("cpu"))

    # Compare the TARGET fields in both outputs
    assert FieldName.TARGET in batchify_output, "TARGET missing in batchify output"
    assert FieldName.TARGET in fallback_output, "TARGET missing in fallback output"

    # Handle fallback case: pad and stack if necessary
    def get_stacked_target(output):
        if isinstance(output[FieldName.TARGET], list):
            max_len = max(t.size(-1) for t in output[FieldName.TARGET])
            padded_targets = [
                F.pad(t, (0, max_len - t.size(-1))) if t.size(-1) < max_len else t for t in output[FieldName.TARGET]
            ]
            return torch.stack(padded_targets, dim=0)
        else:
            return output[FieldName.TARGET]

    batchify_target = get_stacked_target(batchify_output)
    fallback_target = get_stacked_target(fallback_output)

    # Ensure the shapes match
    assert fallback_target.shape == batchify_target.shape, "Shapes between batchify and fallback outputs do not match"

    # Ensure the content matches for the overlapping data (first item should be identical)
    assert torch.allclose(
        fallback_target[0], batchify_target[0]
    ), "First item mismatch between batchify and fallback outputs"


@beartype
def test_custom_stack_fn_numpy_inputs(real_model):
    """Test custom_stack_fn with numpy inputs for consistency and correctness."""
    # Create a TotoPredictor instance
    mode = Multivariate(batch_size=2)
    predictor = TotoPredictor.create_for_eval(
        model=real_model,
        prediction_length=3,
        context_length=4,
        mode=mode,
    )

    # Set a random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Prepare mock data with NumPy arrays
    # Generate the first item once to ensure identical overlapping data
    first_item = {
        FieldName.TARGET: np.random.rand(3, 30),  # NumPy array
        FieldName.START: pd.Period("2021-01-01", freq="D"),
        FieldName.FORECAST_START: pd.Period("2021-01-31", freq="D"),
        FieldName.IS_PAD: np.array([[False] * 30] * 3),
    }

    # Second item for consistent batch (same sequence length)
    second_item_consistent = {
        FieldName.TARGET: np.random.rand(3, 30),  # NumPy array
        FieldName.START: pd.Period("2021-01-02", freq="D"),
        FieldName.FORECAST_START: pd.Period("2021-02-01", freq="D"),
        FieldName.IS_PAD: np.array([[False] * 30] * 3),
    }

    # Second item for problematic batch (shorter sequence length)
    second_item_problematic = {
        FieldName.TARGET: np.random.rand(3, 25),  # NumPy array with shorter length
        FieldName.START: pd.Period("2021-01-02", freq="D"),
        FieldName.FORECAST_START: pd.Period("2021-02-01", freq="D"),
        FieldName.IS_PAD: np.array([[False] * 25] * 3),
    }

    # Define the batches
    consistent_batch = [first_item, second_item_consistent]
    problematic_batch = [first_item, second_item_problematic]

    # Test with the consistent batch (should pass batchify)
    batchify_output = predictor.custom_stack_fn(consistent_batch, device=torch.device("cpu"))

    # Test with the problematic batch (should fall back to custom logic)
    fallback_output = predictor.custom_stack_fn(problematic_batch, device=torch.device("cpu"))

    # Compare the TARGET fields in both outputs
    assert FieldName.TARGET in batchify_output, "TARGET missing in batchify output"
    assert FieldName.TARGET in fallback_output, "TARGET missing in fallback output"

    # Handle outputs: convert to tensors and stack if necessary
    def process_output(output):
        target = output[FieldName.TARGET]
        if isinstance(target, list):
            # Convert NumPy arrays to tensors if necessary
            target = [torch.tensor(arr) if isinstance(arr, np.ndarray) else arr for arr in target]
            # Pad and stack
            max_len = max(t.size(-1) for t in target)
            padded_targets = [F.pad(t, (0, max_len - t.size(-1))) if t.size(-1) < max_len else t for t in target]
            return torch.stack(padded_targets, dim=0)
        else:
            return target

    batchify_target = process_output(batchify_output)
    fallback_target = process_output(fallback_output)

    # Ensure the shapes match
    assert fallback_target.shape == batchify_target.shape, "Shapes between batchify and fallback outputs do not match"

    # Ensure the content matches for the overlapping data (first item should be identical)
    assert torch.allclose(
        fallback_target[0], batchify_target[0], atol=1e-6
    ), "First item mismatch between batchify and fallback outputs"
