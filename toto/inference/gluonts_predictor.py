# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataLoader, InferenceDataLoader
from gluonts.itertools import select
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.forecast_generator import OutputTransform, SampleForecastGenerator
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
)
from tqdm import tqdm

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto


@dataclass(frozen=True)
class Multivariate:
    code: ClassVar[str] = "M"
    batch_size: int = 16


class TotoSampleForecastGenerator(SampleForecastGenerator):
    """
    Wrapper class for generating GluonTS forecasts from Toto models.

    This class differs from the behavior of standard GluonTS generators in one major way:

    Most GluonTS models are encoder-style models with a fixed prediction length, whereas Toto
    is a decoder-only model that can generate forecasts of arbitrary length using
    autoregressive sampling. This wrapper handles the iterative sampling process to generate
    forecasts of the required length, which can be set to any value when the generator is called.

    In addition, this class can also provide both point predictions (when num_samples is None) or
    samples from the predictive distribution (when num_samples is not None). Note that when
    num_samples is not None, the memory usage is significantly higher, so the batch_size and num_variates should be
    set accordingly.
    """

    def freq_to_seconds(self, freq):
        try:
            # Use nanos for fixed frequencies
            return freq.nanos / 1e9  # Convert nanoseconds to seconds
        except ValueError:
            # Handle non-fixed frequencies like Week
            if isinstance(freq, pd.offsets.Week):
                return freq.n * 7 * 24 * 60 * 60  # n weeks to seconds
            elif isinstance(freq, pd.offsets.MonthBegin) or isinstance(freq, pd.offsets.MonthEnd):
                return 30 * 24 * 60 * 60  # Approximate a month as 30 days
            elif isinstance(freq, pd.offsets.QuarterEnd) or isinstance(freq, pd.offsets.QuarterBegin):
                return 90 * 24 * 60 * 60  # Approximate a quarter as 90 days
            elif isinstance(freq, pd.offsets.YearEnd) or isinstance(freq, pd.offsets.YearBegin):
                return 365.25 * 24 * 60 * 60  # Approximate a year as 365.25 days
            else:
                raise ValueError(f"Cannot handle frequency of type {type(freq)}: {freq}")

    def _generate(
        self,
        inference_data_loader: DataLoader,
        forecaster: TotoForecaster,
        prediction_length: int,
        num_samples: Optional[int],
        output_transform: Optional[OutputTransform],
        samples_per_batch: int = 10,
        use_kv_cache: bool = False,
        device: torch.device | str = "auto",
    ) -> Iterator[SampleForecast]:
        INT_MAX = np.iinfo(np.int32).max
        INT_MIN = np.iinfo(np.int32).min

        for batch in tqdm(inference_data_loader):
            if len(batch["past_target"].shape) <= 2:
                inputs = (
                    batch["past_target"]
                    .unsqueeze(-1)
                    .transpose(-1, -2)
                    .to(device=device, dtype=torch.get_default_dtype())
                )
            else:
                inputs = batch["past_target"].transpose(-1, -2).to(device=device, dtype=torch.get_default_dtype())

            padding_mask = repeat(
                (~batch["past_is_pad"].bool()).to(inputs.device),
                "batch time_steps -> batch num_variates time_steps",
                num_variates=inputs.shape[1],
            )

            # First, build a single NumPy array from your list comprehension
            np_timestamps = np.array(
                [
                    (
                        pd.period_range(
                            start=forecast_start - forecast_start.freq,
                            periods=inputs.shape[-1],
                            freq=forecast_start.freq,
                        )
                        .to_timestamp(freq=forecast_start.freq)
                        .view(np.int64)
                        // int(1e9)  # Convert to Unix timestamp in seconds
                    )
                    .clip(INT_MIN, INT_MAX)
                    .astype(int)  # Clamp to torch.int range
                    for forecast_start in batch["forecast_start"]
                ]
            )

            # Then, create a tensor from the contiguous NumPy array
            timestamps = repeat(
                torch.tensor(np_timestamps, dtype=torch.int, device=inputs.device),
                "batch time_steps -> batch num_variates time_steps",
                num_variates=inputs.shape[1],
            )

            time_intervals = repeat(
                torch.tensor(
                    [self.freq_to_seconds(forecast_start.freq) for forecast_start in batch["forecast_start"]],
                    dtype=torch.int,
                    device=inputs.device,
                ),
                "batch -> batch num_variates",
                num_variates=inputs.shape[1],
            )

            id_mask = torch.zeros_like(inputs, dtype=torch.int, device=inputs.device)

            outputs = forecaster.forecast(
                # The GluonTS-based benchmarks don't pack different datasets together for inference
                # the way we do in our training pipeline, so the id_mask is set to None.
                # If we want to support this in the future, we'll need to make further modifications.
                MaskedTimeseries(
                    series=inputs,
                    padding_mask=padding_mask,
                    id_mask=id_mask,
                    timestamp_seconds=timestamps,
                    time_interval_seconds=time_intervals,
                ),
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
                use_kv_cache=use_kv_cache,
            )

            if num_samples is not None:
                samples = rearrange(
                    outputs.samples,
                    "batch variate time_steps samples -> batch samples time_steps variate",
                )
            else:
                samples = rearrange(
                    outputs.mean,
                    "batch variate time_steps -> batch 1 time_steps variate",
                )

            avg = outputs.mean

            avg = rearrange(avg, "batch variate time_steps -> batch time_steps variate")

            samples = samples.squeeze(-1) if samples.shape[-1] == 1 else samples
            avg = avg.squeeze(-1) if avg.shape[-1] == 1 else avg

            avg = avg.to(batch["past_target"].dtype).cpu().numpy()
            samples = samples.to(batch["past_target"].dtype).cpu().numpy()

            if output_transform is not None:
                samples = output_transform(samples)
                mean: np.ndarray = output_transform(mean)

            for item_idx in range(samples.shape[0]):
                yield TotoSampleForecast(
                    samples=samples[item_idx],
                    mean=avg[item_idx],
                    start_date=batch["forecast_start"][item_idx],
                    item_id=batch["item_id"][item_idx],
                    info=None,
                )

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net: Toto,
        prediction_length: int,
        num_samples: Optional[int],
        input_names: List[str],
        output_transform: Optional[OutputTransform],
        mode: Multivariate = Multivariate(batch_size=4),
        samples_per_batch: int = 10,
        use_kv_cache: bool = False,
    ) -> Iterator[SampleForecast]:
        forecaster = TotoForecaster(prediction_net.model)
        yield from self._generate(
            inference_data_loader,
            forecaster,
            prediction_length,
            num_samples,
            output_transform,
            samples_per_batch=samples_per_batch,
            use_kv_cache=use_kv_cache,
            device=next(prediction_net.model.parameters()).device,
        )

    def _construct_batches(self, all_inputs, all_padding_masks, batch_size):
        inputs = torch.cat(all_inputs, dim=0)
        padding_masks = torch.cat(all_padding_masks, dim=0)
        batched_inputs = torch.split(inputs, batch_size, dim=0)
        batched_padding_masks = torch.split(padding_masks, batch_size, dim=0)
        return batched_inputs, batched_padding_masks

    def _stack_variates(self, inference_data_loader, input_names, num_variates):
        all_inputs = []
        all_padding_masks = []
        all_start_dates = []
        all_item_ids = []

        for variates in inference_data_loader:
            # In GluonTS, each group of variates is treated as a separate batch;
            # effectively, what we call the "batch" dimension is always 1.
            # in order to do efficient batched inference with Toto,
            # we need to stack each group of variates into larger batches.
            inputs = select(input_names, variates, ignore_missing=True)
            start_dates = variates[FieldName.FORECAST_START]
            item_ids = variates.get(FieldName.ITEM_ID, [None] * len(start_dates))

            inputs = torch.stack(list(inputs.values()), dim=0)
            padding_mask = torch.zeros(
                1,
                num_variates,
                inputs.shape[2],
                dtype=torch.bool,
                device=inputs.device,
            )
            padding_mask[: inputs.shape[0], :] = True
            all_padding_masks.append(padding_mask)

            if inputs.shape[1] < num_variates:
                # The last group may have fewer variates than the others, so we need to pad it.
                padding_amount = num_variates - inputs.shape[1]
                inputs = F.pad(inputs, (0, 0, 0, padding_amount))
                start_dates = start_dates + [None] * padding_amount
                item_ids = item_ids + [None] * padding_amount

            all_inputs.append(inputs)
            all_start_dates.append(start_dates)
            all_item_ids.append(item_ids)
        return all_inputs, all_padding_masks, all_start_dates, all_item_ids


class TotoSampleForecast(SampleForecast):
    """
    Wrapper around GluonTS's `SampleForecast` class that adds a deterministic
    mean forecast to the samples. By default, `SampleForecast` calculates
    the mean forecast by taking the mean of the samples, but since
    Toto predicts parametrically, we can calculate the mean forecast
    more efficiently by directly computing the mean from the model's output.
    This is useful for evaluation metrics that require a point prediction.
    """

    def __init__(
        self,
        samples: np.ndarray,
        mean: np.ndarray,
        start_date: pd.Period,
        item_id: Optional[str] = None,
        info: Optional[dict] = None,
    ):
        super().__init__(samples, start_date, item_id, info)
        self._mean = mean


class TotoPredictor(PyTorchPredictor):
    """
    Predictor class for Toto models in GluonTS. This class is a thin wrapper
    that adapts Toto to the GluonTS interface for evaluation and forecasting.
    Most of the actual work is done by the `TotoSampleForecastGenerator` class.
    """

    def __init__(
        self,
        input_names: List[str],
        prediction_net: torch.nn.Module,
        prediction_length: int,
        input_transform: Transformation,
        forecast_generator: TotoSampleForecastGenerator,
        output_transform: Optional[OutputTransform] = None,
        lead_time: int = 0,
        device: str | torch.device = "auto",
        mode: Multivariate = Multivariate(batch_size=16),
        samples_per_batch: int = 10,
    ):

        super().__init__(
            batch_size=mode.batch_size,
            input_names=input_names,
            prediction_net=prediction_net,
            prediction_length=prediction_length,
            input_transform=input_transform,
            forecast_generator=forecast_generator,
            output_transform=output_transform,
            lead_time=lead_time,
            device=device,
        )
        self.mode = mode
        self.samples_per_batch = samples_per_batch

    @classmethod
    def create_for_eval(
        cls,
        model: Toto,
        prediction_length: int,
        context_length: int,
        mode: Multivariate = Multivariate(batch_size=1),
        samples_per_batch: int = 10,
    ) -> "TotoPredictor":
        input_transform = AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ) + InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        return cls(
            input_names=["past_target"],
            prediction_net=model,
            prediction_length=prediction_length,
            input_transform=input_transform,
            forecast_generator=TotoSampleForecastGenerator(),
            output_transform=None,
            device=model.device,
            mode=mode,
            samples_per_batch=samples_per_batch,
        )

    def custom_stack_fn(
        self, data: List[Dict[str, Any]], device: torch.types.Device = None
    ) -> Dict[str, Union[torch.Tensor, List[Any]]]:
        """
        Custom stack function for the InferenceDataLoader.
        Attempts to use GluonTS's batchify for stacking. Falls back to manual logic with consistent padding.
        """
        actual_device = device or self.device  # Ensure we have a device to use
        batch: Dict[str, Union[torch.Tensor, List[Any]]] = {}

        def calculate_max_shape(values: List[Union[torch.Tensor, np.ndarray]]) -> Tuple[int, ...]:
            """Calculate the maximum shape for a list of tensors or arrays."""
            return tuple(
                max(v.size(dim) if isinstance(v, torch.Tensor) else v.shape[dim] for v in values)
                for dim in range(len(values[0].shape))
            )

        try:
            # Attempt to use GluonTS's batchify for stacking
            return batchify(data, device=actual_device)
        except (ValueError, TypeError) as e:
            if "setting an array element with a sequence" in str(e) or "only supported types are" in str(e):
                # Calculate max shape once for all tensor/array fields
                tensor_keys = [key for key in data[0].keys() if isinstance(data[0][key], (torch.Tensor, np.ndarray))]
                if tensor_keys:
                    max_shape = calculate_max_shape([item[tensor_keys[0]] for item in data])

                for key in data[0].keys():
                    values = [item[key] for item in data]
                    acceptable_types = [np.float64, np.float32, np.float16]
                    if isinstance(values[0], torch.Tensor):
                        # Stack tensors and move to device
                        batch[key] = torch.stack([v.to(actual_device) for v in values])
                    elif isinstance(values[0], np.ndarray) and (values[0].dtype in acceptable_types):
                        # Handle padding for variable-length sequences
                        max_shape = np.array([v.shape for v in values]).max(axis=0)
                        padded_values = []
                        for v in values:
                            pad_width = [(0, m - s) for s, m in zip(v.shape, max_shape)]
                            padded_v = np.pad(v, pad_width, mode="constant")
                            padded_values.append(padded_v)
                        batch[key] = torch.tensor(padded_values, device=actual_device)
                    elif isinstance(values[0], (int, float, np.number)):
                        # Handle numerical values
                        batch[key] = torch.tensor(values, device=actual_device)
                    elif isinstance(values[0], str):
                        # Keep as list of strings
                        batch[key] = values  # Strings remain as list
                    else:
                        # Handle other types if necessary
                        batch[key] = values  # Keep other types as is
                return batch
            else:
                raise

    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        use_kv_cache: bool = False,
        eval: bool = True,
    ) -> Iterator[Forecast]:

        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=self.custom_stack_fn,
        )

        # CI has a problem with FusedRMS. Runs prediction in train mode when False.
        if eval:
            self.prediction_net.eval()

        with torch.no_grad():
            yield from self.forecast_generator(
                inference_data_loader=inference_data_loader,
                prediction_net=self.prediction_net,
                input_names=self.input_names,
                output_transform=self.output_transform,
                num_samples=num_samples,
                prediction_length=self.prediction_length,
                mode=self.mode,
                samples_per_batch=self.samples_per_batch,
                use_kv_cache=use_kv_cache,
            )
