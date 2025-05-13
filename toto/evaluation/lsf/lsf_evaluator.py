# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from itertools import product

import pandas as pd
from gluonts.dataset.split import TestData
from gluonts.ev import metrics as gluon_metrics
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.time_feature import get_seasonality

from toto.evaluation.lsf.lsf_datasets import LSFDatasetName, get_lsf_sub_dataset
from toto.inference.gluonts_predictor import (
    Multivariate,
    TotoPredictor,
    TotoSampleForecast,
)
from toto.model.toto import Toto

POINT_FORECAST_METRICS = [
    gluon_metrics.MAE("0.5"),
    gluon_metrics.MSE("0.5"),
]


class LSFEvaluator:
    """
    Class to evaluate model on Long Sequence Forecasting benchmark datasets,
    maintained at https://github.com/thuml/Time-Series-Library.

    Uses GluonTS's evaluation tools.
    """

    def __init__(
        self,
        metrics: list[gluon_metrics.Metric] = POINT_FORECAST_METRICS,
        datasets: list[LSFDatasetName] = [
            LSFDatasetName.ETTh1,
            LSFDatasetName.ETTh2,
            LSFDatasetName.ETTm1,
            LSFDatasetName.ETTm2,
            LSFDatasetName.weather,
        ],
        prediction_lengths: list[int] = [96, 192, 336, 720],
        context_lengths: list[int] = [1280],
        mode: Multivariate = Multivariate(1),
        num_samples: int | None = 30,
        lsf_path: str = "./data/",
        data_split: str = "test",
        eval_stride: int = 256,
        samples_per_batch: int = 30,
        use_kv_cache: bool = False,
    ):
        self.metrics = metrics
        self.datasets = datasets
        self.prediction_lengths = prediction_lengths
        self.context_lengths = context_lengths
        self.num_samples = num_samples
        self.lsf_path = lsf_path
        self.data_split = data_split
        self.mode: Multivariate = mode
        self.eval_stride = eval_stride
        self.samples_per_batch = samples_per_batch
        self.use_kv_cache = use_kv_cache

    def eval(
        self, model: Toto, checkpoint_name: str | None = None
    ) -> tuple[pd.DataFrame, list[list[TotoSampleForecast]], list[TestData]]:
        all_evaluations = []
        all_forecasts = []
        all_test_data = []

        for dataset_name, prediction_length, context_length in product(
            self.datasets,
            self.prediction_lengths,
            self.context_lengths,
        ):
            checkpoint_name_str = f" checkpoint {checkpoint_name} " if checkpoint_name else " "
            print(
                f"Evaluating{checkpoint_name_str}on {dataset_name} with prediction length {prediction_length} and context length {context_length}"
            )
            test_data, metadata, _ = get_lsf_sub_dataset(
                dataset_name,
                prediction_length=prediction_length,
                lsf_path=self.lsf_path,
                data_split=self.data_split,
                mode=self.mode.code,
                eval_stride=self.eval_stride,
            )

            predictor = TotoPredictor.create_for_eval(
                model,
                prediction_length,
                context_length,
                self.mode,
                self.samples_per_batch,
            )
            forecasts = list(
                predictor.predict(
                    test_data.input,
                    num_samples=self.num_samples,
                    use_kv_cache=self.use_kv_cache,
                )
            )
            evaluation = evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=self.metrics,
                seasonality=get_seasonality(metadata.freq),
            )
            evaluation["dataset"] = dataset_name.value
            evaluation["context_length"] = context_length
            evaluation["prediction_length"] = prediction_length
            if checkpoint_name is not None:
                evaluation["checkpoint"] = checkpoint_name

            all_evaluations.append(evaluation)
            all_forecasts.append(forecasts)
            all_test_data.append(test_data)

        df = pd.concat(all_evaluations)
        if checkpoint_name is not None:
            df = df.set_index(["checkpoint", "dataset", "context_length", "prediction_length"])
        else:
            df = df.set_index(["dataset", "context_length", "prediction_length"])

        return df, all_forecasts, all_test_data
