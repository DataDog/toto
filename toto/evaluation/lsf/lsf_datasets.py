# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2023 Salesforce, Inc.
# Copyright 2025 Datadog, Inc.
#
# (Adapted from https://github.com/SalesforceAIResearch/uni2ts/tree/ce27c2f9a0c6ee9119997e8ef0026388f143dcd6/src/uni2ts/eval_util)
#

import os
from enum import Enum
from typing import NamedTuple

import numpy as np
import pandas as pd
from typing import Literal


from gluonts.dataset.common import _FileDataset
from gluonts.dataset.split import TestData, split


class MetaData(NamedTuple):
    freq: str
    target_dim: int
    prediction_length: int
    feat_dynamic_real_dim: int = 0
    past_feat_dynamic_real_dim: int = 0
    split: str = "test"


class LSFDatasetName(str, Enum):
    ETTh1 = "ETTh1"
    ETTh2 = "ETTh2"
    ETTm1 = "ETTm1"
    ETTm2 = "ETTm2"
    electricity = "electricity"
    weather = "weather"


class LSFDataset:
    """
    LSFDataset is a class for loading and processing time series datasets for evaluation purposes.
    It supports multiple datasets and modes of operation.
    Attributes:
        dataset_name (LSFDatasetName): The name of the dataset to load. Supported values include "ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", and "weather".
        mode (str): The mode of operation. Supported values are:
            - "S": Single target dimension.
            - "M": Multi-target dimensions.
            - "MS": Mixed single and multi-target dimensions.
        split (str): The data split to use. Supported values are "train", "val", and "test".
        lsf_path (str): The base path to the dataset files.
    Methods:
        __iter__():
            Iterates over the dataset and yields data samples based on the mode of operation.
            - For "S" mode, yields individual target dimensions.
            - For "M" mode, yields all target dimensions transposed.
            - For "MS" mode, yields individual target dimensions along with past features.
        scale(data, start, end):
            Scales the data using the mean and standard deviation of the training set.
            Args:
                data (numpy.ndarray): The data to scale.
                start (int): The start index for the training set.
                end (int): The end index for the training set.
            Returns:
                numpy.ndarray: The scaled data.
        _load_etth():
            Loads and processes the ETTh1 or ETTh2 dataset. Scales the data and splits it into train, validation, and test sets.
        _load_ettm():
            Loads and processes the ETTm1 or ETTm2 dataset. Scales the data and splits it into train, validation, and test sets.
        _load_custom(data_path, freq):
            Loads and processes a custom dataset. Scales the data and splits it into train, validation, and test sets.
            Args:
                data_path (str): The relative path to the dataset file.
                freq (str): The frequency of the time series data (e.g., "h", "10T").
    Raises:
        ValueError: If an unknown dataset name or mode is provided.
    """

    def __init__(
        self, dataset_name: LSFDatasetName, mode: str = "S", split: str = "test", lsf_path: str = "/home/jovyan/data/"
    ):
        self.dataset_name = dataset_name
        self.mode = mode
        self.split = split
        self.lsf_path = lsf_path

        if dataset_name in ["ETTh1", "ETTh2"]:
            self._load_etth()
        elif dataset_name in ["ETTm1", "ETTm2"]:
            self._load_ettm()
        elif dataset_name == "electricity":
            self._load_custom("electricity/electricity.csv", "h")
        elif dataset_name == "weather":
            self._load_custom("weather/weather.csv", "10T")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        if mode == "S":  # TODO(Anna) - should we just support M mode?
            self.target_dim = 1
            self.past_feat_dynamic_real_dim = 0
        elif mode == "M":
            self.target_dim = self.data.shape[-1]
            self.past_feat_dynamic_real_dim = 0
        elif mode == "MS":
            self.target_dim = 1
            self.past_feat_dynamic_real_dim = self.data.shape[-1] - 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __iter__(self):
        if self.mode == "S":
            for i in range(self.data.shape[-1]):
                yield {
                    "target": self.data[:, i],
                    "start": self.start,
                }
        elif self.mode == "M":
            yield {
                "target": self.data.transpose(1, 0),
                "start": self.start,
            }
        elif self.mode == "MS":
            for i in range(self.data.shape[-1]):
                yield {
                    "target": self.data[:, i],
                    "past_feat_dynamic_real": np.concatenate(
                        [self.data[:, :i], self.data[:, i + 1 :]], axis=1
                    ).transpose(1, 0),
                    "start": self.start,
                }

    def scale(self, data, start, end):
        train = data[start:end]
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        return (data - mean) / std

    def _load_etth(self):
        df = pd.read_csv(os.path.join(self.lsf_path, f"ETT-small/{self.dataset_name}.csv"))

        train_length = 8640
        val_length = 2880
        test_length = 2880
        data = self.scale(df[df.columns[1:]], 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = "h"

    def _load_ettm(self):
        df = pd.read_csv(os.path.join(self.lsf_path, f"ETT-small/{self.dataset_name}.csv"))

        train_length = 34560
        val_length = 11520
        test_length = 11520
        data = self.scale(df[df.columns[1:]], 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = "15T"

    def _load_custom(self, data_path: str, freq: str):
        df = pd.read_csv(os.path.join(self.lsf_path, data_path))
        cols = list(df.columns)
        cols.remove("OT")
        cols.remove("date")
        df = df[["date"] + cols + ["OT"]]
        data = df[df.columns[1:]]

        train_length = int(len(data) * 0.7)
        val_length = int(len(data) * 0.1)
        test_length = int(len(data) * 0.2)
        data = self.scale(data, 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = freq


# utility functions
def compute_num_windows(dataset_length: int, window_length: int, window_stride: int) -> int:
    """
    Computes the number of windows that can fit inside the dataset length based on the stride and window length.
    """
    if (dataset_length - window_length) < 0:
        return 0
    return (
        dataset_length - window_length
    ) // window_stride + 1  # equal to how many strides we can fit inside the length + 1 for the first window


def get_lsf_sub_dataset(
    dataset_name: LSFDatasetName,
    prediction_length: int = 96,
    data_split: str = "test",
    mode: str | Literal["M", "S", "MS"] = "M",
    eval_stride: int = 32,
    lsf_path: str = "./data/",
) -> tuple[TestData, MetaData, _FileDataset]:
    """
    loads a subset from the LSF Dataset into a gluonTS TestData object
    """
    lsf_dataset = LSFDataset(dataset_name, mode=mode, split=data_split, lsf_path=lsf_path)
    dataset = _FileDataset(lsf_dataset, freq=lsf_dataset.freq, one_dim_target=lsf_dataset.target_dim == 1)
    _, test_template = split(dataset, offset=-lsf_dataset.length)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=compute_num_windows(
            dataset_length=lsf_dataset.length,
            window_length=prediction_length,
            window_stride=eval_stride,
        ),
        distance=eval_stride,
    )
    metadata = MetaData(
        freq=lsf_dataset.freq,
        target_dim=lsf_dataset.target_dim,
        prediction_length=prediction_length,
        past_feat_dynamic_real_dim=lsf_dataset.past_feat_dynamic_real_dim,
        split=data_split,
    )
    return test_data, metadata, dataset
