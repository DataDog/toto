# Toto - Time Series Optimized Transformer for Observability

**Toto 2.0**: [Model Weights](https://huggingface.co/collections/Datadog/toto-open-20) | Blog (coming soon)  
**Toto 1.0**: [Paper](https://arxiv.org/abs/2505.14766) | [Blog](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/) | [Model Card](https://huggingface.co/Datadog/Toto-Open-Base-1.0)

Toto is a foundation model for multivariate time series forecasting with a focus on observability metrics. **Toto 2.0** is the current recommended release, featuring a family of u-μP-scaled models ranging from 4M to 2.5B parameters.

This repository also hosts the code for evaluating time series models on BOOM (**B**enchmark **o**f **O**bservability **M**etrics), a large-scale forecasting dataset composed of real-world observability data.

### Updates

- **[Apr 2026]** Toto 2.0 released — five model sizes from 4M to 2.5B parameters.
- **[Feb 2026]** Fine-tuning support added to Toto 1.0 (training script, configs, and tutorial notebook).
- **[Feb 2026]** Exogenous covariate support added to Toto 1.0 for fine-tuning and inference.


## Table of Contents
- [Updates](#updates)
- [Toto 2.0](#toto-20)
  - [Features](#features)
  - [Model Weights](#model-weights)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Tutorials](#tutorials)
  - [Evaluation](#evaluation)
  - [Requirements](#requirements)
- [dd-unit-scaling](#dd-unit-scaling)
- [Toto 1.0 (Legacy)](#toto-10-legacy)
  - [Features](#features-1)
  - [Pre-Training Data](#pre-training-data)
  - [Evaluation](#evaluation-1)
    - [LSF Evaluation](#lsf-evaluation)
    - [GIFT-Eval Evaluation](#gift-eval-evaluation)
    - [BOOM Evaluation](#boom-evaluation)
  - [Fine-tuning](#fine-tuning)
    - [Custom Datasets](#custom-datasets)
    - [Evaluations on FEV Datasets](#evaluations-on-fev-datasets)
  - [Requirements](#requirements-1)
  - [Citation (1.0)](#citation-10)
- [BOOM (Benchmark of Observability Metrics)](#boom-benchmark-of-observability-metrics)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## Toto 2.0

Toto 2.0 is the latest generation, featuring a u-μP-scaled transformer with alternating time/variate attention and quantile-based probabilistic forecasting.

> **Note:** Fine-tuning and exogenous variable (EV) support are planned for a future 2.0 release but not yet available. If you need these features today, use [Toto 1.0](#toto-10-legacy).

### Features

- **Zero-Shot Forecasting**: Perform forecasting without fine-tuning on your specific time series.
- **State-of-the-Art Performance**: Achieves top scores on diverse benchmarks, including the multi-domain [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) benchmark and our observability-focused [BOOM](https://huggingface.co/datasets/Datadog/BOOM) benchmark.
- **Multi-Variate Support**: Efficiently process multiple variables using alternating time/variate attention.
- **Probabilistic Predictions**: Generate both point forecasts and uncertainty estimates via a quantile head.
- **High-Dimensional Support**: Handle time series with a large number of variables efficiently.
- **Decoder-Only Architecture**: Supports variable prediction horizons and context lengths.

> **Inference tips for `forecast()`:**
> - **`decode_block_size`** selects the decoding strategy:
>   - `None` (single forward pass): faster, better for short-term accuracy. Used for all leaderboard results.
>   - e.g. `768` (block decode): better long-term stability for horizons ≳1000. Default in the quick start and notebooks.
> - **`has_missing_values=False`** (when your context has no gaps) enables Flash Attention kernels for a meaningful speedup. Leave as `True` (default) if `target_mask` contains any `False` entries.

### Model Weights
| Checkpoint | Parameters |
|---|---|
| [Toto-2.0-4m](https://huggingface.co/Datadog/Toto-2.0-4m) | 4M |
| [Toto-2.0-22m](https://huggingface.co/Datadog/Toto-2.0-22m) | 22M |
| [Toto-2.0-313m](https://huggingface.co/Datadog/Toto-2.0-313m) | 313M |
| [Toto-2.0-1B](https://huggingface.co/Datadog/Toto-2.0-1B) | 1B |
| [Toto-2.0-2.5B](https://huggingface.co/Datadog/Toto-2.0-2.5B) | 2.5B |

### Installation

```bash
pip install "toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"
```

### Quick Start

```python
import torch
from toto2 import Toto2Model

model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m")
model = model.to("cuda").eval()

# (batch, n_variates, time_steps)
target = torch.randn(1, 1, 512, device="cuda")
target_mask = torch.ones_like(target, dtype=torch.bool)
series_ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")

# Returns quantiles of shape (9, batch, n_variates, horizon)
# Quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
quantiles = model.forecast(
    {"target": target, "target_mask": target_mask, "series_ids": series_ids},
    horizon=96,
    decode_block_size=768,
    has_missing_values=False,
)
```

### Tutorials

- [Quick Start](toto2/notebooks/quick_start.ipynb): Load a model, forecast, plot results, handle missing values and multivariate inputs.
- [GluonTS Integration](toto2/notebooks/gluonts_integration.ipynb): Use `Toto2GluonTSModel` with GluonTS evaluation pipelines and built-in datasets.

### Evaluation

- [GIFT-Eval Notebook](https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/toto_2_0.ipynb): Evaluate Toto 2.0 on the GIFT-Eval benchmark.
- [BOOM Evaluation Notebook](boom/notebooks/toto.ipynb) and [BOOM README](boom/README.md): Evaluate Toto 2.0 on the BOOM benchmark.

### Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-capable device (Ampere generation or newer recommended for optimal performance)

## dd-unit-scaling

This repository also includes [`dd-unit-scaling`](dd_unit_scaling/), a compile-friendly, world-size-aware extension of [graphcore-research/unit-scaling](https://github.com/graphcore-research/unit-scaling). It is used internally by Toto 2.0 to make u-μP work correctly with `torch.compile` and FSDP2. See the [dd-unit-scaling README](dd_unit_scaling/README.md) for details.

---

<details>
<summary><h2>Toto 1.0 (Legacy)</h2></summary>

> Toto 1.0 is the previous generation of Toto. It is still the right choice if you need **fine-tuning** or **exogenous variable** support, which are planned for 2.0 but not yet available.

[Toto 1.0 Model Card](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | [BOOM Dataset Card](https://huggingface.co/datasets/Datadog/BOOM)


### Features

- **Zero-Shot Forecasting**: Perform forecasting without fine-tuning on your specific time series
- **State-of-the-Art Performance**: Achieves top scores in benchmarks covering diverse time series forecasting tasks. This includes the established multi-domain benchmark [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), as well as our own observability-focused benchmark
[BOOM](https://huggingface.co/datasets/Datadog/BOOM).
- **Multi-Variate Support**: Efficiently process multiple variables using Proportional Factorized Space-Time Attention
- **Probabilistic Predictions**: Generate both point forecasts and uncertainty estimates using a Student-T mixture model
- **High-Dimensional Support**: Handle time series with a large number of variables efficiently
- **Decoder-Only Architecture**: Support for variable prediction horizons and context lengths
- **Pre-trained on Massive Data**: Trained on over 2 trillion time series data points, the largest pretraining dataset for any open-weights time series foundation model to date.

### Model Weights

| Checkpoint | Parameters | Notes |
|------------|------------|-------|
| [Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | 151M | The initial open release of Toto. Achieves state-of-the-art performance on both general-purpose and observability-focused benchmarking tasks, as described in our paper. |

### Installation

```bash
# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install via pip
pip install toto-ts
```

Or install as a local editable package (recommended for development or fine-tuning):

```bash
cd Toto
pip install -r requirements.txt
pip install -e .
```

For optimal inference speed, it's recommended to install [xformers](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) and [flash-attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) as well.

### Quick Start

```python
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

# Load the pre-trained model
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
toto.to('cuda')  # Move to GPU

# Optionally compile the model for faster inference
toto.compile()  # Uses Torch's JIT compilation for better performance

forecaster = TotoForecaster(toto.model)

# Prepare your input time series (channels, time_steps)
input_series = torch.randn(7, 4096).to('cuda')  # Example with 7 variables and 4096 timesteps

# Prepare timestamp information (optional, but expected by API; not used by the current model release)
timestamp_seconds = torch.zeros(7, 4096).to('cuda')
time_interval_seconds = torch.full((7,), 60*15).to('cuda')  # 15-minute intervals

# Create a MaskedTimeseries object
inputs = MaskedTimeseries(
    series=input_series,
    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
    id_mask=torch.zeros_like(input_series),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)

# Generate forecasts for the next 336 timesteps
forecast = forecaster.forecast(
    inputs,
    prediction_length=336,
    num_samples=256,  # Number of samples for probabilistic forecasting
    samples_per_batch=256,  # Control memory usage during inference
)

# Access results
median_prediction = forecast.median  # Point forecasts
prediction_samples = forecast.samples  # Probabilistic samples
lower_quantile = forecast.quantile(0.1)  # 10th percentile for lower confidence bound
upper_quantile = forecast.quantile(0.9)  # 90th percentile for upper confidence bound
```

### Tutorials

- [Basic Inference Tutorial](toto/notebooks/inference_tutorial.ipynb): Learn how to load the model and make forecasts
- [Fine-tuning Tutorial](toto/notebooks/finetuning_tutorial.ipynb): Learn how to fine-tune Toto on custom datasets with or without exogenous covariates

### Pre-Training Data

Toto was trained on a massive and diverse mixture of time series datasets:

#### Observability Data

The largest portion of pretraining data comes from a dataset of approximately 1 trillion time series points collected from Datadog metrics. These metrics are generated from Datadog's monitoring of internal systems, and **do not** include any customer data. They cover a diverse array of software stacks and types of services, and span wide variety of domains within observability, including application performance, infrastructure, networking, security, databases, and more.

#### Public Datasets

To improve the performance of Toto on general-purpose time series forecasting across many domains, we include publicly available datasets:
- [GiftEval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
- [Chronos pretraining data](https://huggingface.co/datasets/autogluon/chronos_datasets) (Note: only a subset of this dataset was used to avoid leakage with the GiftEval benchmark)

#### Synthetic Data
To improve robustness, approximately 1/3 of the pretraining data mix consists of synthetically-generated time series.

### Evaluation

Toto has been rigorously evaluated on multiple benchmarks, including both general-purpose datasets and observability-focused datasets like BOOM. Below, we provide instructions for reproducing our evaluation results.

#### LSF Evaluation

To reproduce our results on the LSF datasets, follow these steps:

##### Downloading the Datasets

The LSF evaluation requires three datasets: ETT, Electricity, and Weather. You can download them from the [Time-Series-Library repository](https://github.com/thuml/Time-Series-Library). Follow the instructions in the [repository](https://github.com/thuml/Time-Series-Library#:~:text=r%20requirements.txt-,Prepare,-Data.%20You%20can) to obtain the following already pre-processed datasets:

- **[ETT (Electricity Transformer Temperature)](https://drive.google.com/file/d/1bnrv7gpn27yO54WJI-vuXP5NclE5BlBx/view?usp=drive_link)**: Includes four subsets: ETTh1, ETTh2, ETTm1, and ETTm2.
- **[Electricity](https://drive.google.com/file/d/1FHH0S3d6IK_UOpg6taBRavx4MragRLo1/view?usp=drive_link)**
- **[Weather](https://drive.google.com/file/d/1nXdMIJ7K201Bx3IBGNiaNFQ6FzeDEzIr/view?usp=drive_link)**

After downloading, ensure the datasets are placed in the `data/lsf_datasets/` directory within the repository, with the following structure:

```
data/
└── lsf_datasets/
  ├── ETT-small/
  ├── electricity/
  └── weather/
```

##### Running the Evaluation Script

Once the datasets are set up, you can run the LSF evaluation script as follows to reproduce our results:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # For reproducible GPU results
export PYTHONPATH="$(pwd):$(pwd)/toto:$PYTHONPATH"  # Add current and "toto" dirs to Python module search path
python toto/evaluation/run_lsf_eval.py \
    --datasets ETTh1 \
    --context-length 2048 \
    --eval-stride 1 \
    --checkpoint-path [CHECKPOINT-NAME-OR-DIR]
```

To see all available options for the evaluation script, you can use the `--help` flag:

```bash
python toto/evaluation/run_lsf_eval.py --help
```

##### Expected Results
The script evaluates Toto's performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE) across the specified datasets, context lengths, and prediction lengths. It displays a detailed table of results for each prediction length, along with a summary table that averages the results across prediction lengths for each dataset.

To reproduce the results presented in the paper, use the default arguments while setting `--eval-stride 1` and specifying all datasets with `--datasets ETTh1 ETTh2 ETTm1 ETTm2 weather electricity`.

#### GIFT-Eval Evaluation

To reproduce our results on the GIFT-Eval benchmark, we provide a dedicated notebook:

- [GIFT-Eval Evaluation Notebook](toto/evaluation/gift_eval/toto.ipynb): Step-by-step instructions for running Toto on the GIFT-Eval benchmark and reproducing the reported results.

#### BOOM Evaluation

For evaluating Toto on the BOOM (Benchmark of Observability Metrics) dataset, refer to:

- [BOOM Evaluation Notebook](boom/notebooks/toto.ipynb): Example workflow for running Toto on the BOOM dataset.
- [BOOM README](boom/README.md): Detailed instructions and scripts for benchmarking on BOOM.

### Fine-tuning

Toto can be fine-tuned on your own domain-specific datasets to improve performance on specialized forecasting tasks. The fine-tuning pipeline supports both standard time series and datasets with exogenous (known future) variables.

#### Fine-tuning Tutorial

To fine-tune Toto, use the provided [finetuning tutorial](toto/notebooks/finetuning_tutorial.ipynb), which demonstrates fine-tuning with and without exogenous variables.

To customize the fine-tuning recipe, modify the base configuration in [finetune_config.yaml](toto/scripts/configs/finetune_config.yaml).

By default, the tutorial uses the `proenfo_gfc12` dataset from the [autogluon/fev_datasets](https://huggingface.co/datasets/autogluon/fev_datasets) collection.

#### Custom Datasets

There are two ways to use custom datasets for fine-tuning:

##### Option A: HuggingFace Dataset with Configuration Dictionary

The simplest approach is to use a HuggingFace `datasets.Dataset` configured via a dictionary. Modify the `prepare_dataset()` function in [benchmark_finetuning.py](toto/scripts/benchmark_finetuning.py) to load your data:

```python
custom_dataset = {
    "dataset": dataset,              # HuggingFace Dataset object
    "target_fields": ["target"],     # List of field names for target variables
    "target_transform_fns": [...],   # Transform functions for each target field
    "ev_fields": ["temp", "humidity"],  # List of exogenous covariate field names
    "ev_transform_fns": [...],       # Transform functions for each exogenous field
    "dataset_name": "my_dataset",    # Name of your custom dataset
}
```

**HuggingFace Dataset Requirements:**

Your dataset must contain:
- `timestamp`: A 1D array of timestamps for each time series
- Target fields (e.g., `target`): Arrays of shape `(T,)` for each target variable
- Exogenous fields (optional): Arrays of shape `(T,)` for each dynamic exogenous variable

The pipeline uses [FinetuneDataModule](toto/data/datamodule/finetune_datamodule.py), which internally converts your data into `CausalMaskedTimeseries` objects (the input format expected by Toto during fine-tuning) via GluonTS transforms.

##### Option B: Custom PyTorch Dataset and DataModule

For full control over data loading, you can implement your own PyTorch `Dataset` that returns `CausalMaskedTimeseries` objects and wrap it in a custom `LightningDataModule`.

**Step 1:** Create a Dataset class that returns `CausalMaskedTimeseries`:

```python
from torch.utils.data import Dataset
from toto.data.util.dataset import CausalMaskedTimeseries

class MyCustomDataset(Dataset):
    ...
    def __getitem__(self, idx: int) -> CausalMaskedTimeseries:
        # Build and return a CausalMaskedTimeseries for this sample
        # See toto/data/datasets/gluonts_dataset.py for a reference implementation
        ...
```

**Step 2:** Create a custom `LightningDataModule`:

```python
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from toto.data.util.helpers import collate_causal

class MyFinetuneDataModule(LightningDataModule):
    def __init__(self, train_dataset: MyCustomDataset, val_dataset: MyCustomDataset, ...):
        ...

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, collate_fn=collate_causal, ...)  # collate_fn is required

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, collate_fn=collate_causal, ...)
```

**Step 3:** Modify [finetune_toto.py](toto/scripts/finetune_toto.py) to use your custom DataModule:

```python
# Replace the get_datamodule() call with your custom DataModule
dm = MyFinetuneDataModule(train_dataset, val_dataset, batch_size=16)
_ = train(module, dm, config)
```

#### Evaluations on FEV Datasets

The [benchmark_finetuning.py](toto/scripts/benchmark_finetuning.py) script evaluates Toto on a subset of FEV datasets that are not included in Toto's pretraining corpus. These datasets contain known exogenous variables, enabling a comparison of three approaches:

- **Zero-shot Toto** — No fine-tuning
- **Fine-tuned Toto** — Fine-tuned without exogenous variables
- **Fine-tuned Toto with Exogenous Variables** — Fine-tuned with known future covariates

Models are evaluated using sliding windows on the test set (10% of each dataset), with context length and horizon configured per FEV task. Results are aggregated using the geometric mean across datasets in [aggregate_results.ipynb](toto/evaluation/fev/aggregate_results.ipynb):

| Model | MAE | WQL | MASE |
|:------|----------:|-------------------:|-----:|
| Toto (zero-shot) | 6150.242 | 0.111 | 0.632 |
| Toto (fine-tuned) | <u>5397.929</u> | <u>0.100</u> | <u>0.574</u> |
| Toto (fine-tuned + exogenous) | **5117.002** | **0.096** | **0.535** |

### Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-capable device (Ampere generation or newer recommended for optimal performance)

### Citation (1.0)

If you use Toto 1.0 in your research, please cite:

```bibtex
@misc{cohen2025timedifferentobservabilityperspective,
      title={This Time is Different: An Observability Perspective on Time Series Foundation Models},
      author={Ben Cohen and Emaad Khwaja and Youssef Doubli and Salahidine Lemaachi and Chris Lettieri and Charles Masson and Hugo Miccinilli and Elise Ramé and Qiqi Ren and Afshin Rostamizadeh and Jean Ogier du Terrail and Anna-Monica Toon and Kan Wang and Stephan Xie and Zongzhe Xu and Viktoriya Zhukova and David Asker and Ameet Talwalkar and Othmane Abou-Amal},
      year={2025},
      eprint={2505.14766},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.14766},
}
```

</details>

## BOOM (Benchmark of Observability Metrics)

BOOM is used for evaluating both Toto 1.0 and 2.0.

**BOOM** (**B**enchmark **o**f **O**bservability **M**etrics) is a large-scale, real-world time series dataset designed for evaluating models on forecasting tasks in complex observability environments.
Composed of real-world metrics data collected from Datadog, a leading observability platform, the benchmark captures the irregularity, structural complexity, and heavy-tailed statistics typical of production observability data. Unlike synthetic or curated benchmarks, BOOM reflects the full diversity and unpredictability of operational signals observed in distributed systems, covering infrastructure, networking, databases, security, and application-level metrics.

Note: the metrics comprising BOOM were generated from internal monitoring of pre-production environments, and **do not** include any customer data.

For more information on the dataset, including details on its preparation and statistical properties, see the [dataset card](https://huggingface.co/datasets/Datadog/BOOM) in Hugging Face.

For example evaluations of different time series models on the BOOM dataset, see the [boom](boom) folder in this repository.

## Citation

If you use Toto 2.0 in your research or work, please cite:

```bibtex
(citation coming soon)
```

For Toto 1.0, see the [Toto 1.0 citation](#citation-10).

## License
Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License - see [LICENSE](LICENSE) file for details.

This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2025-2026 Datadog, Inc.

## Contributing

We welcome contributions! Please check out our [contributing guidelines](CONTRIBUTING.md) to get started.
