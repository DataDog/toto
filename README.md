# Toto - Time Series Optimized Transformer for Observability
[Paper](https://arxiv.org/abs/2505.14766) | [Toto Model Card](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | [BOOM Dataset Card](https://huggingface.co/datasets/Datadog/BOOM) | [Blogpost](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/)

Toto is a foundation model for multivariate time series forecasting with a focus on observability metrics. This model leverages innovative architectural designs to efficiently handle the high-dimensional, complex time series that are characteristic of observability data.

This repository also hosts the code for evaluating time series models on BOOM (**B**enchmark **o**f **O**bservability **M**etrics), a large-scale forecasting dataset composed of real-world observability data.

## Table of Contents
- [Toto model](#toto-model)
  - [Features](#features)
  - [Model Weights](#model-weights)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Tutorials](#tutorials)
  - [Training Data](#training-data)
  - [Evaluation](#evaluation)
    - [LSF Evaluation](#lsf-evaluation)
    - [GIFT-Eval Evaluation](#gift-eval-evaluation)
    - [BOOM Evaluation](#boom-evaluation)
  - [Requirements](#requirements)
- [BOOM (Benchmark of Observability Metrics)](#boom-benchmark-of-observability-metrics)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## Toto model

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

Toto-Open, the open-weights release of Toto, is available on Hugging Face. Currently available checkpoints:

| Checkpoint | Parameters | Notes |
|------------|------------|-------|
| [Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | 151M | The initial open relase of Toto. Achieves state-of-the-art performance on both general-purpose and observability-focused benchmarking tasks, as described in our paper. |



### Installation

```bash
# Clone the repository
git clone https://github.com/DataDog/toto.git
cd toto

# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install --editable .
```
For optimal inference speed, it's recommended to install [xformers](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) and [flash-attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) as well.

### Quick Start

Here's a simple example to get you started with forecasting:

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
mean_prediction = forecast.mean  # Point forecasts
prediction_samples = forecast.samples  # Probabilistic samples
lower_quantile = forecast.quantile(0.1)  # 10th percentile for lower confidence bound
upper_quantile = forecast.quantile(0.9)  # 90th percentile for upper confidence bound
```

### Tutorials

For a comprehensive guide on using Toto for time series forecasting, check out our tutorial notebooks:

- [Basic Inference Tutorial](toto/notebooks/inference_tutorial.ipynb): Learn how to load the model and make forecasts

### Pre-Training Data

Toto was trained on a massive and diverse mixture of time series datasets:

#### Observability Data

The largest portion of pretraining data comes from a dataset of approximately 1 trillion time series points collected from Datadog metrics. These metrics are generated from Datadog's monitoring of internal systems, and **do not** include any customer data. They cover a diverse array of software stacks and types of services, and span wide variety of domains within observability, including application performance, infrastructure, networking, security, databases, and more.

#### Public Datasets

To improve the performance of Toto on general-purpose time series forecasting across many domains, we include publcly availa
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

These resources provide all necessary steps to run and reproduce BOOM evaluation results with Toto.

### Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-capable device (Ampere generation or newer recommended for optimal performance)

## BOOM (Benchmark of Observability Metrics)

**BOOM** (**B**enchmark **o**f **O**bservability **M**etrics) is a large-scale, real-world time series dataset designed for evaluating models on forecasting tasks in complex observability environments.
Composed of real-world metrics data collected from Datadog, a leading observability platform, the benchmark captures the irregularity, structural complexity, and heavy-tailed statistics typical of production observability data. Unlike synthetic or curated benchmarks, BOOM reflects the full diversity and unpredictability of operational signals observed in distributed systems, covering infrastructure, networking, databases, security, and application-level metrics.

Note: the metrics comprising BOOM were generated from internal monitoring of pre-production environments, and **do not** include any customer data. 

For more information on the dataset, including details on its preparation and statistical properties, see the [dataset card](https://huggingface.co/datasets/Datadog/BOOM) in Hugging Face.

For example evaluations of different time series models on the BOOM dataset, see the [boom](boom) folder in this repository.

## Citation

If you use Toto in your research, please cite our work:

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

## License
Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License - see [LICENSE](LICENSE) file for details.

This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2025 Datadog, Inc.

## Contributing

We welcome contributions! Please check out our [contributing guidelines](CONTRIBUTING.md) to get started.
