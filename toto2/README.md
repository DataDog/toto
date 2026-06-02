# Toto 2.0

[Technical Report](https://arxiv.org/abs/2605.20119) | [Blog](https://www.datadoghq.com/blog/ai/toto-2/) | [Model Collection](https://huggingface.co/collections/Datadog/toto-20) | [BOOM Dataset](https://huggingface.co/datasets/Datadog/BOOM) | [GitHub](https://github.com/DataDog/toto)

Toto 2.0 is a family of foundation models for multivariate time series forecasting, built by Datadog. It features a u-μP-scaled transformer with alternating time/variate attention and quantile-based probabilistic forecasting, ranging from 4M to 2.5B parameters.

> **Note:** Fine-tuning and exogenous variable support are planned for a future 2.0 release. If you need these features today, see [toto-ts](https://pypi.org/project/toto-ts/) (Toto 1.0).

## Features

- **Zero-Shot Forecasting** — No task-specific fine-tuning required.
- **State-of-the-Art Performance** — Top results on [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [BOOM](https://huggingface.co/datasets/Datadog/BOOM).
- **Multivariate Support** — Efficiently handles multiple variables via alternating time/variate attention.
- **Probabilistic Predictions** — Returns 9 quantile levels (0.1–0.9) for uncertainty estimation.
- **High-Dimensional Support** — Scales to time series with a large number of variates.
- **Decoder-Only Architecture** — Supports variable prediction horizons and context lengths.

## Model Weights

| Checkpoint | Parameters |
|---|---|
| [Toto-2.0-4m](https://huggingface.co/Datadog/Toto-2.0-4m) | 4M |
| [Toto-2.0-22m](https://huggingface.co/Datadog/Toto-2.0-22m) | 22M |
| [Toto-2.0-313m](https://huggingface.co/Datadog/Toto-2.0-313m) | 313M |
| [Toto-2.0-1B](https://huggingface.co/Datadog/Toto-2.0-1B) | 1B |
| [Toto-2.0-2.5B](https://huggingface.co/Datadog/Toto-2.0-2.5B) | 2.5B |

## Installation

Requires Python 3.12+ and PyTorch 2.5+. A CUDA-capable GPU (Ampere or newer) is recommended.

```bash
pip install toto-2
```

Or install the [`toto-models`](https://pypi.org/project/toto-models/) umbrella package, which includes `toto-2` and its dependencies:

```bash
pip install toto-models
```

## Quick Start

```python
import torch
from toto2 import Toto2Model

model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Input shape: (batch, n_variates, time_steps)
target = torch.randn(1, 1, 512, device=device)
target_mask = torch.ones_like(target, dtype=torch.bool)
series_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

# Returns quantiles of shape (9, batch, n_variates, horizon)
# Quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
quantiles = model.forecast(
    {"target": target, "target_mask": target_mask, "series_ids": series_ids},
    horizon=96,
    decode_block_size=768,   # None for single forward pass (faster, better short-term)
    has_missing_values=False, # Set True if target_mask contains False entries
)
```

**Inference tips:**
- `decode_block_size=None` — single forward pass, faster and better for short horizons. Used for all leaderboard results.
- `decode_block_size=768` — block decoding, better long-term stability for horizons ≳1000. Default in notebooks.
- `has_missing_values=False` — enables Flash Attention kernels when your context has no gaps.

## Tutorials

- [Quick Start](notebooks/quick_start.ipynb) — Load a model, forecast, plot results, handle missing values and multivariate inputs.
- [GluonTS Integration](notebooks/gluonts_integration.ipynb) — Use `Toto2GluonTSModel` with GluonTS evaluation pipelines and built-in datasets.

## Evaluation

- [GIFT-Eval Notebook](https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/toto_2_0.ipynb) — Evaluate on the GIFT-Eval benchmark.
- [BOOM Evaluation](https://github.com/DataDog/toto/blob/main/boom/notebooks/toto.ipynb) — Evaluate on the BOOM observability benchmark.

## Citation

If you use Toto 2.0 in your research, please cite:

```bibtex
@misc{khwaja2026toto20timeseries,
      title={Toto 2.0: Time Series Forecasting Enters the Scaling Era},
      author={Emaad Khwaja and Chris Lettieri and Gerald Woo and Eden Belouadah and Marc Cenac and Guillaume Jarry and Enguerrand Paquin and Xunyi Zhao and Viktoriya Zhukov and Othmane Abou-Amal and Chenghao Liu and Ameet Talwalkar and David Asker},
      year={2026},
      eprint={2605.20119},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.20119},
}
```

## License

Apache-2.0. See [LICENSE](https://github.com/DataDog/toto/blob/main/LICENSE) for details.
