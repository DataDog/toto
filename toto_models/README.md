# toto-models

[GitHub](https://github.com/DataDog/toto) | [Toto 2.0 Report](https://arxiv.org/abs/2605.20119) | [Toto 2.0 Blog](https://www.datadoghq.com/blog/ai/toto-2/) | [Model Collection](https://huggingface.co/collections/Datadog/toto-20) | [BOOM Dataset](https://huggingface.co/datasets/Datadog/BOOM)

`toto-models` is the recommended way to install Datadog's Toto time series models. It is an umbrella package that pulls in [`toto-2`](https://pypi.org/project/toto-2/) and its dependencies so you don't have to manage them individually.

## Installation

```bash
pip install toto-models
```

To also install Toto 1.0 (for fine-tuning or exogenous variable support, not yet available in 2.0):

```bash
pip install "toto-models[v1]"
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
quantiles = model.forecast(
    {"target": target, "target_mask": target_mask, "series_ids": series_ids},
    horizon=96,
    decode_block_size=768,
    has_missing_values=False,
)
```

For full documentation, see the [`toto-2` package](https://pypi.org/project/toto-2/) or the [GitHub repository](https://github.com/DataDog/toto).

## Citation

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
