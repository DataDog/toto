"""Inference on Apple MPS, which has no float64.

Verifies:
  1. The causal scaler gives the CPU (float64) statistics on MPS, without NaN.
  2. The GluonTS predictor batches past_is_pad as float32, so batches can move to MPS.
  3. The GluonTS predictor forecasts on MPS as on CPU.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from gluonts.dataset.common import ListDataset

from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig, Toto2Model
from toto2.configuration import Toto2ModelConfig
from toto2.model import PatchedCausalStdScaler

PATCH_SIZE = 32
CONTEXT = 4096
HORIZON = 64

requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple MPS device")


def _tiny_model() -> Toto2Model:
    torch.manual_seed(0)
    config = Toto2ModelConfig(
        patch_size=PATCH_SIZE,
        d_model=64,
        num_heads=4,
        num_layers=4,
        layer_group_size=2,
        num_variate_layers_per_group=1,
        variate_layer_first=True,
        residual_attn_ratio=Toto2ModelConfig.compute_residual_attn_ratio(CONTEXT, PATCH_SIZE),
    )
    return Toto2Model(config).eval()


def _gluonts_model() -> Toto2GluonTSModel:
    config = Toto2GluonTSModelConfig(prediction_length=HORIZON, context_length=CONTEXT, target_dim=1)
    return Toto2GluonTSModel(_tiny_model(), config).eval()


def _dataset() -> ListDataset:
    rng = np.random.default_rng(0)
    start = pd.Period("2026-01-01", freq="min")
    series = [
        np.full(CONTEXT, 1e7, dtype=np.float32),  # a flat gauge at 1e7: NaN scale in float32 on MPS
        (100 + rng.normal(size=CONTEXT)).astype(np.float32),
        np.zeros(CONTEXT // 2, dtype=np.float32),  # shorter than the context: padded
    ]
    return ListDataset([{"start": start, "target": s} for s in series], freq="min")


@requires_mps
def test_scaler_on_mps_matches_cpu():
    scaler = PatchedCausalStdScaler(patch_size=PATCH_SIZE)
    data = torch.full((2, CONTEXT), 1e7)
    _, loc_cpu, scale_cpu = scaler(data)
    _, loc_mps, scale_mps = scaler(data.to("mps"))
    assert loc_mps.device.type == "mps" and scale_mps.device.type == "mps"
    torch.testing.assert_close(loc_mps.cpu(), loc_cpu)
    torch.testing.assert_close(scale_mps.cpu(), scale_cpu)


def test_predictor_batches_past_is_pad_as_float32():
    predictor = _gluonts_model().create_predictor(batch_size=4, device="cpu")
    entries = list(predictor.input_transform.apply(_dataset(), is_train=False))
    assert {e["past_is_pad"].dtype for e in entries} == {np.dtype(np.float32)}


@requires_mps
def test_predictor_on_mps_matches_cpu():
    forecasts = {}
    for device in ("cpu", "mps"):
        predictor = _gluonts_model().to(device).create_predictor(batch_size=4, device=device)
        forecasts[device] = np.stack([f.forecast_array for f in predictor.predict(_dataset())])
    assert np.isfinite(forecasts["mps"]).all()
    np.testing.assert_allclose(forecasts["mps"], forecasts["cpu"], rtol=1e-3, atol=1e-3)
