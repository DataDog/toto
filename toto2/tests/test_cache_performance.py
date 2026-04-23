"""Test KV cache and block decoding performance on sinusoids."""
import time

import numpy as np
import torch

from toto2 import Toto2Model


def generate_sinusoid_inputs(
    num_series: int = 10,
    context_length: int = 1024,
    prediction_length: int = 2048,
):
    """Generate synthetic sinusoid inputs as tensors."""
    batch = []
    for i in range(num_series):
        # Multiple sinusoids with different frequencies
        t = np.arange(context_length)
        target = (
            np.sin(2 * np.pi * t / 100) +
            0.5 * np.sin(2 * np.pi * t / 50 + i) +
            0.3 * np.sin(2 * np.pi * t / 200)
        )
        batch.append(target)
    
    target_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1)  # [B, 1, T]
    
    inputs = {
        "target": target_tensor,
        "target_mask": torch.ones_like(target_tensor, dtype=torch.bool),
        "series_ids": torch.zeros((num_series, 1), dtype=torch.long),
    }
    
    return inputs, prediction_length


def run_benchmark(
    model,
    inputs,
    prediction_length,
    decode_block_size,
    num_samples=100,
    warmup=True,
    device="cpu",
):
    """Run inference and measure time."""
    # Move inputs to device
    inputs_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
    
    if warmup:
        # Warmup run
        with torch.no_grad():
            _ = model.forecast(inputs_device, horizon=prediction_length, 
                             num_samples=10, decode_block_size=decode_block_size)
    
    start = time.perf_counter()
    with torch.no_grad():
        forecasts = model.forecast(
            inputs_device,
            horizon=prediction_length,
            num_samples=num_samples,
            decode_block_size=decode_block_size,
        )
    elapsed = time.perf_counter() - start
    
    return forecasts, elapsed


def compute_metrics(forecasts):
    """Compute forecast metrics."""
    # forecasts shape: [num_quantiles, batch, num_variates, horizon]
    forecasts_np = forecasts.cpu().numpy()
    
    return {
        "mean": float(np.mean(forecasts_np)),
        "std": float(np.std(forecasts_np)),
        "min": float(np.min(forecasts_np)),
        "max": float(np.max(forecasts_np)),
        "shape": forecasts_np.shape,
    }


def main():
    print("=" * 80)
    print("KV Cache and Block Decoding Performance Test")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model_path = "/Users/emaad.khwaja/Developer/foundation-models-research/models/Toto-Open-Mini-2.0"
    
    if torch.cuda.is_available():
        device_name = "cuda"
        device_desc = "CUDA"
    elif torch.backends.mps.is_available():
        device_name = "mps"
        device_desc = "MPS"
    else:
        device_name = "cpu"
        device_desc = "CPU"
    
    model = Toto2Model.from_pretrained(model_path, map_location=device_name)
    model = model.to(device_name).eval()
    
    print(f"Device: {device_desc}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate test data
    print("\nGenerating sinusoid inputs...")
    inputs, prediction_length = generate_sinusoid_inputs(
        num_series=10,
        context_length=1024,
        prediction_length=2048,
    )
    print(f"Context length: 1024")
    print(f"Prediction length: {prediction_length}")
    print(f"Input shape: {inputs['target'].shape}")
    
    # Test configurations
    configs = [
        {"name": "Single-pass (no cache)", "decode_block_size": 0},
        {"name": "Block decode 768tp (cache)", "decode_block_size": 768},
        {"name": "Block decode 384tp (cache)", "decode_block_size": 384},
        {"name": "Block decode 1536tp (cache)", "decode_block_size": 1536},
        {"name": "Oversized block (no-op)", "decode_block_size": 4096},
    ]
    
    num_samples = 100
    print(f"\nNum samples: {num_samples}")
    print("\n" + "=" * 80)
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 80)
        
        forecasts, elapsed = run_benchmark(
            model,
            inputs,
            prediction_length,
            config["decode_block_size"],
            num_samples=num_samples,
            warmup=True,
            device=device_name,
        )
        
        metrics = compute_metrics(forecasts)
        
        result = {
            "config": config["name"],
            "decode_block_size": config["decode_block_size"],
            "elapsed_time": elapsed,
            "time_per_series": elapsed / 10,
            "metrics": metrics,
        }
        results.append(result)
        
        print(f"Total time: {elapsed:.3f}s")
        print(f"Time per series: {elapsed / 10:.3f}s")
        print(f"Forecast shape: {metrics['shape']}")
        print(f"Forecast stats: mean={metrics['mean']:.3f}, std={metrics['std']:.3f}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Configuration':<30} {'Time (s)':<12} {'Time/series (s)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_time = results[0]["elapsed_time"]
    for result in results:
        speedup = baseline_time / result["elapsed_time"]
        print(
            f"{result['config']:<30} "
            f"{result['elapsed_time']:<12.3f} "
            f"{result['time_per_series']:<15.3f} "
            f"{speedup:<10.2f}x"
        )
    
    # Test determinism
    print("\n" + "=" * 80)
    print("DETERMINISM TEST")
    print("=" * 80)
    
    print("\nTesting if same config produces identical results...")
    forecasts1, _ = run_benchmark(model, inputs, prediction_length, 768, num_samples=10, 
                                  warmup=False, device=device_name)
    forecasts2, _ = run_benchmark(model, inputs, prediction_length, 768, num_samples=10, 
                                  warmup=False, device=device_name)
    
    samples1 = forecasts1.cpu().numpy()
    samples2 = forecasts2.cpu().numpy()
    
    if np.allclose(samples1, samples2, rtol=1e-5, atol=1e-5):
        print("✅ PASS: Repeated runs produce identical results")
    else:
        max_diff = np.max(np.abs(samples1 - samples2))
        print(f"❌ FAIL: Results differ (max diff: {max_diff:.2e})")
    
    # Test single-pass equivalence
    print("\nTesting single-pass equivalence (decode_block_size=0 vs 4096)...")
    forecasts_no_cache, _ = run_benchmark(model, inputs, prediction_length, 0, num_samples=10, 
                                          warmup=False, device=device_name)
    forecasts_oversized, _ = run_benchmark(model, inputs, prediction_length, 4096, num_samples=10, 
                                           warmup=False, device=device_name)
    
    samples_no_cache = forecasts_no_cache.cpu().numpy()
    samples_oversized = forecasts_oversized.cpu().numpy()
    
    if np.allclose(samples_no_cache, samples_oversized, rtol=1e-5, atol=1e-5):
        print("✅ PASS: No-cache and oversized block produce identical results")
    else:
        max_diff = np.max(np.abs(samples_no_cache - samples_oversized))
        print(f"❌ FAIL: Results differ (max diff: {max_diff:.2e})")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
