# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import platform
import warnings

import pytest
import torch


def skip_if_no_xformers():
    """Skip tests if xformers is not available, particularly on macOS."""
    if platform.system() == "Darwin":  # macOS is identified as 'Darwin'
        warnings.warn("Skipping tests because xformers is not unavailable on your system.")
        pytest.importorskip("xformers", reason="xformers module is not available on this system.")
        pytest.importorskip("triton", reason="triton module is not available on this system.")


def set_default_dtype():
    """Set the default torch dtype based on CUDA availability and GPU support."""
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = torch.cuda.get_device_capability(0)  # (major, minor)

        # Check if the GPU supports bfloat16 (Compute Capability 8.0+)
        if compute_capability >= (8, 0):
            torch.set_default_dtype(torch.bfloat16)
            print(f"Using torch.bfloat16 on {device_props.name} (Compute Capability {compute_capability})")
        else:
            torch.set_default_dtype(torch.float16)
            print(f"bfloat16 not supported on {device_props.name}. Using float16 instead.")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
        torch.set_default_dtype(torch.float32)
        print("Using torch.float32 with MPS.")
    else:
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
        print("CUDA not available. Using CPU with torch.float32.")
