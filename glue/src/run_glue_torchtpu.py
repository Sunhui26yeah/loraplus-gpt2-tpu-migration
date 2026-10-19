"""Run run_glue.py on a TPU through TorchTPU (the PyTorch-native TPU backend).

accelerate supports overriding the device with ACCELERATE_TORCH_DEVICE.
torch_xla must NOT be installed in this environment.
"""
import os
import sys

os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "tpu")

import torch_tpu  # noqa: E402,F401  registers the "tpu" device

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_glue  # noqa: E402

if __name__ == "__main__":
    run_glue.main()
