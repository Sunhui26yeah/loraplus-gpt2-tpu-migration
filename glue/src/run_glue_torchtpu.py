"""Run run_glue.py on a TPU through TorchTPU (the PyTorch-native TPU backend).

transformers 4.37.2 and accelerate 0.27.2 predate TorchTPU and only know cuda/mps/xla/cpu,
so on a TPU VM they would silently train on the CPU. accelerate supports overriding the
device with ACCELERATE_TORCH_DEVICE (model and dataloader placement), and TrainingArguments.device
(where Trainer moves each batch) is redirected to "tpu" below. torch_xla must NOT be installed in this environment
(otherwise run_glue switches to the PyTorch/XLA trainer).
"""
import os
import sys

os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "tpu")

import torch  # noqa: E402
import torch_tpu  # noqa: E402,F401  registers the "tpu" device
from transformers import training_args as _ta  # noqa: E402

# On a machine without CUDA, TrainingArguments._setup_devices resolves to "cpu" regardless of
# accelerate's device, and Trainer moves every batch to args.device. Keep the original getter
# (it runs _setup_devices and creates distributed_state) and only replace the device it returns.
_orig_device = _ta.TrainingArguments.device.fget
_ta.TrainingArguments.device = property(lambda self: (_orig_device(self), torch.device("tpu"))[1])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_glue  # noqa: E402

if __name__ == "__main__":
    run_glue.main()
