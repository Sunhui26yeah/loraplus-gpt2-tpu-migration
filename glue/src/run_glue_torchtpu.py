"""Run run_glue.py on a TPU through TorchTPU (the PyTorch-native TPU backend)."""
import os
import sys

import torch
import torch_tpu  # noqa: F401  registers the "tpu" device

TPU = torch.device("tpu")

from transformers import training_args as _ta  # noqa: E402

# keep the original property getters (they run _setup_devices, which also creates
# distributed_state) and only replace the device they return
_orig_device = _ta.TrainingArguments.device.fget
_orig_n_gpu = _ta.TrainingArguments.n_gpu.fget
_ta.TrainingArguments.device = property(lambda self: (_orig_device(self), TPU)[1])
_ta.TrainingArguments.n_gpu = property(lambda self: (_orig_n_gpu(self), 1)[1])

from accelerate import state as _st  # noqa: E402

_orig_ps_device = _st.PartialState.device.fget
_st.PartialState.device = property(lambda self: (_orig_ps_device(self), TPU)[1])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_glue  # noqa: E402

if __name__ == "__main__":
    run_glue.main()
