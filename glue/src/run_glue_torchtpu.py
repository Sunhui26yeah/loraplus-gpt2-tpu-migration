"""Run run_glue.py on a TPU through TorchTPU (the PyTorch-native TPU backend)."""
import os
import sys

import torch
import torch_tpu  # noqa: F401  registers the "tpu" device

TPU = torch.device("tpu")

from transformers import training_args as _ta  # noqa: E402

_ta.TrainingArguments.device = property(lambda self: TPU)
_ta.TrainingArguments.n_gpu = property(lambda self: 1)

from accelerate import state as _st  # noqa: E402

_st.PartialState.device = property(lambda self: TPU)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_glue  # noqa: E402

if __name__ == "__main__":
    run_glue.main()
