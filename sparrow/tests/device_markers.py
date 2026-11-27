"""The devices the device tests run on, and how closely a GPU must predict as the CPU does.

A test marked with one of these runs on every device this machine has, and is
skipped, with the reason, on the others: the CPU and MPS tests run on an
Apple-silicon Mac, and the CPU and CUDA ones on a machine with an NVIDIA GPU.
Used by test_devices.py and test_predictors_parrot2.py.
"""

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no NVIDIA GPU (CUDA) on this machine")
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="no Apple-silicon GPU (MPS) on this machine"
)
requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="fewer than two NVIDIA GPUs on this machine"
)

#: Every device sparrow runs on, each skipped where this machine does not have it
DEVICES = [
    pytest.param("cpu"),
    pytest.param("mps", marks=requires_mps),
    pytest.param("cuda", marks=requires_cuda),
]

#: How far a network's output on a GPU may be from its output on the CPU, in the
#: units of the output. On an Apple-silicon GPU (MPS), sparrow's 17 PARROT 1 networks are
#: at most 2.3e-5 from the CPU (the end-to-end distance, of up to 100 Angstroms),
#: over 60 random sequences of 5 to 400 residues. PARROT measured CUDA in full
#: float32 precision at ~1e-4 from the CPU, and in TF32 at up to ~1e-2, which this
#: rejects.
DEVICE_ABSOLUTE_TOLERANCE = 5e-4
DEVICE_RELATIVE_TOLERANCE = 1e-5
