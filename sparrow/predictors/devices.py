"""Choosing the device sparrow's networks run on: the CPU, an NVIDIA GPU (CUDA) or an Apple-silicon GPU (MPS).

:func:`~sparrow.predictors.batch_predictor.batch_predict`,
:func:`~sparrow.predictors.batch_predictor.predict_many`,
:func:`~sparrow.predictors.batch_predictor.prepare_model` and
:func:`sparrow.batch` take the same choice of device as PARROT's ``Predictor``:
``'auto'``, ``'cpu'``, ``'cuda'`` or ``'mps'``, with ``gpuid`` choosing which
NVIDIA GPU, as PARROT's ``gpu_id`` does. The predictor classes
(:class:`~sparrow.predictors.base.BaseNetworkPredictor`) take it as their
``device`` too. :func:`resolve_device` turns a choice into the
``torch.device`` to run on:

* ``'auto'``, the default of the batch functions, uses CUDA if this machine has
  it and the CPU otherwise. Unlike PARROT's ``'auto'``, it never chooses MPS by
  itself: sparrow's networks are small, and run 4.5 to 7 times faster on the
  CPU than on MPS (the rg, dssp and transmembrane networks, on 500 sequences of
  30 to 600 residues, on an Apple-silicon Mac), as metapredict found for its
  small networks too. With ``'auto'``, ``gpuid`` chooses the NVIDIA GPU where
  there is one and is otherwise ignored, so code written for a GPU runs
  unchanged on a laptop, as it always has.
* ``'cpu'``, ``'cuda'`` and ``'mps'`` run on that device, and asking for one
  this machine does not have is an error rather than a silent fall back to the
  CPU. ``'cuda'`` runs on the GPU ``gpuid`` names, or else the current one. A
  ``gpuid`` with ``'cpu'`` or ``'mps'`` is an error, as it is in PARROT.
* A device as torch names it, such as ``'cuda:1'`` or a ``torch.device``, which
  the predictor classes have always taken, is accepted too.

On an NVIDIA GPU, sparrow runs its networks in full float32 precision
(:func:`full_float32_precision`), as PARROT 2 does, so a network gives the same
predictions on every device up to float32 rounding, whichever PARROT saved it.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

from sparrow.sparrow_exceptions import SparrowException

#: The device choices, as PARROT's ``Predictor`` takes them
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")

#: The choice the batch functions make when none is given
DEFAULT_DEVICE = "auto"

# The kinds of device sparrow runs on, as torch names them
_DEVICE_TYPES = ("cpu", "cuda", "mps")

# PyTorch's fp32_precision value (PyTorch 2.9 and later) for float32 computed in
# full precision, rather than in TF32 ('tf32') or as PyTorch decides ('none')
_FULL_FLOAT32_PRECISION = "ieee"


def resolve_device(device: str | torch.device = DEFAULT_DEVICE, gpuid: int | None = None) -> torch.device:
    """Turn a choice of device into the ``torch.device`` to run on.

    Parameters
    ----------
    device : str or torch.device, optional
        ``'auto'`` (the default: CUDA if available, else the CPU), ``'cpu'``,
        ``'cuda'`` or ``'mps'``; or a device as torch names it, such as
        ``'cuda:1'``.
    gpuid : int or None, optional
        The index of the NVIDIA GPU to run on. With ``'auto'``, it is used
        only where CUDA is available. It cannot be given with ``'cpu'`` or
        ``'mps'``, or differ from the index a device such as ``'cuda:1'``
        names. Default None, the current CUDA device.

    Returns
    -------
    torch.device
        The device. A CUDA device always has its index.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``device`` is not a choice above, ``gpuid`` is not an int or does
        not fit ``device``, or the device asked for is not on this machine.
    """
    if gpuid is not None and (isinstance(gpuid, bool) or not isinstance(gpuid, int)):
        raise SparrowException(f"gpuid must be the index of an NVIDIA GPU, an int, not {gpuid!r}.")

    choice, index = _choice_and_index(device)
    if index is not None:
        if gpuid is not None and gpuid != index:
            raise SparrowException(f"device {str(device)!r} names CUDA device {index}, but gpuid is {gpuid}.")
        gpuid = index

    if gpuid is not None and choice in ("cpu", "mps"):
        raise SparrowException(
            f"gpuid ({gpuid}) chooses an NVIDIA GPU, so it cannot be given with device {choice!r}: use "
            "device='cuda' or 'auto', or leave gpuid out."
        )

    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        return _mps_device()
    if choice == "cuda" or torch.cuda.is_available():
        return _cuda_device(gpuid)
    # 'auto' on a machine without CUDA
    return torch.device("cpu")


def _choice_and_index(device: str | torch.device) -> tuple[str, int | None]:
    """A device as one of :data:`DEVICE_CHOICES`, with the CUDA index it names, if any.

    Parameters
    ----------
    device : str or torch.device
        As for :func:`resolve_device`.

    Returns
    -------
    tuple
        ``(choice, index)``: the choice, and the CUDA device index a device
        such as ``'cuda:1'`` names, or None.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``device`` is neither one of :data:`DEVICE_CHOICES` nor a CPU, CUDA
        or MPS device as torch names it.
    """
    if isinstance(device, str) and device in DEVICE_CHOICES:
        return device, None

    unknown = SparrowException(
        f"Unknown device {device!r}: choose from 'auto', 'cpu', 'cuda' or 'mps', and choose an NVIDIA GPU with "
        "gpuid (or as 'cuda:1')."
    )
    try:
        named = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise unknown from error
    if named.type not in _DEVICE_TYPES:
        raise unknown

    # only a CUDA device's index chooses anything: there is one CPU and one MPS device
    index = named.index if named.type == "cuda" else None
    return named.type, index


def _cuda_device(gpuid: int | None) -> torch.device:
    """The CUDA device ``gpuid``, or the current one, checking it exists.

    Parameters
    ----------
    gpuid : int or None
        The GPU's index, or None for the current CUDA device.

    Returns
    -------
    torch.device
        The CUDA device, with its index.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If CUDA is not available, or there is no GPU ``gpuid``.
    """
    if not torch.cuda.is_available():
        raise SparrowException(
            "An NVIDIA GPU (CUDA) was asked for, but CUDA is not available on this machine. Use device='auto' to "
            "run on CUDA where there is one and on the CPU otherwise."
        )
    if gpuid is None:
        gpuid = torch.cuda.current_device()

    number_of_gpus = torch.cuda.device_count()
    if not 0 <= gpuid < number_of_gpus:
        raise SparrowException(f"There is no CUDA device {gpuid}: this machine has {number_of_gpus}, numbered from 0.")
    return torch.device("cuda", gpuid)


def _mps_device() -> torch.device:
    """The MPS device, checking it is available.

    Returns
    -------
    torch.device
        The MPS device.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If MPS is not available.
    """
    if not torch.backends.mps.is_available():
        reason = (
            "this PyTorch was built without MPS support"
            if not torch.backends.mps.is_built()
            else "this machine has no Apple-silicon GPU, or its macOS is too old"
        )
        raise SparrowException(f"An Apple-silicon GPU (MPS) was asked for, but MPS is not available: {reason}.")
    return torch.device("mps")


@contextlib.contextmanager
def full_float32_precision() -> Iterator[None]:
    """Compute float32 LSTMs, convolutions and matrix products in full precision on NVIDIA GPUs.

    On Ampere and newer NVIDIA GPUs, PyTorch lets cuDNN compute float32 LSTMs
    and convolutions in TF32 by default, which keeps only 10 bits of the
    mantissa. PARROT measured BRNN predictions on CUDA differing from the same
    network's on the CPU by up to ~1e-2 (in the units of the prediction) in
    TF32, and by ~1e-4 in full precision, at no cost in speed for a BRNN (on an
    RTX A4000), so PARROT 2 predicts in full precision; this is PARROT 2's
    ``parrot.devices.full_float32_precision``, which sparrow cannot import
    because it needs only PARROT 1. sparrow runs its own networks inside it, so
    a network saved by PARROT 1 is computed as one saved by PARROT 2 is.

    The previous settings are restored afterwards, so other code running in the
    same process is unaffected. It has no effect on the CPU or MPS. PyTorch 2.9
    replaced the ``allow_tf32`` flags with ``fp32_precision`` settings, and
    raises if a process mixes the two, so the newer settings are used wherever
    they exist.

    Yields
    ------
    None
        Inside the ``with`` block, float32 is computed in full precision.
    """
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        # Typed as Any: PyTorch's type hints do not describe these settings yet
        cudnn: Any = torch.backends.cudnn
        settings: list[Any] = [torch.backends.cuda.matmul, cudnn.conv, cudnn.rnn]
        previous = [setting.fp32_precision for setting in settings]
        for setting in settings:
            setting.fp32_precision = _FULL_FLOAT32_PRECISION
        try:
            yield
        finally:
            for setting, value in zip(settings, previous):
                setting.fp32_precision = value
        return

    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
