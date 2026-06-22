"""Shared helpers for aeroacoustic solvers."""

from __future__ import annotations

from os import PathLike

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor
PathInput = str | PathLike[str]


def observer_tensor(
    observers: ArrayLike,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return observer coordinates as a contiguous ``(O, 3)`` tensor."""
    values = torch.as_tensor(observers, dtype=dtype, device=device)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("observers must have shape (O, 3).")
    return values.contiguous()


def normalize_observer_batch_size(
    observer_batch_size: int | None,
    *,
    observer_count: int,
) -> int | None:
    """Return a useful observer batch size, or ``None`` for one full batch."""
    if observer_batch_size is None:
        return None
    batch_size = int(observer_batch_size)
    if batch_size <= 0:
        raise ValueError("observer_batch_size must be greater than zero.")
    if batch_size >= observer_count:
        return None
    return batch_size
