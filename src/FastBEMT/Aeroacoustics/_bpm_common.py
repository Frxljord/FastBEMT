"""Shared tensor helpers for BPM component correlations."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def torch_select(
    conditions: Sequence[torch.Tensor],
    choices: Sequence[torch.Tensor | float],
    default_value: float = 0.0,
) -> torch.Tensor:
    """Torch-based conditional selection similar to ``numpy.select``."""
    result = torch.full_like(conditions[0], default_value, dtype=conditions[0].dtype)
    for condition, choice in zip(reversed(conditions), reversed(choices)):
        if not isinstance(choice, torch.Tensor):
            choice = torch.tensor(choice, dtype=result.dtype, device=result.device)
        result = torch.where(condition, choice, result)
    return result


def st(
    frequency: torch.Tensor,
    length: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """Compute Strouhal number."""
    return frequency[:, None] * length / velocity[None, :]


def safe_log10(value: torch.Tensor) -> torch.Tensor:
    """Compute ``log10`` with a tiny clamp instead of additive bias."""
    return torch.log10(torch.clamp(value, min=torch.finfo(value.dtype).tiny))
