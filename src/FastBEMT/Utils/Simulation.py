"""Numerical and temporal settings for aeroacoustic simulations."""

from __future__ import annotations

import math

import numpy as np
import torch


class Simulation:
    """Time discretization and compute-device settings.

    Args:
        revolutions: Number of propeller revolutions to simulate.
        timesteps_per_revolution: Source and observer samples per revolution.
        device: PyTorch device, such as ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        revolutions: int,
        timesteps_per_revolution: int,
        device: str | torch.device,
    ) -> None:
        if revolutions <= 0:
            raise ValueError("revolutions must be greater than zero.")
        if timesteps_per_revolution <= 0:
            raise ValueError("timesteps_per_revolution must be greater than zero.")

        self.revolutions = int(revolutions)
        self.timesteps_per_revolution = int(timesteps_per_revolution)
        self.n_timesteps = self.timesteps_per_revolution * self.revolutions
        self.device = torch.device(device)

        self.rpm: float | None = None
        self.omega: float | None = None
        self.duration: float | None = None
        self.blade_passing_period: float | None = None
        self.observer_duration: float | None = None
        self.dt: float | None = None
        self.source_times: torch.Tensor | None = None
        self.source_times_one_revolution: torch.Tensor | None = None
        self.blade_phase_offsets_rad: torch.Tensor | None = None

    def configure_operating_point(self, rpm: float, n_blades: int) -> None:
        """Build RPM-dependent source and observer time grids.

        Args:
            rpm: Rotational speed in revolutions per minute.
            n_blades: Number of propeller blades.
        """
        rpm = float(rpm)
        if not math.isfinite(rpm) or rpm <= 0.0:
            raise ValueError("rpm must be a finite value greater than zero.")
        if n_blades <= 0:
            raise ValueError("n_blades must be greater than zero.")

        self.rpm = rpm
        self.omega = 2.0 * np.pi * rpm / 60.0
        self.duration = self.revolutions * (2.0 * np.pi / self.omega)
        self.blade_passing_period = (
            self.duration / self.revolutions / int(n_blades)
        )
        self.observer_duration = self.duration
        self.dt = self.duration / self.n_timesteps
        self.source_times = (
            torch.arange(
                self.n_timesteps,
                dtype=torch.float32,
                device=self.device,
            )
            * self.dt
        )
        self.source_times_one_revolution = self.source_times[
            : self.timesteps_per_revolution
        ]
        self.blade_phase_offsets_rad = (
            2.0
            * np.pi
            / int(n_blades)
            * torch.arange(
                int(n_blades),
                dtype=torch.float32,
                device=self.device,
            )
        )

    def configure_custom_source_times(
        self,
        rpm: float,
        n_blades: int,
        source_times: np.ndarray | torch.Tensor,
    ) -> None:
        """Configure metadata from an externally supplied source-time grid."""
        self.rpm = float(rpm)
        self.omega = 2.0 * np.pi * self.rpm / 60.0
        self.source_times = torch.as_tensor(
            source_times,
            device=self.device,
        ).contiguous()
        self.n_timesteps = int(self.source_times.shape[0])
        self.blade_phase_offsets_rad = (
            2.0
            * np.pi
            / int(n_blades)
            * torch.arange(
                int(n_blades),
                dtype=self.source_times.dtype,
                device=self.device,
            )
        )

        time_deltas = torch.diff(self.source_times)
        self.dt = float(torch.median(time_deltas).item())
        self.duration = float((self.source_times[-1] - self.source_times[0]).item())
        rotation_period = 2.0 * np.pi / self.omega
        self.revolutions = max(1, int(round(self.duration / rotation_period)))
        self.observer_duration = self.duration
        self.blade_passing_period = rotation_period / int(n_blades)

        first_rotation_end = self.source_times[0] + rotation_period
        first_rotation_count = int(
            (
                self.source_times
                < first_rotation_end - 0.5 * self.dt
            ).sum().item()
        )
        self.timesteps_per_revolution = first_rotation_count
        self.source_times_one_revolution = self.source_times[:first_rotation_count]
