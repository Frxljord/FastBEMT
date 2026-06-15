"""Numerical and temporal settings for aeroacoustic simulations."""

from __future__ import annotations

import math

import numpy as np
import torch


class Simulation:
    """Time discretization and compute-device settings.

    Args:
        revolutions: Number of propeller revolutions to simulate.
        num_obs_times_per_rev: Observer time samples per revolution.
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
        self.num_obs_times_per_rev = int(timesteps_per_revolution)
        self.num_obs_times = self.num_obs_times_per_rev * self.revolutions
        self.device = torch.device(device)

        self.rpm: float | None = None
        self.omega: float | None = None
        self.duration: float | None = None
        self.blade_passing_period: float | None = None
        self.observer_time_range: float | None = None
        self.dt: float | None = None
        self.src_times: torch.Tensor | None = None
        self.src_times_one_rotation: torch.Tensor | None = None
        self.num_src_times: int | None = None
        self.blade_angles: torch.Tensor | None = None

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
        self.observer_time_range = self.duration
        self.dt = self.duration / self.num_obs_times
        self.src_times = (
            torch.arange(
                self.num_obs_times,
                dtype=torch.float32,
                device=self.device,
            )
            * self.dt
        )
        self.src_times_one_rotation = self.src_times[
            : self.num_obs_times_per_rev
        ]
        self.num_src_times = int(self.src_times.shape[0])
        self.blade_angles = (
            2.0
            * np.pi
            / int(n_blades)
            * torch.arange(
                int(n_blades),
                dtype=torch.float32,
                device=self.device,
            )
        )
