"""Rigid-body kinematics for rotating propeller blade sections."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..Propeller import Propeller


class Kinematics:
    """Compute and store propeller blade-section kinematics.

    The propeller rotates about the global x-axis. Constructing this object
    configures the propeller simulation for the requested RPM, then evaluates
    every blade section at every source time. Time-dependent section tensors
    use dimension order ``(T, B, S, ...)``.

    Args:
        propeller: Propeller containing geometry and simulation settings.
        rpm: Rotational speed in revolutions per minute.
        source_times: Optional source emission timestamps in seconds. When
            omitted, the propeller simulation grid is configured from its
            revolutions and timesteps-per-revolution settings.
        section_geometry: Optional one-dimensional section data overriding the
            propeller section grid. Expected keys are ``"r"`` plus either
            ``"twist_rad"`` or ``"twist"``, ``"com_shift_forward"``, and
            ``"com_shift_up"``.
    """

    def __init__(
        self,
        propeller: Propeller,
        rpm: float,
        source_times: np.ndarray | torch.Tensor | None = None,
        section_geometry: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    ) -> None:
        self.propeller = propeller
        self.rpm = float(rpm)
        self.dtype: torch.dtype = propeller.dtype
        self.device: torch.device = torch.device(propeller.device)

        self.n_blades = propeller.n_blades
        self.uses_custom_source_times = source_times is not None
        self.uses_custom_section_geometry = section_geometry is not None

        if source_times is None:
            propeller.simulation.configure_operating_point(
                self.rpm,
                self.n_blades,
            )

            simulation = propeller.simulation
            if (
                simulation.omega is None
                or simulation.src_times is None
                or simulation.blade_angles is None
            ):
                raise RuntimeError(
                    "The simulation operating point was not configured."
                )

            self.omega = torch.tensor(
                simulation.omega,
                dtype=self.dtype,
                device=self.device,
            )
            self.source_times = torch.as_tensor(
                simulation.src_times,
                dtype=self.dtype,
                device=self.device,
            )
            self.blade_phase_offsets = torch.as_tensor(
                simulation.blade_angles,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            if not math.isfinite(self.rpm) or self.rpm <= 0.0:
                raise ValueError("rpm must be finite and greater than zero.")
            self.omega = torch.tensor(
                2.0 * math.pi * self.rpm / 60.0,
                dtype=self.dtype,
                device=self.device,
            )
            self.source_times = self._validate_source_times(source_times)
            self.blade_phase_offsets = (
                2.0
                * math.pi
                / self.n_blades
                * torch.arange(
                    self.n_blades,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self._mirror_custom_times_to_simulation()

        if section_geometry is None:
            self.radial_positions = propeller.section_radius
            self.twist_rad = propeller.section_twist_rad
            self.com_shift_forward = propeller.section_com_shift_forward
            self.com_shift_up = propeller.section_com_shift_up
        else:
            self.radial_positions = self._section_geometry_tensor(
                section_geometry,
                "r",
            )
            if "twist_rad" in section_geometry:
                self.twist_rad = self._section_geometry_tensor(
                    section_geometry,
                    "twist_rad",
                )
            elif "twist" in section_geometry:
                self.twist_rad = torch.deg2rad(
                    self._section_geometry_tensor(section_geometry, "twist")
                )
            else:
                raise ValueError(
                    "section_geometry must contain 'twist_rad' or 'twist'."
                )
            self.com_shift_forward = self._section_geometry_tensor(
                section_geometry,
                "com_shift_forward",
            )
            self.com_shift_up = self._section_geometry_tensor(
                section_geometry,
                "com_shift_up",
            )

            expected_shape = self.radial_positions.shape
            for name, values in (
                ("twist", self.twist_rad),
                ("com_shift_forward", self.com_shift_forward),
                ("com_shift_up", self.com_shift_up),
            ):
                if values.shape != expected_shape:
                    raise ValueError(
                        "section_geometry arrays must all match the shape of "
                        f"'r' {expected_shape}; '{name}' has shape {values.shape}."
                    )

        self.nt = int(self.source_times.shape[0])
        self.ns = int(self.radial_positions.shape[0])
        self.nb = int(self.blade_phase_offsets.shape[0])
        if self.nb != self.n_blades:
            raise ValueError(
                "The configured blade angles do not match geometry['n_blades']."
            )
        self._compute()

    def _section_geometry_tensor(
        self,
        section_geometry: Mapping[str, np.ndarray | torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        """Return a validated one-dimensional custom section array."""
        if name not in section_geometry:
            raise ValueError(f"section_geometry must contain '{name}'.")
        values = torch.as_tensor(
            section_geometry[name],
            dtype=self.dtype,
            device=self.device,
        )
        if values.ndim != 1:
            raise ValueError(f"section_geometry['{name}'] must be one-dimensional.")
        if values.numel() == 0:
            raise ValueError(f"section_geometry['{name}'] must not be empty.")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"section_geometry['{name}'] must be finite.")
        return values.contiguous()

    def _validate_source_times(
        self,
        source_times: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Return validated one-dimensional source timestamps."""
        source_times_tensor = torch.as_tensor(
            source_times,
            dtype=self.dtype,
            device=self.device,
        )
        if source_times_tensor.ndim != 1:
            raise ValueError("source_times must be one-dimensional.")
        if source_times_tensor.numel() == 0:
            raise ValueError("source_times must contain at least one timestamp.")
        if not bool(torch.isfinite(source_times_tensor).all().item()):
            raise ValueError("source_times must contain only finite values.")
        if source_times_tensor.numel() > 1 and not bool(
            torch.all(source_times_tensor[1:] > source_times_tensor[:-1]).item()
        ):
            raise ValueError("source_times must be strictly increasing.")
        return source_times_tensor.contiguous()

    def _mirror_custom_times_to_simulation(self) -> None:
        """Keep the shared Simulation timing fields consistent enough to inspect."""
        simulation = self.propeller.simulation
        simulation.rpm = self.rpm
        simulation.omega = float(self.omega.item())
        simulation.src_times = self.source_times
        simulation.num_src_times = int(self.source_times.shape[0])
        simulation.blade_angles = self.blade_phase_offsets
        simulation.num_obs_times = int(self.source_times.shape[0])

        if self.source_times.numel() == 1:
            simulation.revolutions = 1
            simulation.dt = None
            simulation.duration = 0.0
            simulation.observer_time_range = 0.0
            simulation.src_times_one_rotation = self.source_times
            simulation.num_obs_times_per_rev = 1
            simulation.blade_passing_period = (
                2.0 * math.pi / float(self.omega.item()) / self.n_blades
            )
            return

        time_deltas = torch.diff(self.source_times)
        median_dt = torch.median(time_deltas).item()
        source_duration = (
            self.source_times[-1] - self.source_times[0]
        ).item()
        rotation_period = 2.0 * math.pi / float(self.omega.item())
        estimated_revolutions = max(
            1,
            int(round(source_duration / rotation_period)),
        )
        first_rotation_end = self.source_times[0] + rotation_period
        first_rotation_mask = self.source_times < (
            first_rotation_end - 0.5 * median_dt
        )
        first_rotation_count = int(first_rotation_mask.sum().item())
        if first_rotation_count <= 0:
            first_rotation_count = min(
                int(self.source_times.shape[0]),
                max(1, int(round(rotation_period / median_dt))),
            )

        simulation.dt = float(median_dt)
        simulation.revolutions = estimated_revolutions
        simulation.duration = float(source_duration)
        simulation.observer_time_range = float(source_duration)
        simulation.blade_passing_period = rotation_period / self.n_blades
        simulation.num_obs_times_per_rev = first_rotation_count
        simulation.src_times_one_rotation = self.source_times[
            :first_rotation_count
        ]

    def _compute(self) -> None:
        """Compute rotations, section positions, and their time derivatives."""
        self.blade_angles = (
            self.omega * self.source_times[:, None]
            + self.blade_phase_offsets[None, :]
        )

        cos_angle = torch.cos(self.blade_angles)
        sin_angle = torch.sin(self.blade_angles)
        zeros = torch.zeros_like(cos_angle)
        ones = torch.ones_like(cos_angle)

        self.global_to_blade_rotation_matrix = torch.stack(
            [
                torch.stack([ones, zeros, zeros], dim=-1),
                torch.stack([zeros, cos_angle, sin_angle], dim=-1),
                torch.stack([zeros, -sin_angle, cos_angle], dim=-1),
            ],
            dim=-2,
        )
        self.blade_to_global_rotation_matrix = torch.stack(
            [
                torch.stack([ones, zeros, zeros], dim=-1),
                torch.stack([zeros, cos_angle, -sin_angle], dim=-1),
                torch.stack([zeros, sin_angle, cos_angle], dim=-1),
            ],
            dim=-2,
        )

        cos_twist = torch.cos(self.twist_rad)
        sin_twist = torch.sin(self.twist_rad)
        twist_zeros = torch.zeros_like(cos_twist)
        twist_ones = torch.ones_like(cos_twist)

        self.blade_to_airfoil_rotation_matrix = torch.stack(
            [
                torch.stack([-sin_twist, twist_zeros, -cos_twist], dim=-1),
                torch.stack([twist_zeros, twist_ones, twist_zeros], dim=-1),
                torch.stack([cos_twist, twist_zeros, -sin_twist], dim=-1),
            ],
            dim=-2,
        )
        self.airfoil_to_blade_rotation_matrix = torch.stack(
            [
                torch.stack([-sin_twist, twist_zeros, cos_twist], dim=-1),
                torch.stack([twist_zeros, twist_ones, twist_zeros], dim=-1),
                torch.stack([-cos_twist, twist_zeros, -sin_twist], dim=-1),
            ],
            dim=-2,
        )

        self.airfoil_shift_blade_frame = torch.stack(
            [
                self.com_shift_up,
                self.radial_positions,
                self.com_shift_forward,
            ],
            dim=-1,
        )
        # REPLACE THIS WITH TRUE COM
        self.section_position_airfoil_frame = torch.stack(
            [
                torch.zeros_like(self.radial_positions),
                torch.zeros_like(self.radial_positions),
                torch.zeros_like(self.radial_positions),
            ],
            dim=-1,
        )
        self.section_position_blade_frame = (
            torch.einsum(
                "sij,sj->si",
                self.airfoil_to_blade_rotation_matrix,
                self.section_position_airfoil_frame,
            )
            + self.airfoil_shift_blade_frame
        )
        self.section_position_global_frame = torch.einsum(
            "tbij,sj->tbsi",
            self.blade_to_global_rotation_matrix,
            self.section_position_blade_frame,
        ).contiguous()

        self.omega_vec = torch.zeros(
            (1, 1, 1, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.omega_vec[..., 0] = self.omega
        self.section_vel = torch.linalg.cross(
            self.omega_vec,
            self.section_position_global_frame,
            dim=-1,
        ).contiguous()
        self.section_acc = torch.linalg.cross(
            self.omega_vec,
            self.section_vel,
            dim=-1,
        ).contiguous()
        self.section_jerk = torch.linalg.cross(
            self.omega_vec,
            self.section_acc,
            dim=-1,
        ).contiguous()

        # Compact aliases for callers that already use the aeroacoustic names.
        self.R_g2b = self.global_to_blade_rotation_matrix
        self.R_b2g = self.blade_to_global_rotation_matrix
        self.R_b2a = self.blade_to_airfoil_rotation_matrix
        self.R_a2b = self.airfoil_to_blade_rotation_matrix
        self.section_position_in_airfoil_frame = (
            self.section_position_airfoil_frame
        )
        self.section_position_in_blade_frame = self.section_position_blade_frame
        self.section_position_in_global_frame = (
            self.section_position_global_frame
        )
        self.section_velocity = self.section_vel
        self.section_acceleration = self.section_acc
        self.section_jerk = self.section_jerk
        self.pos_airfoil = self.section_position_airfoil_frame
        self.pos_blade = self.section_position_blade_frame
        self.pos_fixed = self.section_position_global_frame
        self.vel_fixed = self.section_vel
        self.acc_fixed = self.section_acc
        self.jerk_fixed = self.section_jerk
