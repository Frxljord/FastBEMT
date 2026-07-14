"""Rigid-body kinematics for rotating propeller blade sections."""

from __future__ import annotations

from collections.abc import Mapping
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
            propeller section grid. Expected keys are ``"r"``, ``"twist"``
            in degrees, ``"chord"``, ``"sweep"``, and ``"rake"``.
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

        simulation = propeller.simulation
        if source_times is None:
            simulation.configure_operating_point(
                self.rpm,
                self.n_blades,
            )
        else:
            simulation.configure_custom_source_times(
                self.rpm,
                self.n_blades,
                source_times,
            )

        self.omega = torch.tensor(
            simulation.omega,
            dtype=self.dtype,
            device=self.device,
        )
        self.source_times = torch.as_tensor(
            simulation.source_times,
            dtype=self.dtype,
            device=self.device,
        )
        self.blade_phase_offsets_rad = torch.as_tensor(
            simulation.blade_phase_offsets_rad,
            dtype=self.dtype,
            device=self.device,
        )

        if section_geometry is None:
            self.section_radius = propeller.section_radius
            self.section_chord = propeller.section_chord
            self.section_twist_rad = propeller.section_twist_rad
            self.section_sweep = propeller.section_sweep
            self.section_rake = propeller.section_rake
        else:
            self.section_radius = self._section_geometry_tensor(
                section_geometry,
                "r",
            )
            self.section_chord = self._section_geometry_tensor(
                section_geometry,
                "chord",
            )
            self.section_twist_rad = torch.deg2rad(
                self._section_geometry_tensor(section_geometry, "twist")
            )
            self.section_sweep = self._section_geometry_tensor(
                section_geometry,
                "sweep",
            )
            self.section_rake = self._section_geometry_tensor(
                section_geometry,
                "rake",
            )

        self.n_source_times = int(self.source_times.shape[0])
        self.n_sections = int(self.section_radius.shape[0])
        self._compute()

    def _section_geometry_tensor(
        self,
        section_geometry: Mapping[str, np.ndarray | torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        """Return a custom section array on the propeller tensor backend."""
        return torch.as_tensor(
            section_geometry[name],
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

    def _compute(self) -> None:
        """Compute rotations, section positions, and their time derivatives."""
        self.blade_angles_rad = (
            self.omega * self.source_times[:, None]
            + self.blade_phase_offsets_rad[None, :]
        )

        cos_angle = torch.cos(self.blade_angles_rad)
        sin_angle = torch.sin(self.blade_angles_rad)
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
        self.blade_to_global_rotation_matrix = (
            self.global_to_blade_rotation_matrix.transpose(-2, -1).contiguous()
        )

        cos_twist = torch.cos(self.section_twist_rad)
        sin_twist = torch.sin(self.section_twist_rad)
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
        self.airfoil_to_blade_rotation_matrix = (
            self.blade_to_airfoil_rotation_matrix.transpose(-2, -1).contiguous()
        )

        airfoil_origin_blade_frame = torch.stack(
            [
                self.section_rake,
                self.section_radius,
                -self.section_sweep,
            ],
            dim=-1,
        )
        section_position_airfoil_frame = torch.stack(
            [
                0.75 * self.section_chord,
                torch.zeros_like(self.section_radius),
                torch.zeros_like(self.section_radius),
            ],
            dim=-1,
        )
        section_position_blade_frame = (
            torch.einsum(
                "sij,sj->si",
                self.airfoil_to_blade_rotation_matrix,
                section_position_airfoil_frame,
            )
            + airfoil_origin_blade_frame
        )
        self.section_position_global_frame = torch.einsum(
            "tbij,sj->tbsi",
            self.blade_to_global_rotation_matrix,
            section_position_blade_frame,
        ).contiguous()

        self.omega_vec = torch.zeros(
            (1, 1, 1, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.omega_vec[..., 0] = self.omega
        self.section_velocity_global_frame = torch.linalg.cross(
            self.omega_vec,
            self.section_position_global_frame,
            dim=-1,
        ).contiguous()
        self.section_acceleration_global_frame = torch.linalg.cross(
            self.omega_vec,
            self.section_velocity_global_frame,
            dim=-1,
        ).contiguous()
        self.section_jerk_global_frame = torch.linalg.cross(
            self.omega_vec,
            self.section_acceleration_global_frame,
            dim=-1,
        ).contiguous()
