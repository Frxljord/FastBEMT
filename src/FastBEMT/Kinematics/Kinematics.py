"""Rigid-body kinematics for rotating propeller blade sections."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """

    def __init__(self, propeller: Propeller, rpm: float) -> None:
        self.propeller = propeller
        self.rpm = float(rpm)
        self.dtype: torch.dtype = propeller.dtype
        self.device: torch.device = torch.device(propeller.device)

        geometry = propeller.geometry
        self.n_blades = int(geometry["n_blades"])
        propeller.simulation.configure_operating_point(self.rpm, self.n_blades)

        simulation = propeller.simulation
        if (
            simulation.omega is None
            or simulation.src_times is None
            or simulation.blade_angles is None
        ):
            raise RuntimeError("The simulation operating point was not configured.")

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

        self.radial_positions = torch.as_tensor(
            geometry["r"],
            dtype=self.dtype,
            device=self.device,
        )
        self.twist_rad = torch.deg2rad(
            torch.as_tensor(
                geometry["twist"],
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.com_shift_forward = torch.as_tensor(
            propeller.com_shift_forward,
            dtype=self.dtype,
            device=self.device,
        )
        self.com_shift_up = torch.as_tensor(
            propeller.com_shift_up,
            dtype=self.dtype,
            device=self.device,
        )

        self.nt = int(self.source_times.shape[0])
        self.ns = int(self.radial_positions.shape[0])
        self.nb = int(self.blade_phase_offsets.shape[0])
        self._validate_geometry()
        self._compute()

    def _validate_geometry(self) -> None:
        """Validate section-wise geometry dimensions."""
        section_tensors = {
            "r": self.radial_positions,
            "twist": self.twist_rad,
            "com_shift_forward": self.com_shift_forward,
            "com_shift_up": self.com_shift_up,
        }
        for name, value in section_tensors.items():
            if value.ndim != 1 or value.shape[0] != self.ns:
                raise ValueError(
                    f"{name} must be one-dimensional with {self.ns} entries."
                )
        if self.nb != self.n_blades:
            raise ValueError(
                "The configured blade angles do not match geometry['n_blades']."
            )

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
