from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from ..Propeller import Propeller

if TYPE_CHECKING:
    from ..Aerodynamics.BEMT import BEMT


class BladeStressCalculator:
    """Propeller blade stress analysis."""

    def __init__(self, propeller: Propeller) -> None:
        self.propeller = propeller
        self.geometry = propeller.geometry

    def compute_centrifugal_stress(
        self,
        material_density: float,
        angular_velocity: float,
    ) -> np.ndarray:
        """Return centrifugal stress at each section in Pa.

        Args:
            material_density: Blade-material density in kg/m^3.
            angular_velocity: Propeller angular velocity in rad/s.
        """
        r = np.asarray(self.geometry["r"])
        cross_section = np.asarray(self.geometry["cross_section"])

        dr = r[1:] - r[:-1]
        r_mean = 0.5 * (r[1:] + r[:-1])
        area_average = 0.5 * (cross_section[:-1] + cross_section[1:])

        centrifugal_load = (
            material_density
            * angular_velocity**2
            * r_mean
            * area_average
            * dr
        )
        load_per_area = centrifugal_load / cross_section[:-1]

        stresses = np.zeros_like(cross_section)
        stresses[:-1] = np.cumsum(load_per_area[::-1])[::-1]
        return stresses

    def compute_bending_stress(
        self,
        section_thrust: Iterable[float],
        section_torque: Iterable[float],
    ) -> np.ndarray:
        """Return bending stress over each section airfoil polygon in Pa.

        Args:
            section_thrust: Per-blade thrust load at each radial section in N.
            section_torque: Per-blade torque load at each radial section in N m.
        """
        r = np.asarray(self.geometry["r"])
        section_thrust = np.asarray(section_thrust)
        section_torque = np.asarray(section_torque)
        n_sections = len(self.geometry["airfoils"])

        drag_load = np.divide(
            section_torque,
            r,
            out=np.zeros_like(section_torque),
            where=r != 0,
        )
        bending_moments_x = self._outboard_bending_moment(section_thrust, r)
        bending_moments_z = self._outboard_bending_moment(drag_load, r)

        stresses = []
        for section_index in range(n_sections):
            airfoil = self.geometry["airfoils"][section_index]
            chord = self.geometry["chord"][section_index]
            x_coords = airfoil[:, 0] * chord
            z_coords = airfoil[:, 1] * chord
            i_xx, i_zz, i_xz = self._compute_moment_of_inertia(x_coords, z_coords)

            m_x = -bending_moments_x[section_index]
            m_z = bending_moments_z[section_index]
            denominator = i_xx * i_zz - i_xz**2
            stresses.append(
                -(m_z * i_xx + m_x * i_xz) / denominator * x_coords
                + (m_x * i_zz + m_z * i_xz) / denominator * z_coords
            )
        return np.array(stresses)

    def compute_stress(
        self,
        material_density: float,
        bemt: BEMT,
        rpm: float | None = None,
        v_inf: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute centrifugal and bending stress for a BEMT operating point.

        Args:
            material_density: Blade-material density in kg/m^3.
            bemt: Completed aerodynamic analysis for this propeller.
            rpm: Rotational speed to select when ``bemt`` has multiple points.
            v_inf: Freestream speed to select when ``bemt`` has multiple points.

        Returns:
            Centrifugal section stress and bending stress over each airfoil.
        """
        operating_rpm, operating_v_inf = bemt.resolve_operating_point(rpm, v_inf)
        solution = bemt.solution_for(operating_rpm, operating_v_inf)
        angular_velocity = 2.0 * np.pi * operating_rpm / 60.0
        n_blades = self.geometry["n_blades"]
        aerodynamic_sections = self.propeller.aerodynamic_section_mask
        section_thrust = np.where(
            aerodynamic_sections,
            solution["section_thrust"].to_numpy(),
            0.0,
        )
        section_torque = np.where(
            aerodynamic_sections,
            solution["section_torque"].to_numpy(),
            0.0,
        )

        centrifugal_stress = self.compute_centrifugal_stress(
            material_density,
            angular_velocity,
        )
        bending_stress = self.compute_bending_stress(
            section_thrust / n_blades,
            section_torque / n_blades,
        )
        return centrifugal_stress, bending_stress

    @staticmethod
    def _outboard_bending_moment(
        section_load: np.ndarray,
        radius: np.ndarray,
    ) -> np.ndarray:
        """Return each section's bending moment from its outboard loads."""
        cumulative_load = np.cumsum(section_load[::-1])[::-1]
        cumulative_moment = np.cumsum((section_load * radius)[::-1])[::-1]
        outboard_load = np.zeros_like(cumulative_load)
        outboard_moment = np.zeros_like(cumulative_moment)
        outboard_load[:-1] = cumulative_load[1:]
        outboard_moment[:-1] = cumulative_moment[1:]
        return outboard_moment - radius * outboard_load

    @staticmethod
    def _compute_moment_of_inertia(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute polygon second moments of area in m^4."""
        c_x, c_y = BladeStressCalculator._polygon_centroid(x, y)
        x = x - c_x
        y = y - c_y

        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        i_xx = np.sum((y**2 + y * y_next + y_next**2) * cross)
        i_zz = np.sum((x**2 + x * x_next + x_next**2) * cross)
        i_xz = np.sum(
            (x * y_next + 2.0 * x * y + 2.0 * x_next * y_next + x_next * y)
            * cross
        )

        return abs(i_xx / 12.0), abs(i_zz / 12.0), i_xz / 24.0

    @staticmethod
    def _polygon_centroid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Compute a polygon centroid with the shoelace formula."""
        x_current = x[:-1]
        y_current = y[:-1]
        x_next = x[1:]
        y_next = y[1:]
        cross = x_current * y_next - x_next * y_current
        scale = 3.0 * np.sum(cross)
        return (
            float(np.sum((x_current + x_next) * cross) / scale),
            float(np.sum((y_current + y_next) * cross) / scale),
        )

    def compute_propeller_mass(self, material_density: float) -> float:
        """Return total propeller mass in kg for a material density in kg/m^3."""
        segment_volumes = (
            np.asarray(self.geometry["cross_section"])
            * np.asarray(self.geometry["dr"])
        )
        return float(
            np.sum(segment_volumes)
            * material_density
            * self.geometry["n_blades"]
        )
