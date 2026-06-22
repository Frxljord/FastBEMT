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

    def compute_centrifugal_stress(self, rho: float, omega: float) -> np.ndarray:
        """Return centrifugal stress at each section in Pa."""
        r = np.asarray(self.geometry["r"])
        cross_section = np.asarray(self.geometry["cross_section"])

        dr = r[1:] - r[:-1]
        r_mean = 0.5 * (r[1:] + r[:-1])
        area_average = 0.5 * (cross_section[:-1] + cross_section[1:])

        centrifugal_load = rho * omega**2 * r_mean * area_average * dr
        load_per_area = np.divide(
            centrifugal_load,
            cross_section[:-1],
            out=np.zeros_like(centrifugal_load),
            where=cross_section[:-1] != 0,
        )

        stresses = np.zeros_like(cross_section)
        stresses[:-1] = np.cumsum(load_per_area[::-1])[::-1]
        return stresses

    def compute_bending_stress(
        self,
        d_t_list: Iterable[float],
        d_q_list: Iterable[float],
    ) -> np.ndarray:
        """Return bending stress over each section airfoil polygon in Pa."""
        r = np.asarray(self.geometry["r"])
        d_t = np.asarray(d_t_list)
        d_q = np.asarray(d_q_list)
        n_sections = len(self.geometry["airfoils"])

        drag_like_load = np.divide(d_q, r, out=np.zeros_like(d_q), where=r != 0)
        thrust_cumulative = np.cumsum(d_t[::-1])[::-1]
        thrust_moment_cumulative = np.cumsum((d_t * r)[::-1])[::-1]
        drag_cumulative = np.cumsum(drag_like_load[::-1])[::-1]
        drag_moment_cumulative = np.cumsum((drag_like_load * r)[::-1])[::-1]

        thrust_next = np.zeros_like(thrust_cumulative)
        thrust_moment_next = np.zeros_like(thrust_moment_cumulative)
        drag_next = np.zeros_like(drag_cumulative)
        drag_moment_next = np.zeros_like(drag_moment_cumulative)
        thrust_next[:-1] = thrust_cumulative[1:]
        thrust_moment_next[:-1] = thrust_moment_cumulative[1:]
        drag_next[:-1] = drag_cumulative[1:]
        drag_moment_next[:-1] = drag_moment_cumulative[1:]

        bending_moments_x = thrust_moment_next - r * thrust_next
        bending_moments_z = drag_moment_next - r * drag_next

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

    def blade_stress_report(
        self,
        material_rho: float,
        bemt: BEMT,
        show: bool = False,
        rpm: float | None = None,
        v_inf: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute centrifugal and bending stress for a BEMT operating point."""
        if bemt.propeller is not self.propeller:
            raise ValueError("The BEMT analysis belongs to a different propeller.")

        operating_rpm, operating_v_inf = bemt.resolve_operating_point(rpm, v_inf)
        solution = bemt.solution_for(operating_rpm, operating_v_inf)
        omega = 2.0 * np.pi * operating_rpm / 60.0
        n_blades = self.geometry["n_blades"]

        sigma_c = self.compute_centrifugal_stress(material_rho, omega)
        sigma_b = self.compute_bending_stress(
            solution["d_t"].values / n_blades,
            solution["d_q"].values / n_blades,
        )

        if show:
            self._plot_stress_report(sigma_c, sigma_b)

        return sigma_c, sigma_b

    def _plot_stress_report(self, sigma_c: np.ndarray, sigma_b: np.ndarray) -> None:
        """Plot spanwise stress envelopes."""
        import matplotlib.pyplot as plt

        radius = self.geometry["r"]
        bending_max = sigma_b.max(axis=1)
        plt.figure(figsize=(8, 5))
        plt.plot(radius, sigma_c / 1e6, label="Centrifugal")
        plt.plot(radius, bending_max / 1e6, label="Bending")
        plt.plot(
            radius,
            (sigma_c + bending_max) / 1e6,
            label="Total",
            linestyle="--",
        )
        plt.xlabel("Radius [m]")
        plt.ylabel("Stress [MPa]")
        plt.title("Blade Stress Envelopes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _compute_moment_of_inertia(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute polygon second moments of area in m^4."""
        c_x, c_y = BladeStressCalculator._compute_com(x, y)
        x = x - c_x
        y = y - c_y

        i_xx = 0.0
        i_zz = 0.0
        i_xz = 0.0
        for index in range(len(x)):
            x0, y0 = x[index], y[index]
            x1, y1 = x[(index + 1) % len(x)], y[(index + 1) % len(x)]
            common = x0 * y1 - x1 * y0
            i_xx += (y0**2 + y0 * y1 + y1**2) * common
            i_zz += (x0**2 + x0 * x1 + x1**2) * common
            i_xz += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * common

        return abs(i_xx / 12.0), abs(i_zz / 12.0), i_xz / 24.0

    @staticmethod
    def _compute_com(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Compute polygon centroid with the shoelace formula."""
        xi = x[:-1]
        yi = y[:-1]
        xi1 = x[1:]
        yi1 = y[1:]

        area_terms = xi * yi1 - xi1 * yi
        area = 0.5 * np.sum(area_terms)
        c_x = np.sum((xi + xi1) * area_terms) / (6.0 * area)
        c_y = np.sum((yi + yi1) * area_terms) / (6.0 * area)
        return c_x, c_y

    def compute_propeller_mass(self, rho: float) -> float:
        """Return total propeller mass in kg."""
        segment_volumes = (
            np.asarray(self.geometry["cross_section"])
            * np.asarray(self.geometry["dr"])
        )
        return float(np.sum(segment_volumes) * rho * self.geometry["n_blades"])
