from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .Propeller import Propeller


class BladeStressCalculator:
    """Compute centrifugal and bending stresses along a propeller blade.

    Uses geometry and solution data from a `Propeller` instance.
    """

    def __init__(self, propeller: Propeller) -> None:
        self.propeller = propeller
        self.geometry = propeller.geometry

        # Reuse areas computed in Propeller when available.
        if "cross_section" not in self.geometry:
            self.propeller.section_areas()

    def compute_centrifugal_stress(self, rho: float, omega: float) -> np.ndarray:
        """Compute centrifugal stress at each radial section.

        Args:
            rho: Material density (kg/m^3).
            omega: Angular velocity (rad/s).

        Returns:
            Array of centrifugal stress values (Pa) per section.
        """
        r = self.geometry["r"]
        cross_section = self.geometry["cross_section"]
        stresses = [0.0] * len(cross_section)

        for i in reversed(range(len(self.geometry["r"]) - 1)):
            r1 = r[i]
            r2 = r[i + 1]
            dr = r2 - r1
            r_mean = (r1 + r2) / 2

            a1 = cross_section[i]
            a2 = cross_section[i + 1]

            fc = rho * omega**2 * r_mean * a2 * dr
            stresses[i] = fc / a1 + stresses[i + 1]

        return np.array(stresses)

    def compute_bending_stress(
        self,
        d_t_list: Iterable[float],
        d_q_list: Iterable[float],
    ) -> np.ndarray:
        """Compute bending stress at each radial section.

        Args:
            d_t_list: Sectional thrust distribution (N).
            d_q_list: Sectional torque distribution (N m).

        Returns:
            Array of bending stress values (Pa) per section.
        """
        r = np.asarray(self.geometry["r"])
        d_t_list = np.asarray(d_t_list)
        d_q_list = np.asarray(d_q_list)
        n_sections = len(self.geometry["airfoil"])

        d_d_list = np.divide(d_q_list, r, out=np.zeros_like(d_q_list), where=r != 0)

        bending_moments_x = np.zeros(n_sections)
        bending_moments_z = np.zeros(n_sections)

        for i in range(n_sections - 1):
            moment_arms = r[i + 1 :] - r[i]
            bending_moments_x[i] = np.sum(d_t_list[i + 1 :] * moment_arms)
            bending_moments_z[i] = np.sum(d_d_list[i + 1 :] * moment_arms)

        stresses = []

        for i in range(n_sections):
            airfoil = self.geometry["airfoil"][i]
            chord = self.geometry["chord"][i]

            x_coords = airfoil[:, 0] * chord
            z_coords = airfoil[:, 1] * chord

            i_xx, i_zz, i_xz = self._compute_moment_of_inertia(x_coords, z_coords)

            m_x = bending_moments_x[i]
            m_z = bending_moments_z[i]

            denominator = i_xx * i_zz - i_xz**2
            sigma_total = (
                -(m_z * i_xx + m_x * i_xz) / denominator * x_coords
                + (m_x * i_zz + m_z * i_xz) / denominator * z_coords
            )
            stresses.append(np.max(np.abs(sigma_total)))

        return np.array(stresses)

    def blade_stress_report(
        self,
        material_rho: float,
        show: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute centrifugal and bending stresses, with optional plot.

        Args:
            material_rho: Material density (kg/m^3).
            show: If True, plot the stress distribution.

        Returns:
            Tuple of (centrifugal_stress, bending_stress).

        Notes:
            Uses `propeller.params.omega` and `propeller.solution_data`.
            Ensure `propeller.run_bemt()` has been called before this method.
        """
        sigma_c = self.compute_centrifugal_stress(material_rho, self.propeller.params.omega)
        sigma_b = self.compute_bending_stress(self.propeller.solution_data["d_t"].values / self.geometry["n_blades"], self.propeller.solution_data["d_q"].values / self.geometry["n_blades"])

        if show:
            # Lazy import to keep plotting optional.
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(self.geometry["r"], sigma_c, label="Centrifugal Stress")
            plt.plot(self.geometry["r"], sigma_b, label="Bending Stress")
            plt.plot(
                self.geometry["r"],
                sigma_c + sigma_b,
                label="Total Stress",
                linestyle="--",
            )
            plt.xlabel("Radius [m]")
            plt.ylabel("Stress [Pa]")
            plt.title("Blade Stresses vs Radius")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return sigma_c, sigma_b

    @staticmethod
    def _compute_moment_of_inertia(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute centroidal second moments of area for a polygon."""
        n = len(x)
        c_x, c_y = BladeStressCalculator._compute_com(x, y)
        x = x - c_x
        y = y - c_y

        i_xx = 0.0
        i_zz = 0.0
        i_xz = 0.0

        for i in range(n):
            x0, y0 = x[i], y[i]
            x1, y1 = x[(i + 1) % n], y[(i + 1) % n]
            common = x0 * y1 - x1 * y0

            i_xx += (y0**2 + y0 * y1 + y1**2) * common
            i_zz += (x0**2 + x0 * x1 + x1**2) * common
            i_xz += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * common

        i_xx *= 1.0 / 12.0
        i_zz *= 1.0 / 12.0
        i_xz *= 1.0 / 24.0

        i_xx = abs(i_xx)
        i_zz = abs(i_zz)

        return i_xx, i_zz, i_xz

    @staticmethod
    def _compute_com(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute polygon centroid (x, y) using the shoelace formula."""
        xi = x[:-1]
        yi = y[:-1]
        xi1 = x[1:]
        yi1 = y[1:]

        a = xi * yi1 - xi1 * yi
        area = 0.5 * np.sum(a)

        c_x = (1.0 / (6.0 * area)) * np.sum((xi + xi1) * a)
        c_y = (1.0 / (6.0 * area)) * np.sum((yi + yi1) * a)
        return c_x, c_y