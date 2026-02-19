from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .Propeller import Propeller


class BladeStressCalculator:
    '''Propeller blade stress analysis.

    Computes centrifugal and bending stresses along blade span using
    geometry and BEMT solution data from Propeller instance.
    '''

    def __init__(self, propeller: Propeller) -> None:
        self.propeller = propeller
        self.geometry = propeller.geometry

    def compute_centrifugal_stress(self, rho: float, omega: float) -> np.ndarray:
        '''Compute centrifugal stress distribution.

        Args:
            rho: Material density (kg/m³).
            omega: Angular velocity (rad/s).

        Returns:
            Centrifugal stress at each section (Pa), shape (n_sections,).
        '''
        r = np.asarray(self.geometry["r"])
        cross_section = np.asarray(self.geometry["cross_section"])

        dr = r[1:] - r[:-1]
        r_mean = 0.5 * (r[1:] + r[:-1])
        a1 = cross_section[:-1]
        a2 = cross_section[1:]
        a_avg = 0.5 * (a1 + a2)

        fc = rho * omega**2 * r_mean * a_avg * dr
        load_per_area = np.divide(fc, a1, out=np.zeros_like(fc), where=a1 != 0)

        stresses = np.zeros_like(cross_section)
        stresses[:-1] = np.cumsum(load_per_area[::-1])[::-1]

        return stresses

    def compute_bending_stress(
        self,
        d_t_list: Iterable[float],
        d_q_list: Iterable[float],
    ) -> np.ndarray:
        '''Compute bending stress distribution.

        Args:
            d_t_list: Thrust per section (N), shape (n_sections,).
            d_q_list: Torque per section (N·m), shape (n_sections,).

        Returns:
            Bending stress field at each section (Pa), shape (n_sections, n_airfoil_points).
        '''
        r = np.asarray(self.geometry["r"])
        d_t_list = np.asarray(d_t_list)
        d_q_list = np.asarray(d_q_list)
        n_sections = len(self.geometry["airfoil"])

        d_d_list = np.divide(d_q_list, r, out=np.zeros_like(d_q_list), where=r != 0)

        sum_t = np.cumsum(d_t_list[::-1])[::-1]
        sum_tr = np.cumsum((d_t_list * r)[::-1])[::-1]
        sum_d = np.cumsum(d_d_list[::-1])[::-1]
        sum_dr = np.cumsum((d_d_list * r)[::-1])[::-1]

        sum_t_next = np.zeros_like(sum_t)
        sum_tr_next = np.zeros_like(sum_tr)
        sum_d_next = np.zeros_like(sum_d)
        sum_dr_next = np.zeros_like(sum_dr)

        sum_t_next[:-1] = sum_t[1:]
        sum_tr_next[:-1] = sum_tr[1:]
        sum_d_next[:-1] = sum_d[1:]
        sum_dr_next[:-1] = sum_dr[1:]

        bending_moments_x = sum_tr_next - r * sum_t_next
        bending_moments_z = sum_dr_next - r * sum_d_next

        stresses = []

        for i in range(n_sections):
            airfoil = self.geometry["airfoil"][i]
            chord = self.geometry["chord"][i]

            x_coords = airfoil[:, 0] * chord
            z_coords = airfoil[:, 1] * chord

            i_xx, i_zz, i_xz = self._compute_moment_of_inertia(x_coords, z_coords)

            m_x = -bending_moments_x[i]
            m_z = bending_moments_z[i]

            denominator = i_xx * i_zz - i_xz**2
            sigma_total = (
                -(m_z * i_xx + m_x * i_xz) / denominator * x_coords
                + (m_x * i_zz + m_z * i_xz) / denominator * z_coords
            )
            # stresses.append(np.max(np.abs(sigma_total)))
            stresses.append(sigma_total)
        return np.array(stresses)

    def blade_stress_report(
        self,
        material_rho: float,
        show: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Compute and optionally plot stress distributions.

        Args:
            material_rho: Material density (kg/m³).
            show: Plot stress distributions if True.

        Returns:
            Tuple of (sigma_c, sigma_b) where:
            sigma_c: centrifugal stress (Pa), shape (n_sections,)
            sigma_b: bending stress (Pa), shape (n_sections, n_airfoil_points)
            
        Note:
            Requires propeller.run_bemt() to have been called first.
        '''
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
        '''Compute second moments of area for polygon.
        
        Args:
            x: X coordinates, shape (n_points,).
            y: Y coordinates, shape (n_points,).
            
        Returns:
            Tuple of (I_xx, I_zz, I_xz) in m^4.
        '''
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
        '''Compute polygon centroid using Shoelace formula.
        
        Args:
            x: X coordinates, shape (n_points,).
            y: Y coordinates, shape (n_points,).
            
        Returns:
            Tuple of (x_centroid, y_centroid).
        '''
        xi = x[:-1]
        yi = y[:-1]
        xi1 = x[1:]
        yi1 = y[1:]

        a = xi * yi1 - xi1 * yi
        area = 0.5 * np.sum(a)

        c_x = (1.0 / (6.0 * area)) * np.sum((xi + xi1) * a)
        c_y = (1.0 / (6.0 * area)) * np.sum((yi + yi1) * a)
        return c_x, c_y