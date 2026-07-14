"""Blade-element momentum section solver."""

from __future__ import annotations

import warnings

import aerosandbox as asb
import numpy as np
import scipy.optimize

from ..Utils.Environment import Environment


class SectionForces:
    """Solve BEMT forces for one radial blade section."""

    def __init__(
        self,
        airfoil: asb.Airfoil,
        r: float,
        dr: float,
        chord: float,
        theta: float,
        environment: Environment,
        omega: float,
        tip_radius: float,
        hub_radius: float,
        n_blades: int,
    ) -> None:
        self.airfoil = airfoil
        self.r = r
        self.dr = dr
        self.chord = chord
        self.theta = theta
        self.environment = environment
        self.omega = omega
        self.tip_radius = tip_radius
        self.hub_radius = hub_radius
        self.n_blades = n_blades
        self._tables: dict[tuple[float, float], tuple[np.ndarray, ...]] = {}
        self.v_inf = 0.0
        self.re = np.nan
        self.ma = np.nan
        self.delta_star_upper = np.nan
        self.delta_star_lower = np.nan
        self._build_prandtl_loss_table()

    @property
    def sigma(self) -> float:
        """Local solidity ratio."""
        return self.n_blades * self.chord / (2.0 * np.pi * self.r)

    def _build_prandtl_loss_table(self) -> None:
        """Precompute combined tip and hub Prandtl loss factors."""
        self._phi_grid = np.linspace(np.radians(-89.9), np.radians(89.9), 401)
        sin_phi = np.maximum(np.abs(np.sin(self._phi_grid)), np.finfo(float).eps)
        denominator = 2.0 * self.r * sin_phi
        f_tip = self.n_blades * (self.tip_radius - self.r) / denominator
        f_hub = self.n_blades * (self.r - self.hub_radius) / denominator
        self._f_grid = (
            2.0
            * np.arccos(np.exp(-np.clip(f_tip, 0.0, 500.0)))
            / np.pi
            * 2.0
            * np.arccos(np.exp(-np.clip(f_hub, 0.0, 500.0)))
            / np.pi
        )

    def prandtl_loss(self, phi: float) -> float:
        """Interpolate the combined tip-hub Prandtl loss factor."""
        return float(np.interp(phi, self._phi_grid, self._f_grid))

    def airfoil_coefficients(
        self,
        alpha: float,
        reynolds_number: float,
        mach_number: float,
        model_size: str = "xxxlarge",
    ) -> tuple[float, float]:
        """Return lift and drag coefficients, using cached NeuralFoil tables."""
        reynolds_bin = round(reynolds_number, -4)
        mach_bin = round(mach_number, 0)
        key = (reynolds_bin, mach_bin)

        if key not in self._tables:
            alpha_grid = np.linspace(-20.0, 40.0, 121)
            output = asb.Airfoil.get_aero_from_neuralfoil(
                self.airfoil,
                alpha=alpha_grid,
                Re=reynolds_bin,
                mach=mach_bin,
                model_size=model_size,
            )
            self._tables[key] = (
                alpha_grid,
                np.asarray(output["CL"], dtype=float),
                np.asarray(output["CD"], dtype=float),
                np.asarray(output["upper_bl_H_31"], dtype=float),
                np.asarray(output["lower_bl_H_31"], dtype=float),
                np.asarray(output["upper_bl_theta_31"], dtype=float),
                np.asarray(output["lower_bl_theta_31"], dtype=float),
            )

        (
            alpha_grid,
            c_l_grid,
            c_d_grid,
            h_upper_grid,
            h_lower_grid,
            theta_upper_grid,
            theta_lower_grid,
        ) = self._tables[key]
        c_l = float(np.interp(alpha, alpha_grid, c_l_grid))
        c_d = float(np.interp(alpha, alpha_grid, c_d_grid))
        self.delta_star_upper = float(
            np.interp(alpha, alpha_grid, h_upper_grid)
            * np.interp(alpha, alpha_grid, theta_upper_grid)
            * self.chord
        )
        self.delta_star_lower = float(
            np.interp(alpha, alpha_grid, h_lower_grid)
            * np.interp(alpha, alpha_grid, theta_lower_grid)
            * self.chord
        )

        if np.isnan(c_l) or np.isnan(c_d):
            output = asb.Airfoil.get_aero_from_neuralfoil(
                self.airfoil,
                alpha=alpha,
                Re=reynolds_number,
                mach=mach_number,
                model_size=model_size,
            )
            c_l = float(output["CL"].item())
            c_d = float(output["CD"].item())
        return c_l, c_d

    def section_parameters(self, phi: float) -> tuple[float, ...]:
        """Return aerodynamic state for an inflow angle."""
        alpha = np.degrees(self.theta - phi)
        c_l, c_d = self.airfoil_coefficients(alpha, self.re, self.ma)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        normal_force_coefficient = c_l * cos_phi - c_d * sin_phi
        tangential_force_coefficient = c_l * sin_phi + c_d * cos_phi
        prandtl_loss_factor = self.prandtl_loss(phi)
        axial_induction_term = self.sigma * normal_force_coefficient / (
            4.0 * prandtl_loss_factor * sin_phi * cos_phi
        )
        tangential_induction_term = self.sigma * tangential_force_coefficient / (
            4.0 * prandtl_loss_factor * sin_phi * cos_phi
        )

        induced_velocity = (
            self.omega
            * self.r
            * axial_induction_term
            / (1.0 + tangential_induction_term)
        )
        tangential_induction_factor = tangential_induction_term / (
            1.0 + tangential_induction_term
        )
        axial_velocity = self.v_inf + induced_velocity
        tangential_velocity = self.omega * self.r * (
            1.0 - tangential_induction_factor
        )
        relative_velocity = np.sqrt(axial_velocity**2 + tangential_velocity**2)
        return (
            alpha,
            c_l,
            c_d,
            prandtl_loss_factor,
            induced_velocity,
            tangential_induction_factor,
            relative_velocity,
            normal_force_coefficient,
            tangential_force_coefficient,
            axial_velocity,
            tangential_velocity,
        )

    def residual_function(self, phi: float) -> float:
        """Momentum equation residual."""
        *_, v_a, v_t = self.section_parameters(phi)
        return float(np.tan(phi) - v_a / v_t)

    def solve(
        self,
        v_inf: float,
        prev_phi: float | None = None,
    ) -> tuple[float, ...]:
        """Solve this section for the current operating point."""
        self.v_inf = v_inf
        local_velocity = np.sqrt(self.v_inf**2 + (self.omega * self.r) ** 2)
        self.ma = local_velocity / self.environment.a_inf
        self.re = (
            self.environment.rho
            * local_velocity
            * self.chord
            / self.environment.mu
        )

        bracket = self._initial_phi_bracket(prev_phi)
        try:
            result = scipy.optimize.root_scalar(
                self.residual_function,
                method="brentq",
                xtol=1e-4,
                bracket=bracket,
            )
        except ValueError:
            result = scipy.optimize.root_scalar(
                self.residual_function,
                method="newton",
                xtol=1e-4,
                x0=np.mean(bracket),
            )
        if result.converged:
            phi = float(result.root)
        else:
            warnings.warn(
                f"Root finding did not converge at r = {self.r}",
                RuntimeWarning,
                stacklevel=2,
            )
            phi = (
                prev_phi if prev_phi is not None else float(np.mean(bracket))
            )
        (
            alpha,
            _,
            _,
            prandtl_loss_factor,
            induced_velocity,
            tangential_induction_factor,
            relative_velocity,
            normal_force_coefficient,
            tangential_force_coefficient,
            _,
            _,
        ) = self.section_parameters(phi)

        self.re = (
            self.environment.rho
            * relative_velocity
            * self.chord
            / self.environment.mu
        )
        self.ma = relative_velocity / self.environment.a_inf
        section_thrust = (
            self.sigma
            * np.pi
            * self.environment.rho
            * relative_velocity**2
            * normal_force_coefficient
            * self.r
            * self.dr
        )
        section_torque = (
            self.sigma
            * np.pi
            * self.environment.rho
            * relative_velocity**2
            * tangential_force_coefficient
            * self.r**2
            * self.dr
        )

        return (
            phi,
            section_thrust,
            section_torque,
            alpha,
            induced_velocity,
            tangential_induction_factor,
            normal_force_coefficient,
            tangential_force_coefficient,
            prandtl_loss_factor,
            relative_velocity,
            self.re,
            self.ma,
            self.delta_star_upper,
            self.delta_star_lower,
        )

    def _initial_phi_bracket(self, prev_phi: float | None) -> list[float]:
        residual_at_zero = self.residual_function(1.0e-6)
        if residual_at_zero > 0.0:
            phi_min = np.radians(-89.9)
            phi_max = np.radians(-0.1)
        else:
            phi_min = np.radians(0.1)
            phi_max = np.radians(89.9)

        if prev_phi is None:
            return [phi_min, phi_max]

        center = np.clip(prev_phi, phi_min, phi_max)
        delta = np.radians(1.0)
        while True:
            lower = max(phi_min, center - delta)
            upper = min(phi_max, center + delta)
            f_lower = self._safe_residual(lower)
            f_upper = self._safe_residual(upper)

            if (
                np.isfinite(f_lower)
                and np.isfinite(f_upper)
                and f_lower * f_upper < 0
            ):
                return [lower, upper]
            if lower <= phi_min or upper >= phi_max:
                return [phi_min, phi_max]
            delta *= 2.0

    def _safe_residual(self, phi: float) -> float:
        try:
            return self.residual_function(phi)
        except Exception:
            return np.nan
