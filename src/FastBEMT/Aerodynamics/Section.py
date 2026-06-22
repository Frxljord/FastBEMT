"""Blade-element momentum section solver."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import aerosandbox as asb
import numpy as np
import scipy.optimize

from ..Utils.Environment import Environment


@dataclass
class RootResult:
    """Fallback root finding result compatible with scipy's root object."""

    root: float


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
        prop_radius: float,
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
        self.prop_radius = prop_radius
        self.hub_radius = hub_radius
        self.n_blades = n_blades
        self._tables: dict[tuple[float, float], tuple[np.ndarray, ...]] = {}
        self.v_inf: float | None = None
        self.re: float | None = None
        self.ma: float | None = None
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
        f_tip = self.n_blades * (self.prop_radius - self.r) / denominator
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
        re: float,
        ma: float,
        model_size: str = "xxxlarge",
    ) -> tuple[float, float]:
        """Return lift and drag coefficients, using cached NeuralFoil tables."""
        re_bin = round(re, -4)
        ma_bin = round(ma, 0)
        key = (re_bin, ma_bin)

        if key not in self._tables:
            alpha_grid = np.linspace(-20.0, 40.0, 121)
            output = asb.Airfoil.get_aero_from_neuralfoil(
                self.airfoil,
                alpha=alpha_grid,
                Re=re_bin,
                mach=ma_bin,
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
                Re=re,
                mach=ma,
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
        c_l_prime = c_l * cos_phi - c_d * sin_phi
        c_d_prime = c_l * sin_phi + c_d * cos_phi
        loss_factor = self.prandtl_loss(phi)
        k_t = self.sigma * c_l_prime / (4.0 * loss_factor * sin_phi * cos_phi)
        k_q = self.sigma * c_d_prime / (4.0 * loss_factor * sin_phi * cos_phi)

        u = self.omega * self.r * k_t / (1.0 + k_q)
        a_prime = k_q / (1.0 + k_q)
        v_a = self.v_inf + u
        v_t = self.omega * self.r * (1.0 - a_prime)
        w = np.sqrt(v_a**2 + v_t**2)
        return (
            alpha,
            c_l,
            c_d,
            loss_factor,
            u,
            a_prime,
            w,
            c_l_prime,
            c_d_prime,
            v_a,
            v_t,
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
        if self.re is None or self.ma is None:
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
        if not result.converged:
            warnings.warn(
                f"Root finding did not converge at r = {self.r}",
                RuntimeWarning,
                stacklevel=2,
            )
            result = RootResult(
                root=prev_phi if prev_phi is not None else float(np.mean(bracket)),
            )

        phi = result.root
        (
            alpha,
            _,
            _,
            loss_factor,
            u,
            a_prime,
            w,
            c_l_prime,
            c_d_prime,
            _,
            _,
        ) = self.section_parameters(phi)

        self.re = self.environment.rho * w * self.chord / self.environment.mu
        self.ma = w / self.environment.a_inf
        d_t = (
            self.sigma
            * np.pi
            * self.environment.rho
            * w**2
            * c_l_prime
            * self.r
            * self.dr
        )
        d_q = (
            self.sigma
            * np.pi
            * self.environment.rho
            * w**2
            * c_d_prime
            * self.r**2
            * self.dr
        )

        return (
            phi,
            d_t,
            d_q,
            alpha,
            u,
            a_prime,
            c_l_prime,
            c_d_prime,
            loss_factor,
            w,
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
