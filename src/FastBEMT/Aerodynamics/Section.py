'''Blade Element Momentum Theory (BEMT) section solver.

Provides aerodynamic analysis for propeller blade sections with:
- Pre-built aerosandbox.Airfoil objects per section
- Cached airfoil coefficient tables for performance
- Prandtl tip and hub loss corrections
'''

import numpy as np
import scipy.optimize
import aerosandbox as asb
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from ..Utils.JobParameters import LowFidelityParameters

warnings.filterwarnings("ignore", category=RuntimeWarning)

def compute_com(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
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


@dataclass
class RootResult:
    '''Fallback root finding result with root attribute.
    
    Attributes:
        root: Solution value (radians).
    '''
    root: float

class SectionForces:
    '''BEMT solver for a single propeller blade radial section.

    Solves momentum and blade element equations iteratively to determine
    inflow angle, forces, and aerodynamic coefficients at one radial station.
    '''

    def __init__(
        self,
        airfoil: asb.Airfoil,
        r: float,
        dr: float,
        chord: float,
        theta: float,
        params: LowFidelityParameters,
        prop_radius: float,
        hub_radius: float,
        n_blades: int,
    ) -> None:
        '''Initialize blade section geometry and flow parameters.

        Args:
            airfoil: AeroSandbox Airfoil object for coefficient lookups.
            r: Radial position from hub center (m).
            dr: Radial element width (m).
            chord: Section chord length (m).
            theta: Geometric twist angle (radians).
            params: Simulation parameters.
            prop_radius: Propeller tip radius (m).
            hub_radius: Propeller hub radius (m).
            n_blades: Number of blades.
        '''
        self.airfoil = airfoil
        self.r = r
        self.dr = dr
        self.chord = chord
        self.theta = theta
        self.params = params
        self.prop_radius = prop_radius
        self.hub_radius = hub_radius
        self.n_blades = n_blades
        self._tables: dict = {}
        self.v_inf: float | None = None
        self.re: float | None = None
        self.ma: float | None = None
        self._build_prandtl_loss_table()

    @property
    def sigma(self) -> float:
        '''Local solidity ratio.

        Returns:
            Blade area ratio at this radial station (dimensionless).
        '''
        return self.n_blades * self.chord / (2 * np.pi * self.r)

    def _build_prandtl_loss_table(self) -> None:
        '''Precompute Prandtl tip and hub loss factors.

        Tabulates loss factor vs inflow angle for fast interpolation during
        iterative BEMT solution, avoiding repeated computation.
        '''
        self._phi_grid = np.linspace(np.radians(-89.9), np.radians(89.9), 401)

        sin_phi = np.sin(self._phi_grid)
        n_blades = self.n_blades
        r = self.r

        f_tip = n_blades * (self.prop_radius - r) / (2 * r * np.abs(sin_phi))
        f_hub = n_blades * (r - self.hub_radius) / (2 * r * np.abs(sin_phi))

        # Clip to avoid overflow in exponential
        f_tip = np.clip(f_tip, 0.0, 500.0)
        f_hub = np.clip(f_hub, 0.0, 500.0)

        # Compute tip and hub loss factors
        f_tip_loss = 2 * np.arccos(np.exp(-f_tip)) / np.pi
        f_hub_loss = 2 * np.arccos(np.exp(-f_hub)) / np.pi

        self._f_grid = f_tip_loss * f_hub_loss

    def prandtl_loss(self, phi: float) -> float:
        '''Get Prandtl combined tip-hub loss factor.

        Args:
            phi: Inflow angle (radians).

        Returns:
            Loss factor from 0 to 1, where 1 indicates no loss.
        '''
        return np.interp(phi, self._phi_grid, self._f_grid)

    def airfoil_coefficients(
        self,
        alpha: float,
        re: float,
        ma: float,
        model_size: str = 'xxxlarge',
    ) -> tuple[float, float]:
        '''Compute lift and drag coefficients using NeuralFoil.

        Caches coefficient tables binned by Reynolds and Mach numbers for
        fast interpolation. Also computes boundary layer displacement thickness.

        Args:
            alpha: Angle of attack (degrees).
            re: Reynolds number (dimensionless).
            ma: Mach number (dimensionless).
            model_size: NeuralFoil model size, default 'xxxlarge'.

        Returns:
            Tuple of (Cl, Cd).
        '''
        # Bin Reynolds and Mach to coarse grid for table reuse
        re_bin = round(re, -4)
        ma_bin = round(ma, 0)
        key = (re_bin, ma_bin)

        # Lazily build lookup table for this (Re_bin, Ma_bin)
        if key not in self._tables:
            # Alpha grid in degrees for tabulation
            alpha_grid = np.linspace(-20.0, 40.0, 121)
            full_output = asb.Airfoil.get_aero_from_neuralfoil(
                self.airfoil,
                alpha=alpha_grid,
                Re=re_bin,
                mach=ma_bin,
                model_size=model_size,
            )
            c_l_grid = np.asarray(full_output["CL"], dtype=float)
            c_d_grid = np.asarray(full_output["CD"], dtype=float)
            H_u_te_grid = np.asarray(full_output["upper_bl_H_31"], dtype=float)
            H_l_te_grid = np.asarray(full_output["lower_bl_H_31"], dtype=float)
            theta_u_te_grid = np.asarray(full_output["upper_bl_theta_31"], dtype=float)
            theta_l_te_grid = np.asarray(full_output["lower_bl_theta_31"], dtype=float)
            self._tables[key] = (
                alpha_grid,
                c_l_grid,
                c_d_grid,
                H_u_te_grid,
                H_l_te_grid,
                theta_u_te_grid,
                theta_l_te_grid,
            )

        (
            alpha_grid,
            c_l_grid,
            c_d_grid,
            H_u_te_grid,
            H_l_te_grid,
            theta_u_te_grid,
            theta_l_te_grid,
        ) = self._tables[key]
        # Linear interpolation in alpha
        c_l = np.interp(alpha, alpha_grid, c_l_grid)
        c_d = np.interp(alpha, alpha_grid, c_d_grid)
        H_u_te = np.interp(alpha, alpha_grid, H_u_te_grid)
        H_l_te = np.interp(alpha, alpha_grid, H_l_te_grid)
        theta_u_te = np.interp(alpha, alpha_grid, theta_u_te_grid)
        theta_l_te = np.interp(alpha, alpha_grid, theta_l_te_grid)

        self.delta_star_upper = H_u_te * theta_u_te * self.chord
        self.delta_star_lower = H_l_te * theta_l_te * self.chord

        # Fallback: query directly if interpolation fails
        if np.isnan(c_l) or np.isnan(c_d):
            full_output = asb.Airfoil.get_aero_from_neuralfoil(
                self.airfoil,
                alpha=alpha,
                Re=re,
                mach=ma,
                model_size=model_size,
            )
            c_l = full_output["CL"].item()
            c_d = full_output["CD"].item()

        return c_l, c_d

    def section_parameters(self, phi: float) -> tuple:
        '''Compute section aerodynamic state for given inflow angle.

        Args:
            phi: Inflow angle (radians).

        Returns:
            Tuple of (alpha, c_l, c_d, loss_factor, u, a_prime, w, c_l_prime,
            c_d_prime, v_a, v_t) containing angle of attack (deg), coefficients,
            loss factor, induced velocity (m/s), tangential induction factor,
            relative velocity (m/s), rotated coefficients, and velocity components.
        '''
        # Compute angle of attack
        alpha = np.degrees(self.theta - phi)

        # Get aerodynamic coefficients from airfoil lookup
        c_l, c_d = self.airfoil_coefficients(alpha, self.re, self.ma)

        # Rotate coefficients to inflow frame
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        c_l_prime = c_l * cos_phi - c_d * sin_phi
        c_d_prime = c_l * sin_phi + c_d * cos_phi

        # Compute loss factor and momentum parameters
        loss_factor = self.prandtl_loss(phi)
        k_t = self.sigma * c_l_prime / (4 * loss_factor * sin_phi * cos_phi)
        k_q = self.sigma * c_d_prime / (4 * loss_factor * sin_phi * cos_phi)

        # Compute velocity components
        u = self.params.omega * self.r * k_t / (1 + k_q)
        a_prime = k_q / (1 + k_q)
        v_a = self.v_inf + u
        v_t = self.params.omega * self.r * (1 - a_prime)
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
        '''Momentum equation residual for iterative solution.

        Args:
            phi: Inflow angle (radians).

        Returns:
            Residual value, zero at solution.
        '''
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            v_a,
            v_t,
        ) = self.section_parameters(phi)
        return np.tan(phi) - v_a / v_t

    def solve(
        self,
        v_inf: float,
        prev_phi: Optional[float] = None,
    ) -> tuple:
        '''Solve BEMT equations for this section.

        Iteratively finds inflow angle satisfying momentum balance. Uses previous
        solution for faster convergence when available.

        Args:
            v_inf: Freestream velocity (m/s).
            prev_phi: Previous inflow angle for warm start (radians).

        Returns:
            Tuple of (phi, d_t, d_q, alpha, u, a_prime, c_l, c_d, loss_factor,
            w, reynolds, mach, delta_star_upper, delta_star_lower) where:
            phi: inflow angle (radians)
            d_t: thrust contribution (N)
            d_q: torque contribution (N·m)
            alpha: angle of attack (deg)
            u: induced velocity (m/s)
            a_prime: tangential induction factor
            c_l, c_d: rotated coefficients
            loss_factor: Prandtl factor
            w: relative velocity (m/s)
            reynolds, mach: flow conditions
            delta_star_upper, delta_star_lower: displacement thickness (m)
        '''
        self.v_inf = v_inf
        if self.re is None or self.ma is None:
            v_local = np.sqrt(self.v_inf**2 + (self.params.omega * self.r) ** 2)
            self.ma = v_local / self.params.a_inf
            self.re = self.params.rho * v_local * self.chord / self.params.mu

        # Define bounds for inflow angle based on residual sign at phi=0.
        residual_at_zero = self.residual_function(1e-6)
        if residual_at_zero > 0:
            phi_min_default = np.radians(-89.9)
            phi_max_default = np.radians(-0.1)
        else:
            phi_min_default = np.radians(0.1)
            phi_max_default = np.radians(89.9)

        if prev_phi is None:
            bracket = [phi_min_default, phi_max_default]
        else:
            # Use previous phi to bracket search more tightly
            center = np.clip(prev_phi, phi_min_default, phi_max_default)
            delta = np.radians(1)  # initial half-width: ±1 degree

            def safe_residual(x: float) -> float:
                """Safely evaluate residual, returning NaN on exception."""
                try:
                    return self.residual_function(x)
                except Exception:
                    return np.nan

            # Expand bracket until sign change found or bounds reached
            while True:
                lower = max(phi_min_default, center - delta)
                upper = min(phi_max_default, center + delta)

                f_lower = safe_residual(lower)
                f_upper = safe_residual(upper)

                if (
                    np.isfinite(f_lower)
                    and np.isfinite(f_upper)
                    and f_lower * f_upper < 0
                ):
                    bracket = [lower, upper]
                    break

                if lower <= phi_min_default or upper >= phi_max_default:
                    bracket = [phi_min_default, phi_max_default]
                    break

                delta *= 2.0
                

        # Root finding for inflow angle
        try:
            result = scipy.optimize.root_scalar(
                self.residual_function,
                method="brentq",
                xtol=1e-4,
                bracket=bracket,
            )
        except ValueError:
            # print('used newton')
            # Fallback to Newton's method if brentq fails
            result = scipy.optimize.root_scalar(
                self.residual_function,
                method="newton",
                xtol=1e-4,
                x0=np.mean(bracket),
            )
        if not result.converged:
            # Plot residual vs phi to aid debugging.
            # import matplotlib.pyplot as plt

            # phi_grid = np.linspace(phi_min_default, phi_max_default, 200)
            # residuals = []
            # for phi_val in phi_grid:
            #     try:
            #         residuals.append(self.residual_function(phi_val))
            #     except Exception:
            #         residuals.append(np.nan)

            # plt.figure(figsize=(8, 4))
            # plt.plot(np.degrees(phi_grid), residuals, label="Residual")
            # plt.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
            # plt.xlabel("Phi [deg]")
            # plt.ylabel("Residual")
            # plt.title("BEMT Residual vs Phi (Non-Converged)")
            # plt.grid(True, alpha=0.3)
            # plt.tight_layout()
            # plt.show()
            warnings.warn(f"Root finding did not converge at r = {self.r}")
            result = RootResult(
                root=prev_phi,
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

        # Update Reynolds and Mach numbers
        self.re = self.params.rho * w * self.chord / self.params.mu
        self.ma = w / self.params.a_inf

        # Compute local thrust and torque
        d_t = self.sigma * np.pi * self.params.rho * w**2 * c_l_prime * self.r * self.dr
        d_q = (
            self.sigma
            * np.pi
            * self.params.rho
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
