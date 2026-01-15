"""Fast Blade Element Momentum Theory (BEMT) solver.

This module provides a faster version of the BEMT solver by:
- Reusing pre-built `aerosandbox.Airfoil` objects per section
- Caching / pre-tabulating airfoil coefficients
"""

import numpy as np
import scipy.optimize
import pandas as pd
import aerosandbox as asb
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class SectionForces:
    """Solve BEMT for a single blade section."""
    def __init__(self, airfoil, r, dr, chord, theta, propellerParams):
        """Initialize per-section geometry, flow state, and cached data."""
        self.airfoil = airfoil
        self.r = r
        self.dr = dr
        self.chord = chord
        self.theta = theta
        self.propellerParams = propellerParams
        self._tables = {}
        self._buildPrandtlLossTable()

    @property
    def sigma(self):
        """Local solidity for this section."""
        return self.propellerParams.n_blades * self.chord / (2 * np.pi * self.r)

    def _buildPrandtlLossTable(self):
        """Precompute Prandtl loss factor on a phi grid for fast interpolation."""
        self._phi_grid = np.linspace(np.radians(0.1), np.radians(89.9), 100)

        sin_phi = np.sin(self._phi_grid)
        B = self.propellerParams.n_blades
        r = self.r

        f_tip = B * (self.propellerParams.prop_radius - r) / (2 * r * sin_phi)
        f_hub = B * (r - self.propellerParams.hub_radius) / (2 * r * sin_phi)

        # Clip to avoid overflow
        f_tip = np.clip(f_tip, 0.0, 500.0)
        f_hub = np.clip(f_hub, 0.0, 500.0)

        F_tip = 2 * np.arccos(np.exp(-f_tip)) / np.pi
        F_hub = 2 * np.arccos(np.exp(-f_hub)) / np.pi

        self._F_grid = F_tip * F_hub

    def prandtlLoss(self, phi):
        """Return Prandtl tip-hub loss factor for a given inflow angle phi (rad)."""
        if phi <= 0:
            return 1.0
        return np.interp(phi, self._phi_grid, self._F_grid)

    def airfoilCoefficients(self, alpha, Re, Ma, modelSize="xxxlarge"):
        """Return CL, CD for the given alpha, Re, Ma using a pre-tabulated lookup."""
        # Bin Reynolds and Mach to a coarse grid to maximize table reuse
        ReBin = round(Re, -4)
        MaBin = round(Ma, 1)
        key = (ReBin, MaBin)

        # Lazily build the CL/CD vs alpha table for this (ReBin, MaBin)
        if key not in self._tables:
            # Alpha grid in degrees for tabulation
            alpha_grid = np.linspace(0.0, 30.0, 61)
            full_output = asb.Airfoil.get_aero_from_neuralfoil(self.airfoil, alpha=alpha_grid, Re=ReBin, mach=MaBin, model_size=modelSize)
            cl_grid = np.asarray(full_output["CL"], dtype=float)
            cd_grid = np.asarray(full_output["CD"], dtype=float)
            self._tables[key] = (alpha_grid, cl_grid, cd_grid)

        alpha_grid, cl_grid, cd_grid = self._tables[key]
        # 1D linear interpolation in alpha (degrees)
        cL = np.interp(alpha, alpha_grid, cl_grid)
        cD = np.interp(alpha, alpha_grid, cd_grid)
        if np.isnan(cL) or np.isnan(cD):
            full_output = asb.Airfoil.get_aero_from_neuralfoil(self.airfoil, alpha=alpha, Re=Re, mach=Ma, model_size=modelSize)
            cL, cD = full_output["CL"].item(), full_output["CD"].item()
        return cL, cD

    def sectionParameters(self, phi):
        """Compute aerodynamic section parameters for a given inflow angle phi."""
        alpha = np.degrees(self.theta - phi)
        cL, cD = self.airfoilCoefficients(alpha, self.Re, self.Ma)
        c, s = np.cos(phi), np.sin(phi)
        cLPrime = cL * c - cD * s
        cDPrime = cL * s + cD * c
        F = self.prandtlLoss(phi)
        k_t = self.sigma * cLPrime / (4 * F * s * c) 
        k_q = self.sigma * cDPrime / (4 * F * s * c)
        u = self.propellerParams.omega * self.r * k_t / (1 + k_q)
        aPrime = k_q / (1 + k_q)
        vA = self.v_inf + u
        vT = self.propellerParams.omega * self.r * (1 - aPrime)
        W = np.sqrt(vA**2 + vT**2)
        self.Re = self.propellerParams.rho * W * self.chord / self.propellerParams.mu
        self.Ma = W / self.propellerParams.a_inf
        return alpha, cL, cD, F, u, aPrime, W, cLPrime, cDPrime, vA, vT

    def residualFunction(self, phi):
        """Residual of the inflow angle equation for root finding."""
        _, _, _, _, u, aPrime, _, _, _, vA, vT = self.sectionParameters(phi)
        return np.tan(phi) - vA / vT

    def solve(self, v_inf, prevPhi=None):
        """Solve for inflow angle phi and return section forces and kinematics.

        If a previous-section phi is provided, use it to define a tighter
        bracket for the root search and expand it until a sign change is found;
        otherwise fall back to the full range (> 0, < 90).
        """
        self.v_inf = v_inf
        v_local = np.sqrt(self.v_inf**2 +(self.propellerParams.omega * self.r)**2)
        self.Ma = v_local / self.propellerParams.a_inf
        self.Re = self.propellerParams.rho * v_local * self.chord / self.propellerParams.mu

        phiMinDefault = np.radians(0.1)
        phiMaxDefault = np.radians(89.9)

        if prevPhi is None:
            bracket = [phiMinDefault, phiMaxDefault]
        else:
            center = np.clip(prevPhi, phiMinDefault, phiMaxDefault)
            delta = np.radians(1)  # initial half-width: ±1 deg

            def safeResidual(x):
                try:
                    return self.residualFunction(x)
                except Exception:
                    return np.nan

            # Expand the bracket until we find a sign change or hit the defaults
            while True:
                lower = max(phiMinDefault, center - delta)
                upper = min(phiMaxDefault, center + delta)

                fLower = safeResidual(lower)
                fUpper = safeResidual(upper)

                if np.isfinite(fLower) and np.isfinite(fUpper) and fLower * fUpper < 0:
                    bracket = [lower, upper]
                    break

                if lower <= phiMinDefault or upper >= phiMaxDefault:
                    bracket = [phiMinDefault, phiMaxDefault]
                    break

                delta *= 2.0

        try:
            result = scipy.optimize.root_scalar(
                self.residualFunction,
                method="brentq",
                xtol=1e-4,
                bracket=bracket,
            )
        except ValueError:
            result = scipy.optimize.root_scalar(
                self.residualFunction,
                method="newton",
                xtol=1e-4,
                x0=np.mean(bracket)
            )
        if not result.converged:
            raise RuntimeError("Root finding did not converge")

        phi = result.root
        alpha, _, _, F, u, aPrime, W, cLPrime, cDPrime, _, _ = self.sectionParameters(phi)
        dT = self.sigma * np.pi * self.propellerParams.rho * W**2 * cLPrime * self.r * self.dr
        dQ = self.sigma * np.pi * self.propellerParams.rho * W**2 * cDPrime * self.r**2 * self.dr
        return phi, dT, dQ, alpha, u, aPrime, cLPrime, cDPrime, F, W, self.Re, self.Ma
