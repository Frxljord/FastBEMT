"""Fast Blade Element Momentum Theory (BEMT) solver.

This module provides a faster version of the BEMT solver by:
- Reusing pre-built `aerosandbox.Airfoil` objects per section
- Caching / pre-tabulating airfoil coefficients
- Avoiding repeated DataFrame c[oncatenations
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
        # Pre-built aerosandbox Airfoil object (reused across many evaluations)
        self.airfoil = airfoil
        self.r = r
        self.dr = dr
        self.chord = chord
        self.theta = theta
        self.propellerParams = propellerParams
        self.Ma = self.propellerParams.omega * self.r / self.propellerParams.aInf
        self.Re = self.propellerParams.rho * self.propellerParams.omega * self.r * self.chord / self.propellerParams.mu
        # Pre-tabulated CL/CD vs alpha tables keyed by (ReBin, MaBin)
        self._tables = {}
        # Precompute Prandtl loss factor table
        self._buildPrandtlLossTable()

    @property
    def sigma(self):
        """Local solidity for this section."""
        return self.propellerParams.nBlades * self.chord / (2 * np.pi * self.r)

    def _buildPrandtlLossTable(self):
        """Precompute Prandtl loss factor on a phi grid for fast interpolation."""
        # Phi grid (radians)
        self._phi_grid = np.linspace(np.radians(0.1), np.radians(89.9), 100)

        sin_phi = np.sin(self._phi_grid)
        B = self.propellerParams.nBlades
        r = self.r

        f_tip = B * (self.propellerParams.propRadius - r) / (2 * r * sin_phi)
        f_hub = B * (r - self.propellerParams.hubRadius) / (2 * r * sin_phi)

        # Clip to avoid overflow
        f_tip = np.clip(f_tip, 0.0, 500.0)
        f_hub = np.clip(f_hub, 0.0, 500.0)

        F_tip = 2 * np.arccos(np.exp(-f_tip)) / np.pi
        F_hub = 2 * np.arccos(np.exp(-f_hub)) / np.pi

        self._F_grid = F_tip * F_hub

    def prandtlLoss(self, phi):
        """Return Prandtl tip–hub loss factor for a given inflow angle phi (rad)."""
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
            alpha_grid = np.linspace(0.0, 30.0, 61)  # 0.5 deg step
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
        cLPrime = cL * np.cos(phi) - cD * np.sin(phi)
        cDPrime = cL * np.sin(phi) + cD * np.cos(phi)
        F = self.prandtlLoss(phi)
        a = 1 / ((4 * F * np.sin(phi) ** 2) / (self.sigma * cLPrime) - 1)
        aPrime = 1 / ((4 * F * np.sin(phi) * np.cos(phi)) / (self.sigma * cDPrime) + 1)
        vA = (1 + a) * self.propellerParams.vInf
        vT = self.propellerParams.omega * self.r * (1 - aPrime)
        W = np.sqrt(vA**2 + vT**2)
        self.Re = self.propellerParams.rho * W * self.chord / self.propellerParams.mu
        self.Ma = W / self.propellerParams.aInf
        return alpha, cL, cD, F, a, aPrime, W, cLPrime, cDPrime

    def residualFunction(self, phi):
        """Residual of the inflow angle equation for root finding."""
        _, _, _, _, a, aPrime, _, _, _ = self.sectionParameters(phi)
        return np.sin(phi) / (1 + a) - self.propellerParams.vInf / (self.propellerParams.omega * self.r) * (np.cos(phi) / (1 - aPrime))

    def solve(self, prevPhi=None):
        """Solve for inflow angle phi and return section forces and kinematics.

        If a previous-section phi is provided, use it to define a tighter
        bracket for the root search and expand it until a sign change is found;
        otherwise fall back to the full range (> 0, < 90).
        """
        # Default wide bracket in radians
        phiMinDefault = np.radians(0.1)
        phiMaxDefault = np.radians(89.9)

        if prevPhi is None:
            # No prior information: use the full bracket
            bracket = [phiMinDefault, phiMaxDefault]
        else:
            # Start with a tight bracket around the previous section's solution
            center = np.clip(prevPhi, phiMinDefault, phiMaxDefault)
            delta = np.radians(1)  # initial half-width: ±1 deg

            # Helper to evaluate residual safely
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

                # Check for valid sign change
                if np.isfinite(fLower) and np.isfinite(fUpper) and fLower * fUpper < 0:
                    bracket = [lower, upper]
                    break

                # If we've already reached the full default bracket, give up and use it
                if lower <= phiMinDefault or upper >= phiMaxDefault:
                    bracket = [phiMinDefault, phiMaxDefault]
                    break

                # Otherwise, expand the search region
                delta *= 2.0

        result = scipy.optimize.root_scalar(
            self.residualFunction,
            method="brentq",
            xtol=1e-4,
            bracket=bracket,
        )
        if not result.converged:
            raise RuntimeError("Root finding did not converge")

        phi = result.root
        alpha, cL, cD, F, a, aPrime, W, cLPrime, cDPrime = self.sectionParameters(phi)
        dT = self.sigma * np.pi * self.propellerParams.rho * W**2 * cLPrime * self.r * self.dr
        dQ = self.sigma * np.pi * self.propellerParams.rho * W**2 * cDPrime * self.r**2 * self.dr
        return phi, dT, dQ, alpha, a, aPrime, cLPrime, cDPrime, F, W, self.Re, self.Ma
