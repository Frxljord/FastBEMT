import numpy as np
import pandas as pd
import aerosandbox as asb

from .Section import SectionForces


class Propeller:
    """High-level propeller BEMT analysis using the fast per-section solver."""

    def __init__(self, propellerGeometry, propellerParams):
        """Initialize analysis with geometry and global propeller parameters."""
        self.propellerGeometry = propellerGeometry
        self.propellerParams = propellerParams
        self.solutionData = pd.DataFrame(
            columns=[
                "radius",
                "chord",
                "twist",
                "phi",
                "alpha",
                "Cl",
                "Cd",
                "a",
                "a_prime",
                "dT",
                "dQ",
                "F",
                "W",
                "Re",
                "Ma",
            ]
        )
        # Internal row buffer to avoid repeated DataFrame concatenation
        self._rows = []
        self._sectionAirfoils = []
        for airfoil in self.propellerGeometry["airfoil"]:
            self._sectionAirfoils.append(asb.Airfoil(coordinates=airfoil))

    def processSection(self, r, dr, chord, thetaDeg, airfoilAsb, prevPhi=None):
        """Run the fast BEMT solver for a single section at radius r."""
        theta = np.radians(thetaDeg)

        sectionForce = SectionForces(
            airfoil=airfoilAsb,
            r=r,
            dr=dr,
            chord=chord,
            theta=theta,
            propellerParams=self.propellerParams,
        )

        try:
            phi, dT, dQ, alpha, a, aPrime, Cl, Cd, F, W, Re, Ma = sectionForce.solve(
                prevPhi=prevPhi
            )
            return [
                r,
                chord,
                np.degrees(theta),
                np.degrees(phi),
                alpha,
                Cl,
                Cd,
                a,
                aPrime,
                dT,
                dQ,
                F,
                W,
                Re,
                Ma,
            ]
        except RuntimeError as e:
            print(f"Error in section {r}: {e}")
            return [np.nan] * 14

    def runBEMT(self):
        """Run the BEMT solution along the blade radius (sequentially)."""
        prevPhi = None  # Previous section's phi in radians
        self.results = []
        self._rows = []

        for r, dr, chord, twist, airfoilAsb in zip(
            self.propellerGeometry["r"],
            self.propellerGeometry["dr"],
            self.propellerGeometry["chord"],
            self.propellerGeometry["twist"],
            self._sectionAirfoils,
        ):
            sectionResult = self.processSection(
                r, dr, chord, twist, airfoilAsb, prevPhi=prevPhi
            )

            (
                rVal,
                chordVal,
                thetaDeg,
                phiDeg,
                alpha,
                Cl,
                Cd,
                a,
                aPrime,
                dT,
                dQ,
                F,
                W,
                Re,
                Ma,
            ) = sectionResult

            if not np.isnan(phiDeg):
                prevPhi = np.radians(phiDeg)

            self.results.append(sectionResult)
            self._rows.append(
                [
                    rVal,
                    chordVal,
                    thetaDeg,
                    phiDeg,
                    alpha,
                    Cl,
                    Cd,
                    a,
                    aPrime,
                    dT,
                    dQ,
                    F,
                    W,
                    Re,
                    Ma,
                ]
            )

        self.solutionData = pd.DataFrame(self._rows, columns=self.solutionData.columns)

    def computeTotalForces(self):
        """Compute total thrust, torque, and nondimensional coefficients Ct, Cp."""
        totalThrust = self.solutionData["dT"].sum()
        totalTorque = self.solutionData["dQ"].sum()
        Ct = totalThrust / (
            self.propellerParams.rho
            * (self.propellerParams.RPM / 60) ** 2
            * (self.propellerParams.propDiameter) ** 4
        )
        Cp = 2 * np.pi * totalTorque / (
            self.propellerParams.rho
            * (self.propellerParams.RPM / 60) ** 2
            * (self.propellerParams.propDiameter) ** 5
        )
        return totalThrust, totalTorque, Ct, Cp