"""Physical properties of the operating environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Environment:
    """Fluid and acoustic reference properties.

    Args:
        a_inf: Speed of sound in m/s.
        rho: Fluid density in kg/m^3.
        mu: Dynamic viscosity in Pa s.
        p_ref: Reference acoustic pressure in Pa.
    """

    a_inf: float = 343.0
    rho: float = 1.225
    mu: float = 1.81e-5
    p_ref: float = 2e-5
