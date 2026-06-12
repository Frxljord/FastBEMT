"""Physical properties of the operating environment."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Environment:
    """Fluid and acoustic reference properties.

    Args:
        a_inf: Speed of sound in m/s.
        rho: Fluid density in kg/m^3.
        mu: Dynamic viscosity in Pa s.
        p_ref: Reference acoustic pressure in Pa.
    """

    a_inf: float
    rho: float
    mu: float
    p_ref: float = 2e-5

    def __post_init__(self) -> None:
        """Validate physical properties."""
        for name in ("a_inf", "rho", "mu", "p_ref"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite value greater than zero.")
            object.__setattr__(self, name, value)
