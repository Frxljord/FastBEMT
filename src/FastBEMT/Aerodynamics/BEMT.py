"""Blade element momentum theory analysis for one or more operating points."""

from __future__ import annotations

from numbers import Real
import warnings
from typing import Final, NamedTuple, Sequence, TYPE_CHECKING

import aerosandbox as asb
import numpy as np
import pandas as pd

from .Section import SectionForces
from ..Utils.Environment import Environment

if TYPE_CHECKING:
    from ..Propeller import Propeller


SOLUTION_COLUMNS: Final[tuple[str, ...]] = (
    "r",
    "dr",
    "chord",
    "twist",
    "phi",
    "alpha",
    "Cl",
    "Cd",
    "u",
    "a_prime",
    "d_t",
    "d_q",
    "F",
    "W",
    "Re",
    "Ma",
    "ds",
    "dp",
)

PERFORMANCE_COLUMNS: Final[tuple[str, ...]] = (
    "thrust",
    "torque",
    "thrust_coefficient",
    "torque_coefficient",
    "figure_of_merit",
)

OperatingInput = Real | Sequence[float] | np.ndarray


class BEMTPerformance(NamedTuple):
    """Integrated propeller performance at one operating point."""

    thrust: float
    torque: float
    thrust_coefficient: float
    torque_coefficient: float
    figure_of_merit: float


class BEMT:
    """Solve BEMT over RPM and freestream or advance-ratio values.

    The analysis is performed during initialization. ``solution_data`` uses a
    ``(rpm, v_inf, section)`` MultiIndex, while ``performance`` uses a
    ``(rpm, v_inf)`` MultiIndex. When ``J`` is provided, the freestream
    velocity is calculated as ``v_inf = J * (rpm / 60) * diameter``.

    Args:
        propeller: Propeller containing the blade geometry.
        environment: Fluid and acoustic reference properties.
        rpm: One RPM value or a one-dimensional sequence of RPM values.
        v_inf: One freestream velocity or a one-dimensional sequence of
            freestream velocities. Each value is applied uniformly to all
            radial sections for its operating point. Mutually exclusive with
            ``J``.
        J: One advance ratio or a one-dimensional sequence of advance ratios.
            Each value is combined with every RPM value. Mutually exclusive
            with ``v_inf``.
    """

    def __init__(
        self,
        propeller: Propeller,
        environment: Environment,
        rpm: OperatingInput,
        v_inf: OperatingInput | None = None,
        *,
        J: OperatingInput | None = None,
    ) -> None:
        self.propeller = propeller
        self.environment = environment
        if environment is not propeller.environment:
            raise ValueError(
                "BEMT environment must be the environment stored by the propeller."
            )

        self.rpm = self._normalize_operating_values(
            rpm,
            name="rpm",
            require_positive=True,
        )
        self.omega = 2.0 * np.pi * self.rpm / 60.0

        if (v_inf is None) == (J is None):
            raise ValueError("Exactly one of v_inf and J must be provided.")

        if J is None:
            self.J: np.ndarray | None = None
            self.v_inf = self._normalize_operating_values(v_inf, name="v_inf")
            self.operating_points = pd.MultiIndex.from_product(
                [self.rpm, self.v_inf],
                names=["rpm", "v_inf"],
            )
        else:
            self.J = self._normalize_operating_values(J, name="J")
            diameter = 2.0 * float(self.propeller.geometry["tip_radius"])
            if not np.isfinite(diameter) or diameter <= 0.0:
                raise ValueError(
                    "Propeller diameter must be finite and greater than zero "
                    "when J is provided."
                )

            computed_v_inf = np.multiply.outer(
                self.rpm / 60.0 * diameter,
                self.J,
            )
            self.v_inf = computed_v_inf.reshape(-1)
            self.operating_points = pd.MultiIndex.from_arrays(
                [
                    np.repeat(self.rpm, self.J.size),
                    self.v_inf,
                ],
                names=["rpm", "v_inf"],
            )

        self._airfoils = [
            asb.Airfoil(coordinates=coordinates)
            for coordinates in self.propeller.geometry["airfoil"]
        ]
        self.solution_data = self._solve()
        self.performance = self._integrate_performance()

    @property
    def has_single_operating_point(self) -> bool:
        """Whether this analysis contains exactly one RPM/velocity pair."""
        return len(self.operating_points) == 1

    def solution_for(
        self,
        rpm: float | None = None,
        v_inf: float | None = None,
    ) -> pd.DataFrame:
        """Return section results for one operating point.

        If the analysis contains one operating point, both arguments may be
        omitted. Multi-point analyses require both values.
        """
        operating_point = self.resolve_operating_point(rpm, v_inf)
        return self.solution_data.xs(
            operating_point,
            level=("rpm", "v_inf"),
            drop_level=True,
        )

    def performance_for(
        self,
        rpm: float | None = None,
        v_inf: float | None = None,
    ) -> BEMTPerformance:
        """Return integrated performance for one operating point."""
        operating_point = self.resolve_operating_point(rpm, v_inf)
        values = self.performance.loc[operating_point]
        return BEMTPerformance(
            *(float(values[column]) for column in PERFORMANCE_COLUMNS)
        )

    def compute_total_forces(self) -> pd.DataFrame:
        """Return the performance table for all operating points."""
        return self.performance

    def _solve(self) -> pd.DataFrame:
        """Solve all radial sections for every operating point."""
        case_frames: list[pd.DataFrame] = []

        for rpm, v_inf in self.operating_points:
            sections = self._build_sections(float(rpm))
            case_frames.append(self._solve_operating_point(sections, float(v_inf)))

        return pd.concat(
            case_frames,
            keys=list(self.operating_points),
            names=["rpm", "v_inf", "section"],
        )

    def _build_sections(self, rpm: float) -> list[SectionForces | None]:
        """Create fresh section solvers for the active RPM."""
        geometry = self.propeller.geometry
        omega = 2.0 * np.pi * rpm / 60.0
        sections: list[SectionForces | None] = []

        for index, airfoil in enumerate(self._airfoils):
            section_width = float(geometry["dr"][index])
            if section_width <= 0.0:
                sections.append(None)
                continue

            sections.append(
                SectionForces(
                    airfoil=airfoil,
                    r=float(geometry["r"][index]),
                    dr=section_width,
                    chord=float(geometry["chord"][index]),
                    theta=float(np.radians(geometry["twist"][index])),
                    environment=self.environment,
                    omega=omega,
                    prop_radius=float(geometry["tip_radius"]),
                    hub_radius=float(geometry["hub_radius"]),
                    n_blades=int(geometry["n_blades"]),
                )
            )

        return sections

    def _solve_operating_point(
        self,
        sections: list[SectionForces | None],
        v_inf: float,
    ) -> pd.DataFrame:
        """Solve one operating point, warm-starting along the blade radius."""
        rows: list[list[float]] = []
        previous_phi: float | None = None

        for section_index in range(len(sections)):
            row = self._solve_section(
                sections,
                section_index,
                v_inf,
                previous_phi,
            )
            if np.isfinite(row[4]):
                previous_phi = float(np.radians(row[4]))
            rows.append(row)

        frame = pd.DataFrame(rows, columns=SOLUTION_COLUMNS)
        frame.index = pd.RangeIndex(len(frame), name="section")
        return frame

    def _solve_section(
        self,
        sections: list[SectionForces | None],
        section_index: int,
        v_inf: float,
        previous_phi: float | None,
    ) -> list[float]:
        """Solve one radial section and format its output row."""
        geometry = self.propeller.geometry
        section = sections[section_index]
        if section is None:
            return self._empty_section_row(section_index)

        try:
            (
                phi,
                d_t,
                d_q,
                alpha,
                u,
                a_prime,
                c_l,
                c_d,
                loss_factor,
                relative_velocity,
                reynolds,
                mach,
                delta_star_upper,
                delta_star_lower,
            ) = section.solve(v_inf, prev_phi=previous_phi)
        except RuntimeError as error:
            warnings.warn(
                f"BEMT section at r={geometry['r'][section_index]} m failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._empty_section_row(section_index)

        return [
            float(geometry["r"][section_index]),
            float(geometry["dr"][section_index]),
            float(geometry["chord"][section_index]),
            float(geometry["twist"][section_index]),
            float(np.degrees(phi)),
            float(alpha),
            float(c_l),
            float(c_d),
            float(u),
            float(a_prime),
            float(d_t),
            float(d_q),
            float(loss_factor),
            float(relative_velocity),
            float(reynolds),
            float(mach),
            float(delta_star_upper),
            float(delta_star_lower),
        ]

    def _empty_section_row(self, section_index: int) -> list[float]:
        """Return geometry and empty outputs for an unsolved section."""
        geometry = self.propeller.geometry
        return [
            float(geometry["r"][section_index]),
            float(geometry["dr"][section_index]),
            float(geometry["chord"][section_index]),
            float(geometry["twist"][section_index]),
            *([np.nan] * 14),
        ]

    def _integrate_performance(self) -> pd.DataFrame:
        """Integrate loads and calculate coefficients for every case."""
        rows: list[BEMTPerformance] = []

        for rpm, v_inf in self.operating_points:
            case = self.solution_for(float(rpm), float(v_inf))
            thrust = float(case["d_t"].sum())
            torque = float(case["d_q"].sum())
            revolutions_per_second = float(rpm) / 60.0
            diameter = 2.0 * float(self.propeller.geometry["tip_radius"])
            density = float(self.environment.rho)

            thrust_coefficient = thrust / (
                density * revolutions_per_second**2 * diameter**4
            )
            torque_coefficient = torque / (
                density * revolutions_per_second**2 * diameter**5
            )
            if thrust_coefficient > 0.0 and torque_coefficient > 0.0:
                figure_of_merit = (
                    np.sqrt(2.0 / np.pi)
                    * thrust_coefficient**1.5
                    / (2.0 * np.pi * torque_coefficient)
                )
            else:
                figure_of_merit = np.nan

            rows.append(
                BEMTPerformance(
                    thrust=thrust,
                    torque=torque,
                    thrust_coefficient=float(thrust_coefficient),
                    torque_coefficient=float(torque_coefficient),
                    figure_of_merit=float(figure_of_merit),
                )
            )

        return pd.DataFrame(
            rows,
            index=self.operating_points,
            columns=PERFORMANCE_COLUMNS,
        )

    def resolve_operating_point(
        self,
        rpm: float | None,
        v_inf: float | None,
    ) -> tuple[float, float]:
        """Validate and return one operating-point key."""
        if rpm is None and v_inf is None:
            if self.has_single_operating_point:
                point = self.operating_points[0]
                return float(point[0]), float(point[1])
            raise ValueError(
                "rpm and v_inf are required when the BEMT contains multiple "
                "operating points."
            )
        if rpm is None or v_inf is None:
            raise ValueError("rpm and v_inf must be provided together.")

        operating_point = (float(rpm), float(v_inf))
        if operating_point not in self.operating_points:
            raise KeyError(f"Operating point {operating_point} was not computed.")
        return operating_point

    @staticmethod
    def _normalize_operating_values(
        values: OperatingInput,
        *,
        name: str,
        require_positive: bool = False,
    ) -> np.ndarray:
        """Return a validated one-dimensional array of operating values."""
        if isinstance(values, Real):
            normalized = np.array([float(values)], dtype=float)
        else:
            normalized = np.asarray(values, dtype=float)

        if normalized.ndim != 1 or normalized.size == 0:
            raise ValueError(f"{name} must be a scalar or a non-empty 1D array.")
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{name} must contain only finite values.")
        if require_positive and np.any(normalized <= 0.0):
            raise ValueError(f"{name} values must be greater than zero.")
        if np.unique(normalized).size != normalized.size:
            raise ValueError(f"{name} values must be unique.")
        return normalized
