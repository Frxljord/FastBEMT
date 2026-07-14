from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .Kinematics import Kinematics
from .Utils.DataLoader import normalize_propeller_geometry
from .Utils.Environment import Environment
from .Utils.Simulation import Simulation

THIRD_OCTAVE_FREQUENCIES_HZ = (
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
)


class Propeller:
    """Propeller geometry, environment, and simulation state.

    Aerodynamic operating-point calculations are handled by
    :class:`FastBEMT.Aerodynamics.BEMT`. Aeroacoustic analyses consume this
    object directly through F1A and BPM.
    """

    def __init__(
        self,
        geometry: Mapping[str, Any],
        environment: Environment,
        simulation: Simulation,
    ) -> None:
        """Initialize propeller geometry and cached section tensors."""
        self.geometry = normalize_propeller_geometry(geometry)
        self.environment = environment
        self.simulation = simulation
        self.dtype = torch.float32
        self.device = simulation.device
        self._calculate_section_areas()
        self._calculate_boat_tail_angles()
        self._initialize_geometry_cache()

        self.third_octave_freqs = torch.as_tensor(
            THIRD_OCTAVE_FREQUENCIES_HZ,
            dtype=self.dtype,
            device=self.device,
        )

        self.kinematics: Kinematics | None = None

    def _initialize_geometry_cache(self) -> None:
        """Cache canonical section data on the compute device."""
        section_arrays = {
            "r": self.geometry["r"],
            "dr": self.geometry["dr"],
            "chord": self.geometry["chord"],
            "twist": self.geometry["twist"],
            "area": self.geometry["cross_section"],
            "boat_tail_angle": self.geometry["boat_tail_angle"],
            "sweep": self.geometry["sweep"],
            "rake": self.geometry["rake"],
        }
        section_count = int(section_arrays["r"].shape[0])
        for values in section_arrays.values():
            values.setflags(write=False)

        self.n_sections = section_count
        self.n_blades = int(self.geometry["n_blades"])
        self.section_geometry_np = section_arrays
        self.aerodynamic_section_mask = (
            (section_arrays["r"] > float(self.geometry["hub_radius"]))
            & (section_arrays["r"] < float(self.geometry["tip_radius"]))
        )
        self.section_radius = torch.tensor(
            section_arrays["r"],
            dtype=self.dtype,
            device=self.device,
        )
        self.section_width = torch.tensor(
            section_arrays["dr"],
            dtype=self.dtype,
            device=self.device,
        )
        self.section_chord = torch.tensor(
            section_arrays["chord"],
            dtype=self.dtype,
            device=self.device,
        )
        self.section_twist_rad = torch.deg2rad(
            torch.tensor(
                section_arrays["twist"],
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.section_area = torch.tensor(
            section_arrays["area"],
            dtype=self.dtype,
            device=self.device,
        )
        self.section_sweep = torch.tensor(
            section_arrays["sweep"],
            dtype=self.dtype,
            device=self.device,
        )
        self.section_rake = torch.tensor(
            section_arrays["rake"],
            dtype=self.dtype,
            device=self.device,
        )
        self.rho_tensor = torch.tensor(
            self.environment.rho,
            dtype=self.dtype,
            device=self.device,
        )
        self.a_inf_tensor = torch.tensor(
            self.environment.a_inf,
            dtype=self.dtype,
            device=self.device,
        )
        self.f1a_thickness_strength = (
            self.rho_tensor
            * self.section_area
            * self.section_width
            / (4.0 * np.pi)
        )
        self.f1a_dipole_strength = (
            self.section_width / (4.0 * np.pi * self.a_inf_tensor)
        )

    def _calculate_section_areas(self) -> None:
        """Calculate airfoil cross-sectional areas with the shoelace formula."""
        areas: list[float] = []
        for section_index, coords in enumerate(self.geometry["airfoils"]):
            x = coords[:, 0]
            y = coords[:, 1]
            area_normalized = 0.5 * np.abs(
                np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
            )
            areas.append(
                area_normalized * self.geometry["chord"][section_index] ** 2
            )
        self.geometry["cross_section"] = np.array(areas)

    def _calculate_boat_tail_angles(self) -> None:
        """Calculate trailing-edge boat-tail angle for each airfoil section."""
        angles: list[float] = []
        for coords in self.geometry["airfoils"]:
            leading_edge_index = np.argmin(coords[:, 0])
            upper = coords[: leading_edge_index + 1]
            lower = coords[leading_edge_index:]

            upper_te = upper[upper[:, 0] > 0.95]
            lower_te = lower[lower[:, 0] > 0.95]
            upper_slope, _ = np.polyfit(upper_te[:, 0], upper_te[:, 1], 1)
            lower_slope, _ = np.polyfit(lower_te[:, 0], lower_te[:, 1], 1)
            angle = np.degrees(
                np.abs(np.arctan(upper_slope) - np.arctan(lower_slope))
            )
            angles.append(angle)

        self.geometry["boat_tail_angle"] = np.array(angles)
