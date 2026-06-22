from __future__ import annotations

import numpy as np
import pyfar as pf
import torch

from .Kinematics import Kinematics
from .Utils.DataLoader import normalize_propeller_geometry
from .Utils.Environment import Environment
from .Utils.Simulation import Simulation

GeometryDict = dict[str, np.ndarray | int | float | list]


class Propeller:
    """Propeller geometry, environment, and simulation state.

    Aerodynamic operating-point calculations are handled by
    :class:`FastBEMT.Aerodynamics.BEMT`. Aeroacoustic analyses consume this
    object directly through F1A and BPM.
    """

    def __init__(
        self,
        geometry: GeometryDict,
        environment: Environment,
        simulation: Simulation,
    ) -> None:
        """Initialize propeller geometry and cached section tensors."""
        self.geometry = normalize_propeller_geometry(geometry)
        self.environment = environment
        self.simulation = simulation
        self.dtype = torch.float32
        self.device = simulation.device
        self.section_areas()
        self.calculate_boat_tail_angle()

        self.sweep = np.asarray(self.geometry["sweep"]).tolist()
        self.rake = np.asarray(self.geometry["rake"]).tolist()
        self._initialize_geometry_cache()

        octave_freqs = pf.dsp.filter.fractional_octave_frequencies(
            num_fractions=3,
            frequency_range=(20, 20000),
        )[0]
        self.third_octave_freqs = torch.as_tensor(
            octave_freqs,
            dtype=self.dtype,
            device=self.device,
        )

        self.kinematics: Kinematics | None = None

    def _initialize_geometry_cache(self) -> None:
        """Validate and cache immutable section data on the compute device."""
        section_arrays = {
            "r": np.array(self.geometry["r"], dtype=np.float64, copy=True),
            "dr": np.array(self.geometry["dr"], dtype=np.float64, copy=True),
            "chord": np.array(
                self.geometry["chord"],
                dtype=np.float64,
                copy=True,
            ),
            "twist": np.array(
                self.geometry["twist"],
                dtype=np.float64,
                copy=True,
            ),
            "area": np.array(
                self.geometry["cross_section"],
                dtype=np.float64,
                copy=True,
            ),
            "boat_tail_angle": np.array(
                self.geometry["boat_tail_angle"],
                dtype=np.float64,
                copy=True,
            ),
            "sweep": np.array(
                self.sweep,
                dtype=np.float64,
                copy=True,
            ),
            "rake": np.array(
                self.rake,
                dtype=np.float64,
                copy=True,
            ),
        }
        if section_arrays["r"].ndim != 1:
            raise ValueError("geometry['r'] must be one-dimensional.")
        section_count = int(section_arrays["r"].shape[0])
        if section_count == 0:
            raise ValueError("Propeller geometry must contain at least one section.")
        for name, values in section_arrays.items():
            if values.ndim != 1 or values.shape[0] != section_count:
                raise ValueError(
                    f"geometry['{name}'] must be one-dimensional with "
                    f"{section_count} entries."
                )
            values.setflags(write=False)

        self.n_sections = section_count
        self.n_blades = int(self.geometry["n_blades"])
        if self.n_blades <= 0:
            raise ValueError("geometry['n_blades'] must be greater than zero.")

        self.section_geometry_np = section_arrays
        self.f1a_geometry_mask = (
            np.isfinite(section_arrays["r"])
            & (np.abs(section_arrays["r"]) > 1.0e-12)
            & np.isfinite(section_arrays["dr"])
            & (section_arrays["dr"] > 0.0)
            & np.isfinite(section_arrays["chord"])
            & np.isfinite(section_arrays["twist"])
            & np.isfinite(section_arrays["area"])
            & np.isfinite(section_arrays["sweep"])
            & np.isfinite(section_arrays["rake"])
        )
        self.bpm_geometry_mask = (
            self.f1a_geometry_mask
            & np.isfinite(section_arrays["boat_tail_angle"])
        )
        self.all_f1a_geometry_valid = bool(np.all(self.f1a_geometry_mask))
        self.all_bpm_geometry_valid = bool(np.all(self.bpm_geometry_mask))
        if not np.any(self.f1a_geometry_mask):
            raise ValueError("No valid blade sections are available for F1A.")

        self.f1a_geometry_mask_tensor = torch.tensor(
            self.f1a_geometry_mask,
            dtype=torch.bool,
            device=self.device,
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

    def section_areas(self) -> None:
        """Calculate airfoil cross-sectional areas with the shoelace formula."""
        areas: list[float] = []
        for idx, coords in enumerate(self.geometry["airfoils"]):
            x = coords[:, 0]
            y = coords[:, 1]
            area_normalized = 0.5 * np.abs(
                np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
            )
            areas.append(area_normalized * self.geometry["chord"][idx] ** 2)
        self.geometry["cross_section"] = np.array(areas)

    def calculate_boat_tail_angle(self) -> None:
        """Calculate trailing-edge boat-tail angle for each airfoil section."""
        angles: list[float] = []
        for coords in self.geometry["airfoils"]:
            leading_edge_index = np.argmin(coords[:, 0])
            upper = coords[: leading_edge_index + 1]
            lower = coords[leading_edge_index:]

            upper_te = upper[upper[:, 0] > 0.95]
            lower_te = lower[lower[:, 0] > 0.95]
            if len(upper_te) < 2 or len(lower_te) < 2:
                angles.append(0.0)
                continue

            upper_slope, _ = np.polyfit(upper_te[:, 0], upper_te[:, 1], 1)
            lower_slope, _ = np.polyfit(lower_te[:, 0], lower_te[:, 1], 1)
            angle = np.degrees(np.abs(np.arctan(upper_slope) - np.arctan(lower_slope)))
            angles.append(angle)

        self.geometry["boat_tail_angle"] = np.array(angles)
