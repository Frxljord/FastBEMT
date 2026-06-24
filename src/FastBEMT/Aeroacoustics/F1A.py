from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ._common import (
    ArrayLike,
    PathInput,
    normalize_observer_batch_size,
    observer_tensor,
)
from .Utils import (
    a_weighting_db,
    spl_spectrum_to_overall_level,
    time_domain_to_spl_spectrum,
)
from ..Kinematics import Kinematics

if TYPE_CHECKING:
    from ..Aerodynamics.BEMT import BEMT
    from ..Propeller import Propeller


class F1A:
    """Farassat formulation 1A for compact rotating sources.

    Args:
        propeller: Propeller containing geometry and simulation settings.
        bemt: Optional BEMT analysis providing steady section loads. Omit this
            when using direct loading files.
        kinematics: Optional shared propeller kinematics. When omitted, F1A
            reuses ``propeller.kinematics`` if it matches the RPM, otherwise it
            creates one.
        rpm: Operating RPM. Optional for a single-point BEMT analysis and
            required for multi-point BEMT analyses or direct loadings.
        v_inf: BEMT freestream velocity. Required with ``rpm`` when
            selecting from a multi-point BEMT analysis.
        loadings: Optional path to a ``.pt`` dictionary containing direct
            global-frame force per unit span and section geometry. The required
            format is ``{"loadings": tensor, "sections": dict}``, where
            ``loadings`` has shape ``(T, B, S, 4)`` with source time in channel
            0 and force components in channels 1:4, and ``sections`` contains
            ``r_mid_m`` and ``dr_m``.
        last_rotations: Optional number of final rotor revolutions to keep from
            direct loading files. Use this to discard startup transients before
            F1A builds kinematics and differentiates the load history.
    Observer-dependent source tensors use dimension order
    ``(O, T, B, S, ...)``.
    """

    def __init__(
        self,
        propeller: Propeller,
        bemt: BEMT | PathInput | None = None,
        *,
        kinematics: Kinematics | None = None,
        rpm: float | None = None,
        v_inf: float | None = None,
        loadings: PathInput | None = None,
        last_rotations: float | None = None,
    ) -> None:
        self.propeller = propeller
        self.environment = propeller.environment

        self.device = torch.device(propeller.device)
        self.dtype = propeller.dtype
        self.rho = propeller.rho_tensor
        self.a_inf = propeller.a_inf_tensor
        self.loadings_path: Path | None = None

        from ..Aerodynamics.BEMT import BEMT

        if bemt is not None and not isinstance(bemt, BEMT):
            if loadings is not None:
                raise ValueError(
                    "Specify direct F1A loadings either as the second "
                    "positional argument or with loadings=, not both."
                )
            loadings = bemt
            bemt = None

        using_direct_loadings = loadings is not None
        if using_direct_loadings == (bemt is not None):
            raise ValueError("Specify exactly one of bemt or direct loadings.")

        if using_direct_loadings:
            self.bemt: BEMT | None = None
            if rpm is None:
                raise ValueError("rpm is required for direct F1A loadings.")
            self.rpm = float(rpm)
            if not np.isfinite(self.rpm) or self.rpm <= 0.0:
                raise ValueError("rpm must be finite and greater than zero.")
            if v_inf is not None:
                raise ValueError(
                    "v_inf is only used when F1A loadings come from BEMT."
                )
            self.v_inf = None

            self.loadings_path = self._required_file_path(
                loadings,
                name="loadings",
                suffix=".pt",
            )
            direct_loadings, direct_source_times, direct_sections = (
                self._load_direct_loadings(self.loadings_path)
            )
            direct_geometry = self._direct_loading_geometry(direct_sections)
            direct_loadings, direct_source_times = (
                self._trim_direct_loading_history(
                    direct_loadings,
                    direct_source_times,
                    last_rotations,
                )
            )
            loads, direct_geometry, section_mask = (
                self._select_direct_load_sections(
                    direct_loadings,
                    direct_geometry,
                    direct_source_times,
                )
            )
            loads = -loads

            kinematic_geometry = {
                "r": direct_geometry["r"],
                "chord": direct_geometry["chord"],
                "twist": direct_geometry["twist"],
                "sweep": direct_geometry["sweep"],
                "rake": direct_geometry["rake"],
            }
            if kinematics is not None:
                if not self._kinematics_matches(
                    kinematics,
                    source_times=direct_source_times,
                    section_radius=direct_geometry["r"],
                ):
                    raise ValueError(
                        "The supplied Kinematics object must belong to this "
                        "propeller, operating RPM, source-time grid, and "
                        "direct-loading section grid."
                    )
                self.kinematics = kinematics
            else:
                self.kinematics = Kinematics(
                    propeller,
                    rpm=self.rpm,
                    source_times=direct_source_times,
                    section_geometry=kinematic_geometry,
                )
            self.nt = self.kinematics.nt
            self.nb = self.kinematics.nb

            self.section_mask = section_mask
            self._uses_all_sections = True
            self._selected_sections = torch.ones(
                direct_geometry["r"].shape,
                dtype=torch.bool,
                device=self.device,
            )
            self.r = torch.as_tensor(
                direct_geometry["r"],
                dtype=self.dtype,
                device=self.device,
            )
            self.dr = torch.as_tensor(
                direct_geometry["dr"],
                dtype=self.dtype,
                device=self.device,
            )
            self.area = torch.as_tensor(
                direct_geometry["area"],
                dtype=self.dtype,
                device=self.device,
            )
            self.chord = torch.as_tensor(
                direct_geometry["chord"],
                dtype=self.dtype,
                device=self.device,
            )
            self.twist_rad = torch.deg2rad(
                torch.as_tensor(
                    direct_geometry["twist"],
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.sweep = torch.as_tensor(
                direct_geometry["sweep"],
                dtype=self.dtype,
                device=self.device,
            )
            self.rake = torch.as_tensor(
                direct_geometry["rake"],
                dtype=self.dtype,
                device=self.device,
            )
            self.thickness_strength = (
                self.rho * self.area * self.dr / (4.0 * np.pi)
            )
            self.dipole_strength = self.dr / (4.0 * np.pi * self.a_inf)
        else:
            if not isinstance(bemt, BEMT):
                raise TypeError("bemt must be a FastBEMT Aerodynamics.BEMT object.")
            if last_rotations is not None:
                raise ValueError(
                    "last_rotations is only supported for direct F1A loadings."
                )
            if bemt.propeller is not propeller:
                raise ValueError("The BEMT analysis belongs to a different propeller.")
            if bemt.environment is not self.environment:
                raise ValueError("F1A and BEMT must use the same environment.")
            self.bemt = bemt
            self.rpm, self.v_inf = bemt.resolve_operating_point(rpm, v_inf)
            solution = bemt.solution_for(self.rpm, self.v_inf)

            if kinematics is not None:
                if not self._kinematics_matches(kinematics):
                    raise ValueError(
                        "The supplied Kinematics object must belong to this "
                        "propeller and operating RPM."
                    )
                self.kinematics = kinematics
            elif self._kinematics_matches(propeller.kinematics):
                self.kinematics = propeller.kinematics
            else:
                self.kinematics = Kinematics(propeller, rpm=self.rpm)
            propeller.kinematics = self.kinematics
            self.nt = self.kinematics.nt
            self.nb = self.kinematics.nb

            section_mask = propeller.f1a_geometry_mask.copy()
            d_t = np.asarray(solution["d_t"].values, dtype=np.float64)
            d_q = np.asarray(solution["d_q"].values, dtype=np.float64)
            section_mask &= np.isfinite(d_t) & np.isfinite(d_q)
            if not np.any(section_mask):
                raise ValueError("No valid blade sections available for F1A.")
            all_sections_selected = bool(np.all(section_mask))
            selected_sections = (
                propeller.f1a_geometry_mask_tensor
                if all_sections_selected
                else torch.as_tensor(
                    section_mask,
                    dtype=torch.bool,
                    device=self.device,
                )
            )
            d_t_tensor = torch.as_tensor(
                d_t,
                dtype=self.dtype,
                device=self.device,
            )
            d_q_tensor = torch.as_tensor(
                d_q,
                dtype=self.dtype,
                device=self.device,
            )
            section_width = (
                propeller.section_width
                if all_sections_selected
                else propeller.section_width[selected_sections]
            )
            section_radius = (
                propeller.section_radius
                if all_sections_selected
                else propeller.section_radius[selected_sections]
            )
            axial_load = (
                (
                    d_t_tensor
                    if all_sections_selected
                    else d_t_tensor[selected_sections]
                )
                / section_width
                / self.nb
            )
            tangential_load = (
                (
                    d_q_tensor
                    if all_sections_selected
                    else d_q_tensor[selected_sections]
                )
                / section_width
                / section_radius
                / self.nb
            )
            loads = torch.stack(
                (
                    axial_load,
                    torch.zeros_like(axial_load),
                    -tangential_load,
                ),
                dim=-1,
            )

            self.section_mask = section_mask
            self._selected_sections = selected_sections
            self._uses_all_sections = all_sections_selected
            if all_sections_selected:
                self.r = propeller.section_radius
                self.dr = propeller.section_width
                self.area = propeller.section_area
                self.chord = propeller.section_chord
                self.twist_rad = propeller.section_twist_rad
                self.sweep = propeller.section_sweep
                self.rake = propeller.section_rake
                self.thickness_strength = propeller.f1a_thickness_strength
                self.dipole_strength = propeller.f1a_dipole_strength
            else:
                self.r = propeller.section_radius[selected_sections]
                self.dr = propeller.section_width[selected_sections]
                self.area = propeller.section_area[selected_sections]
                self.chord = propeller.section_chord[selected_sections]
                self.twist_rad = propeller.section_twist_rad[selected_sections]
                self.sweep = propeller.section_sweep[selected_sections]
                self.rake = propeller.section_rake[selected_sections]
                self.thickness_strength = propeller.f1a_thickness_strength[
                    selected_sections
                ]
                self.dipole_strength = propeller.f1a_dipole_strength[
                    selected_sections
                ]
        self.ns = int(self.r.shape[0])
        self.loads = loads

        self._initialize_loading()

        self.observers: torch.Tensor | None = None
        self.observer_times: torch.Tensor | None = None
        self.t: torch.Tensor | None = None
        self.observer_rotations: int | None = None
        self.observer_time_range: float | None = None
        self.num_observer_times: int | None = None
        self.sample_spacing: float | None = None
        self.p_m: torch.Tensor | None = None
        self.p_d: torch.Tensor | None = None
        self.p_tot: torch.Tensor | None = None
        self.frequencies: torch.Tensor | None = None
        self.spl: torch.Tensor | None = None
        self.spl_a: torch.Tensor | None = None
        self.ospl: torch.Tensor | None = None
        self.oaspl: torch.Tensor | None = None
        self.source_p_m: torch.Tensor | None = None
        self.source_p_d: torch.Tensor | None = None

    def _kinematics_matches(
        self,
        kinematics: Kinematics | None,
        *,
        source_times: torch.Tensor | None = None,
        section_radius: np.ndarray | torch.Tensor | None = None,
    ) -> bool:
        """Return whether cached kinematics match this F1A source grid."""
        if kinematics is None:
            return False
        if kinematics.propeller is not self.propeller:
            return False
        if not np.isclose(float(kinematics.rpm), float(self.rpm)):
            return False

        uses_custom_source_times = getattr(
            kinematics,
            "uses_custom_source_times",
            False,
        )
        if source_times is None:
            if uses_custom_source_times:
                return False
        else:
            if not uses_custom_source_times:
                return False
            if kinematics.source_times.shape != source_times.shape:
                return False
            if not torch.allclose(
                kinematics.source_times,
                source_times,
                rtol=1.0e-6,
                atol=1.0e-9,
            ):
                return False

        uses_custom_section_geometry = getattr(
            kinematics,
            "uses_custom_section_geometry",
            False,
        )
        if section_radius is None:
            return not uses_custom_section_geometry

        section_radius_tensor = torch.as_tensor(
            section_radius,
            dtype=self.dtype,
            device=self.device,
        )
        if kinematics.radial_positions.shape != section_radius_tensor.shape:
            return False
        return torch.allclose(
            kinematics.radial_positions,
            section_radius_tensor,
            rtol=1.0e-6,
            atol=1.0e-9,
        )

    def _required_file_path(
        self,
        value: object,
        *,
        name: str,
        suffix: str,
    ) -> Path:
        """Validate a required direct-input path."""
        if value is None:
            raise ValueError(
                f"{name} is required for direct F1A loadings and must be a "
                f"path to a {suffix} file."
            )
        if not isinstance(value, (str, PathLike)):
            raise TypeError(
                f"{name} must be a path to a {suffix} file for direct F1A "
                f"loadings; got {type(value).__name__}."
            )

        path = Path(value)
        if path.suffix.lower() != suffix:
            raise ValueError(
                f"{name} must point to a {suffix} file; got '{path}'."
            )
        if not path.is_file():
            raise FileNotFoundError(f"{name} file not found: {path}")
        return path

    def _load_direct_loadings(
        self,
        path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, ArrayLike]]:
        """Load the direct F1A dictionary format from ``.pt``."""
        loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, Mapping):
            raise TypeError(
                "Direct F1A loading files must contain a dictionary with "
                "'loadings' and 'sections' entries; got "
                f"{type(loaded).__name__}."
            )
        missing_keys = [
            key
            for key in ("loadings", "sections")
            if key not in loaded
        ]
        if missing_keys:
            raise ValueError(
                "Direct F1A loading dictionaries must contain "
                f"{', '.join(missing_keys)}."
            )

        loaded_loadings = loaded["loadings"]
        if not isinstance(loaded_loadings, (torch.Tensor, np.ndarray)):
            raise TypeError(
                "Direct F1A loading dictionary entry 'loadings' must be a "
                f"torch.Tensor or numpy.ndarray; got {type(loaded_loadings).__name__}."
            )
        loadings = torch.as_tensor(
            loaded_loadings,
            dtype=self.dtype,
            device=self.device,
        )
        loadings, source_times = self._split_direct_loading_table(
            loadings,
            path,
        )
        sections = loaded["sections"]
        if not isinstance(sections, Mapping):
            raise TypeError(
                "Direct F1A loading dictionary entry 'sections' must be a "
                f"dictionary; got {type(sections).__name__}."
            )
        return loadings, source_times, sections

    def _split_direct_loading_table(
        self,
        loadings: torch.Tensor,
        path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``(T, B, S, 4)`` into source times and force channels."""
        if loadings.ndim != 4 or loadings.shape[-1] != 4:
            raise ValueError(
                "Direct F1A loading dictionary entry 'loadings' must have "
                f"shape (T, B, S, 4); got {tuple(loadings.shape)} in '{path}'."
            )
        source_times = loadings[..., 0][:, 0, 0]
        if not self._is_broadcast_source_time_channel(
            loadings[..., 0],
            source_times,
        ):
            raise ValueError(
                "Direct F1A loading dictionary entry 'loadings' must store "
                "source time in channel 0, broadcast over blade and section."
            )
        return (
            loadings[..., 1:4].contiguous(),
            source_times.contiguous(),
        )

    def _is_broadcast_source_time_channel(
        self,
        channel: torch.Tensor,
        candidate: torch.Tensor,
    ) -> bool:
        if candidate.ndim != 1 or candidate.numel() == 0:
            return False
        if not bool(torch.isfinite(candidate).all().item()):
            return False
        if candidate.numel() > 1 and not bool(
            torch.all(candidate[1:] > candidate[:-1]).item()
        ):
            return False
        reference = candidate[:, None, None].expand_as(channel)
        return bool(torch.allclose(channel, reference, rtol=1.0e-6, atol=1.0e-8))

    def _trim_direct_loading_history(
        self,
        loadings: torch.Tensor,
        source_times: torch.Tensor,
        last_rotations: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep only the final direct-loading rotations, when requested."""
        if last_rotations is None:
            return loadings, source_times

        rotations = float(last_rotations)
        if not np.isfinite(rotations) or rotations <= 0.0:
            raise ValueError("last_rotations must be finite and greater than zero.")

        if loadings.ndim == 4 and loadings.shape[0] != source_times.numel():
            raise ValueError(
                "Unsteady direct F1A loadings must have the same number of "
                "time samples as source_times before last_rotations trimming; "
                f"got {loadings.shape[0]} and {source_times.numel()}."
            )

        rotation_period = 60.0 / self.rpm
        window = torch.as_tensor(
            rotations * rotation_period,
            dtype=self.dtype,
            device=self.device,
        )
        tolerance = torch.as_tensor(
            max(1.0e-12, 1.0e-7 * rotation_period),
            dtype=self.dtype,
            device=self.device,
        )
        start_time = source_times[-1] - window
        keep = source_times >= (start_time - tolerance)
        if not bool(keep.any().item()):
            raise ValueError(
                "last_rotations did not retain any direct F1A source samples."
            )

        trimmed_times = source_times[keep].contiguous()
        if loadings.ndim == 4:
            loadings = loadings[keep, ...].contiguous()
        return loadings, trimmed_times

    def _geometry_vector(
        self,
        geometry: Mapping[str, ArrayLike],
        name: str,
    ) -> np.ndarray:
        """Return a one-dimensional finite geometry vector."""
        if name not in geometry:
            raise ValueError(
                "Direct F1A loading dictionary entry 'sections' must contain "
                f"'{name}'."
            )
        values = np.asarray(geometry[name], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"sections['{name}'] must be one-dimensional.")
        if values.size == 0:
            raise ValueError(f"sections['{name}'] must not be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"sections['{name}'] must be finite.")
        return values

    def _direct_loading_geometry(
        self,
        sections: Mapping[str, ArrayLike],
    ) -> dict[str, np.ndarray]:
        """Build direct-loading section geometry from embedded section data."""
        r = self._geometry_vector(sections, "r_mid_m")
        dr = self._geometry_vector(sections, "dr_m")
        if dr.shape != r.shape:
            raise ValueError(
                "sections['dr_m'] must have the same shape as "
                f"sections['r_mid_m'] {r.shape}; got {dr.shape}."
            )
        if np.any(np.abs(r) <= 1.0e-12):
            raise ValueError("sections['r_mid_m'] must not contain zero radii.")
        if np.any(dr <= 0.0):
            raise ValueError("sections['dr_m'] must be greater than zero.")

        source = self.propeller.section_geometry_np
        source_r = np.asarray(source["r"], dtype=np.float64)
        source_mask = (
            np.isfinite(source_r)
            & np.isfinite(source["chord"])
            & np.isfinite(source["twist"])
            & np.isfinite(source["area"])
            & np.isfinite(source["sweep"])
            & np.isfinite(source["rake"])
        )
        if np.count_nonzero(source_mask) < 2:
            raise ValueError(
                "Propeller geometry needs at least two finite sections to "
                "resample direct F1A loading geometry."
            )

        order = np.argsort(source_r[source_mask])
        source_r_sorted = source_r[source_mask][order]
        if np.any(np.diff(source_r_sorted) <= 0.0):
            raise ValueError(
                "Propeller section radii must be strictly increasing for "
                "direct F1A loading geometry resampling."
            )

        radial_tolerance = max(
            1.0e-9,
            1.0e-6 * float(np.ptp(source_r_sorted)),
        )
        if (
            np.min(r) < source_r_sorted[0] - radial_tolerance
            or np.max(r) > source_r_sorted[-1] + radial_tolerance
        ):
            raise ValueError(
                "sections['r_mid_m'] must lie inside the propeller radial "
                f"range [{source_r_sorted[0]:.9g}, {source_r_sorted[-1]:.9g}]."
            )

        direct_geometry = {
            "r": r.copy(),
            "dr": dr.copy(),
        }
        for name in (
            "chord",
            "twist",
            "area",
            "sweep",
            "rake",
        ):
            source_values = np.asarray(source[name], dtype=np.float64)[source_mask]
            direct_geometry[name] = np.interp(
                r,
                source_r_sorted,
                source_values[order],
            )
        return direct_geometry

    def _select_direct_load_sections(
        self,
        loadings: torch.Tensor,
        geometry: dict[str, np.ndarray],
        source_times: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, np.ndarray], np.ndarray]:
        """Drop invalid direct-loading sections and align geometry to loads."""
        section_count = int(geometry["r"].shape[0])
        unsteady_shape = (
            int(source_times.numel()),
            self.propeller.n_blades,
            section_count,
            3,
        )
        if tuple(loadings.shape) != unsteady_shape:
            raise ValueError(
                "Direct F1A loadings must have shape "
                f"{unsteady_shape} after splitting time channel 0; "
                f"got {tuple(loadings.shape)}."
            )

        section_mask = np.ones(section_count, dtype=bool)
        for values in geometry.values():
            section_mask &= np.isfinite(values)

        finite_load_sections = (
            torch.isfinite(loadings).all(dim=(0, 1, 3)).cpu().numpy()
        )
        section_mask &= finite_load_sections
        if not np.any(section_mask):
            raise ValueError("No valid blade sections available for direct F1A.")

        section_mask_tensor = torch.as_tensor(
            section_mask,
            dtype=torch.bool,
            device=self.device,
        )
        selected_loadings = loadings[:, :, section_mask_tensor, :]
        selected_geometry = {
            name: values[section_mask]
            for name, values in geometry.items()
        }
        return selected_loadings.contiguous(), selected_geometry, section_mask

    def _initialize_loading(self) -> None:
        """Form global-frame load vectors and source-time derivatives."""
        if self.bemt is None:
            self._initialize_global_frame_loading()
            return
        self._initialize_blade_frame_loading()

    def _initialize_blade_frame_loading(self) -> None:
        """Rotate blade-frame loads and form their complete derivative."""
        blade_to_global_rotation = (
            self.kinematics.blade_to_global_rotation_matrix
        )
        if self.loads.ndim == 2:
            self.f0dot = torch.einsum(
                "tbij,sj->tbsi",
                blade_to_global_rotation,
                self.loads,
            ).contiguous()
            intrinsic_derivative = None
        else:
            self.f0dot = torch.einsum(
                "tbij,tbsj->tbsi",
                blade_to_global_rotation,
                self.loads,
            ).contiguous()
            if self.nt == 1:
                intrinsic_derivative = torch.zeros_like(self.loads)
            else:
                edge_order = 2 if self.nt >= 3 else 1
                intrinsic_derivative = torch.gradient(
                    self.loads,
                    spacing=(self.kinematics.source_times,),
                    dim=(0,),
                    edge_order=edge_order,
                )[0]

        self.f1dot = torch.linalg.cross(
            self.kinematics.omega_vec,
            self.f0dot,
            dim=-1,
        )
        if intrinsic_derivative is not None:
            self.f1dot = self.f1dot + torch.einsum(
                "tbij,tbsj->tbsi",
                blade_to_global_rotation,
                intrinsic_derivative,
            )
        self.f1dot = self.f1dot.contiguous()

    def _initialize_global_frame_loading(self) -> None:
        """Use inertial load vectors and differentiate them directly."""
        if self.loads.ndim == 2:
            self.f0dot = (
                self.loads[None, None, :, :]
                .expand(self.nt, self.nb, -1, -1)
                .contiguous()
            )
            self.f1dot = torch.zeros_like(self.f0dot)
            return

        self.f0dot = self.loads.contiguous()
        if self.nt == 1:
            self.f1dot = torch.zeros_like(self.f0dot)
            return

        edge_order = 2 if self.nt >= 3 else 1
        self.f1dot = torch.gradient(
            self.loads,
            spacing=(self.kinematics.source_times,),
            dim=(0,),
            edge_order=edge_order,
        )[0].contiguous()

    @torch.inference_mode()
    def run(
        self,
        observers: ArrayLike,
        observer_time_range: float,
        num_observer_times: int | None = None,
        *,
        observer_batch_size: int | None = None,
        retain_source_terms: bool = False,
    ) -> None:
        """Calculate and interpolate compact F1A pressure to observer time.

        Args:
            observers: Stationary observer coordinates, shape ``(O, 3)``.
            observer_time_range: Maximum duration of the observer-time grid.
                The actual duration is the largest whole number of rotor
                revolutions supported by this request and the source data.
            num_observer_times: Requested number of output samples. The
                implied samples per revolution are preserved. Defaults to
                ``T``.
            observer_batch_size: Optional number of observers to process at a
                time. Results are merged onto this object in observer order.
            retain_source_terms: Keep the uncombined ``(O, T, B, S)`` source
                pressure tensors for diagnostics.

        Results are stored in ``p_m``, ``p_d``, and ``p_tot`` with shape
        ``(O, T_observer)``. ``p_tot`` has its observer-wise mean removed.
        The actual grid is described by ``observer_rotations``,
        ``observer_time_range``, ``num_observer_times``, and
        ``sample_spacing``.
        ``frequencies`` has shape ``(F,)``; ``spl`` and ``spl_a`` have shape
        ``(O, F)``; ``ospl`` and ``oaspl`` have shape ``(O,)``.
        """
        observer_values = observer_tensor(
            observers,
            dtype=self.dtype,
            device=self.device,
        )
        requested_observer_count = (
            self.nt
            if num_observer_times is None
            else int(num_observer_times)
        )
        if requested_observer_count <= 0:
            raise ValueError("num_observer_times must be greater than zero.")
        requested_time_range = float(observer_time_range)
        if not np.isfinite(requested_time_range) or requested_time_range <= 0.0:
            raise ValueError("observer_time_range must be finite and positive.")

        batch_size = normalize_observer_batch_size(
            observer_batch_size,
            observer_count=int(observer_values.shape[0]),
        )
        if batch_size is None:
            self._run_observer_batch(
                observer_values,
                requested_time_range=requested_time_range,
                requested_observer_count=requested_observer_count,
                retain_source_terms=retain_source_terms,
            )
            return

        observer_rotations = self._observer_rotations_for_batches(
            observer_values,
            requested_time_range=requested_time_range,
            requested_observer_count=requested_observer_count,
            batch_size=batch_size,
        )
        batch_results = []
        for start in range(0, int(observer_values.shape[0]), batch_size):
            observer_batch = observer_values[start : start + batch_size]
            self._run_observer_batch(
                observer_batch,
                requested_time_range=requested_time_range,
                requested_observer_count=requested_observer_count,
                retain_source_terms=retain_source_terms,
                forced_observer_rotations=observer_rotations,
            )
            batch_results.append(
                self._snapshot_batch_results(
                    retain_source_terms=retain_source_terms,
                )
            )

        self._merge_batch_results(
            observer_values,
            batch_results,
            retain_source_terms=retain_source_terms,
        )

    def _reset_results(self) -> None:
        self.observer_times = None
        self.t = None
        self.observer_rotations = None
        self.observer_time_range = None
        self.num_observer_times = None
        self.sample_spacing = None
        self.p_m = None
        self.p_d = None
        self.p_tot = None
        self.frequencies = None
        self.spl = None
        self.spl_a = None
        self.ospl = None
        self.oaspl = None
        self.source_p_m = None
        self.source_p_d = None

    def _observer_rotations_for_batches(
        self,
        observers: torch.Tensor,
        *,
        requested_time_range: float,
        requested_observer_count: int,
        batch_size: int,
    ) -> int:
        observer_rotations = None
        for start in range(0, int(observers.shape[0]), batch_size):
            observer_batch = observers[start : start + batch_size]
            batch_observer_times = self._calculate_observer_times(observer_batch)
            batch_interpolation_times = (
                self._extend_periodic_steady_observer_times(
                    batch_observer_times,
                )
            )
            batch_rotations, _, _, _ = self._resolve_observer_time_grid(
                batch_observer_times,
                batch_interpolation_times,
                requested_time_range=requested_time_range,
                requested_observer_count=requested_observer_count,
            )
            observer_rotations = (
                batch_rotations
                if observer_rotations is None
                else min(observer_rotations, batch_rotations)
            )
        if observer_rotations is None:
            raise ValueError("observers must contain at least one observer.")
        return observer_rotations

    def _run_observer_batch(
        self,
        observers: torch.Tensor,
        *,
        requested_time_range: float,
        requested_observer_count: int,
        retain_source_terms: bool,
        forced_observer_rotations: int | None = None,
    ) -> None:
        self._reset_results()
        self.observers = observers

        source_p_m, source_p_d = self._calculate_source_pressures(self.observers)
        self.observer_times = self._calculate_observer_times(self.observers)
        source_pressure = torch.stack((source_p_m, source_p_d), dim=-1)
        interpolation_times, interpolation_pressure = (
            self._extend_periodic_steady_sources(
                self.observer_times,
                source_pressure,
            )
        )

        (
            self.observer_rotations,
            self.observer_time_range,
            self.num_observer_times,
            self.sample_spacing,
        ) = self._resolve_observer_time_grid(
            self.observer_times,
            interpolation_times,
            requested_time_range=requested_time_range,
            requested_observer_count=requested_observer_count,
            forced_observer_rotations=forced_observer_rotations,
        )
        latest_reception_start = torch.amax(
            self.observer_times[:, 0, :, :],
            dim=(1, 2),
        )
        observer_time_offsets = (
            torch.arange(
                self.num_observer_times,
                dtype=self.dtype,
                device=self.device,
            )
            * self.sample_spacing
        )
        self.t = latest_reception_start[:, None] + observer_time_offsets[None, :]

        observer_pressure = self._interpolate_sources(
            self.t,
            interpolation_times,
            interpolation_pressure,
        )
        self.p_m = observer_pressure[..., 0]
        self.p_d = observer_pressure[..., 1]
        pressure_total = self.p_m + self.p_d
        self.p_tot = pressure_total - torch.mean(
            pressure_total,
            dim=1,
            keepdim=True,
        )
        self.frequencies, self.spl = time_domain_to_spl_spectrum(
            self.p_tot,
            self.sample_spacing,
            self.environment.p_ref,
            time_dim=1,
        )
        self.spl_a = self.spl + a_weighting_db(
            self.frequencies
        )[None, :]
        self.ospl = spl_spectrum_to_overall_level(
            self.spl,
            self.frequencies,
            frequency_dim=1,
        )
        self.oaspl = spl_spectrum_to_overall_level(
            self.spl,
            self.frequencies,
            weighted=True,
            frequency_dim=1,
        )

        if retain_source_terms:
            self.source_p_m = source_p_m
            self.source_p_d = source_p_d
        else:
            self.source_p_m = None
            self.source_p_d = None

    def _resolve_observer_time_grid(
        self,
        observer_times: torch.Tensor,
        interpolation_times: torch.Tensor,
        *,
        requested_time_range: float,
        requested_observer_count: int,
        forced_observer_rotations: int | None = None,
    ) -> tuple[int, float, int, float]:
        latest_reception_start = torch.amax(
            observer_times[:, 0, :, :],
            dim=(1, 2),
        )
        earliest_reception_end = torch.amin(
            interpolation_times[:, -1, :, :],
            dim=(1, 2),
        )

        rotation_period = 60.0 / self.rpm
        requested_rotations = int(
            np.floor(requested_time_range / rotation_period + 1.0e-7)
        )
        if requested_rotations < 1:
            raise ValueError(
                "observer_time_range must contain at least one complete "
                f"rotor revolution ({rotation_period:.9g} s)."
            )

        requested_spacing = requested_time_range / requested_observer_count
        samples_per_rotation = max(
            1,
            int(np.rint(rotation_period / requested_spacing)),
        )
        sample_spacing = rotation_period / samples_per_rotation
        available_time_range = torch.amin(
            earliest_reception_end - latest_reception_start
        ).item()
        available_rotations = int(
            np.floor(
                (
                    available_time_range
                    + sample_spacing
                    + 1.0e-7 * rotation_period
                )
                / rotation_period
            )
        )

        if forced_observer_rotations is None:
            observer_rotations = min(requested_rotations, available_rotations)
        else:
            observer_rotations = int(forced_observer_rotations)
            if observer_rotations > requested_rotations:
                raise ValueError(
                    "forced observer rotations exceed the requested duration."
                )
            if observer_rotations > available_rotations:
                raise ValueError(
                    "The source-time data do not cover the forced observer "
                    "rotation count for this batch."
                )

        if observer_rotations < 1:
            raise ValueError(
                "The source-time data do not cover one complete observer "
                "revolution after the latest initial reception time. "
                "Provide at least one additional source revolution."
            )

        observer_time_range = observer_rotations * rotation_period
        num_observer_times = observer_rotations * samples_per_rotation
        return (
            observer_rotations,
            observer_time_range,
            num_observer_times,
            sample_spacing,
        )

    def _snapshot_batch_results(
        self,
        *,
        retain_source_terms: bool,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "observer_times": self.observer_times,
            "t": self.t,
            "observer_rotations": self.observer_rotations,
            "observer_time_range": self.observer_time_range,
            "num_observer_times": self.num_observer_times,
            "sample_spacing": self.sample_spacing,
            "p_m": self.p_m,
            "p_d": self.p_d,
            "p_tot": self.p_tot,
            "frequencies": self.frequencies,
            "spl": self.spl,
            "spl_a": self.spl_a,
            "ospl": self.ospl,
            "oaspl": self.oaspl,
        }
        if retain_source_terms:
            result["source_p_m"] = self.source_p_m
            result["source_p_d"] = self.source_p_d
        return result

    def _merge_batch_results(
        self,
        observers: torch.Tensor,
        batch_results: list[dict[str, object]],
        *,
        retain_source_terms: bool,
    ) -> None:
        first = batch_results[0]
        self.observers = observers
        self.observer_rotations = int(first["observer_rotations"])
        self.observer_time_range = float(first["observer_time_range"])
        self.num_observer_times = int(first["num_observer_times"])
        self.sample_spacing = float(first["sample_spacing"])
        self.frequencies = first["frequencies"]

        for result in batch_results[1:]:
            if int(result["observer_rotations"]) != self.observer_rotations:
                raise RuntimeError("F1A observer batches used different durations.")
            if int(result["num_observer_times"]) != self.num_observer_times:
                raise RuntimeError(
                    "F1A observer batches used different sample counts."
                )
            if not np.isclose(
                float(result["sample_spacing"]),
                self.sample_spacing,
                rtol=1.0e-7,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "F1A observer batches used different sample spacing."
                )
            if not torch.allclose(result["frequencies"], self.frequencies):
                raise RuntimeError(
                    "F1A observer batches produced different frequency grids."
                )

        self.observer_times = torch.cat(
            [result["observer_times"] for result in batch_results],
            dim=0,
        )
        self.t = torch.cat([result["t"] for result in batch_results], dim=0)
        self.p_m = torch.cat([result["p_m"] for result in batch_results], dim=0)
        self.p_d = torch.cat([result["p_d"] for result in batch_results], dim=0)
        self.p_tot = torch.cat(
            [result["p_tot"] for result in batch_results],
            dim=0,
        )
        self.spl = torch.cat([result["spl"] for result in batch_results], dim=0)
        self.spl_a = torch.cat(
            [result["spl_a"] for result in batch_results],
            dim=0,
        )
        self.ospl = torch.cat(
            [result["ospl"] for result in batch_results],
            dim=0,
        )
        self.oaspl = torch.cat(
            [result["oaspl"] for result in batch_results],
            dim=0,
        )
        if retain_source_terms:
            self.source_p_m = torch.cat(
                [result["source_p_m"] for result in batch_results],
                dim=0,
            )
            self.source_p_d = torch.cat(
                [result["source_p_d"] for result in batch_results],
                dim=0,
            )
        else:
            self.source_p_m = None
            self.source_p_d = None

    def _extend_periodic_steady_observer_times(
        self,
        observer_times: torch.Tensor,
    ) -> torch.Tensor:
        if self.loads.ndim != 2:
            return observer_times

        simulation = self.propeller.simulation
        samples_per_rotation = simulation.num_obs_times_per_rev
        source_duration = simulation.duration
        if source_duration is None or samples_per_rotation > self.nt:
            return observer_times

        periodic_slice = slice(0, samples_per_rotation)
        return torch.cat(
            (
                observer_times,
                observer_times[:, periodic_slice] + source_duration,
            ),
            dim=1,
        )

    def _extend_periodic_steady_sources(
        self,
        observer_times: torch.Tensor,
        source_pressure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one periodic source revolution for steady section loads."""
        if self.loads.ndim != 2:
            return observer_times, source_pressure

        simulation = self.propeller.simulation
        samples_per_rotation = simulation.num_obs_times_per_rev
        source_duration = simulation.duration
        if source_duration is None or samples_per_rotation > self.nt:
            return observer_times, source_pressure

        periodic_slice = slice(0, samples_per_rotation)
        extended_times = self._extend_periodic_steady_observer_times(observer_times)
        extended_pressure = torch.cat(
            (
                source_pressure,
                source_pressure[:, periodic_slice],
            ),
            dim=1,
        )
        return extended_times, extended_pressure

    def _calculate_source_pressures(
        self,
        observers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate uncombined source-time pressure terms."""
        x_obs = observers[:, None, None, None, :]

        # Kinematics are stored as (T, B, S, 3). Adding the observer axis
        # gives every source term the native order (O, T, B, S, ...).
        if self._uses_all_sections:
            y0dot = self.kinematics.section_position_global_frame.unsqueeze(0)
            y1dot = self.kinematics.section_vel.unsqueeze(0)
            y2dot = self.kinematics.section_acc.unsqueeze(0)
            y3dot = self.kinematics.section_jerk.unsqueeze(0)
        else:
            y0dot = self.kinematics.section_position_global_frame[
                :, :, self._selected_sections
            ].unsqueeze(0)
            y1dot = self.kinematics.section_vel[
                :, :, self._selected_sections
            ].unsqueeze(0)
            y2dot = self.kinematics.section_acc[
                :, :, self._selected_sections
            ].unsqueeze(0)
            y3dot = self.kinematics.section_jerk[
                :, :, self._selected_sections
            ].unsqueeze(0)

        rv = x_obs - y0dot
        r = torch.linalg.vector_norm(rv, dim=-1)
        R10 = r.reciprocal()
        rhat = rv * R10[..., None]

        rv1dot = -y1dot
        r1dot = torch.sum(rhat * rv1dot, dim=-1)

        rv2dot = -y2dot
        r2dot = (
            torch.sum(rv1dot * rv1dot, dim=-1)
            + torch.sum(rv * rv2dot, dim=-1)
            - r1dot.square()
        ) * R10

        rv3dot = -y3dot

        Mr = torch.sum((-rv1dot / self.a_inf) * rhat, dim=-1)

        R10_sq = R10.square()
        rhat1dot = rv1dot * R10[..., None] - (
            r1dot * R10_sq
        )[..., None] * rv
        Mr1dot = (
            torch.sum(rv2dot * rhat, dim=-1)
            + torch.sum(rv1dot * rhat1dot, dim=-1)
        ) / (-self.a_inf)

        rhat2dot = (
            (2.0 * r1dot.square() * R10_sq * R10)[..., None] * rv
            - (r2dot * R10_sq)[..., None] * rv
            - (2.0 * r1dot * R10_sq)[..., None] * rv1dot
            + R10[..., None] * rv2dot
        )
        Mr2dot = (
            torch.sum(rv3dot * rhat, dim=-1)
            + 2.0 * torch.sum(rv2dot * rhat1dot, dim=-1)
            + torch.sum(rv1dot * rhat2dot, dim=-1)
        ) / (-self.a_inf)

        # Rnm = r^(-n) * (1 - Mr)^(-m)
        R01 = (1.0 - Mr).reciprocal()
        R11 = R10 * R01
        R02 = R01.square()
        R21 = R11 * R10

        # Source-time derivatives for the compact loading/thickness terms.
        R10dot = -R10_sq * r1dot
        R01dot = R01.square() * Mr1dot
        R11_factor = -R10 * r1dot + R01 * Mr1dot
        R11dot = R11_factor * R11
        R11dotdot = (
            -R10dot * r1dot
            - R10 * r2dot
            + R01dot * Mr1dot
            + R01 * Mr2dot
        ) * R11 + R11_factor * R11dot

        C1A = R02 * R11dotdot + R01 * R01dot * R11dot
        p_m = C1A * self.thickness_strength[None, None, None, :]

        D1A = (R01 * R11)[..., None] * rhat
        E1A = R01[..., None] * (
            R11dot[..., None] * rhat + R11[..., None] * rhat1dot
        ) + (self.a_inf * R21)[..., None] * rhat

        f0dot = self.f0dot.unsqueeze(0)
        f1dot = self.f1dot.unsqueeze(0)
        p_d = (
            torch.sum(f1dot * D1A, dim=-1)
            + torch.sum(f0dot * E1A, dim=-1)
        ) * self.dipole_strength[None, None, None, :]

        return p_m, p_d

    def _calculate_observer_times(
        self,
        observers: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate source-to-observer arrival times in ``(O, T, B, S)``."""
        x_obs = observers[:, None, None, None, :]
        if self._uses_all_sections:
            source_position = (
                self.kinematics.section_position_global_frame.unsqueeze(0)
            )
        else:
            source_position = self.kinematics.section_position_global_frame[
                :, :, self._selected_sections
            ].unsqueeze(0)
        distance = torch.linalg.vector_norm(x_obs - source_position, dim=-1)
        return (
            self.kinematics.source_times[None, :, None, None]
            + distance / self.a_inf
        )

    def _interpolate_sources(
        self,
        output_times: torch.Tensor,
        observer_times: torch.Tensor,
        source_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate and sum ``(O, T, B, S, C)`` source terms."""
        n_observers, n_source_times, n_blades, n_sections = (
            observer_times.shape
        )
        n_output_times = output_times.shape[1]
        n_components = source_pressure.shape[-1]

        if n_source_times == 1:
            combined = source_pressure[:, 0].sum(dim=(1, 2))
            return combined[:, None, :].expand(-1, n_output_times, -1)

        target_times = (
            output_times[:, :, None, None]
            .expand(-1, -1, n_blades, n_sections)
        )

        lower_index = torch.zeros(
            target_times.shape,
            dtype=torch.long,
            device=self.device,
        )
        upper_index = torch.full_like(lower_index, n_source_times - 1)
        for _ in range((n_source_times - 1).bit_length()):
            active = upper_index - lower_index > 1
            middle_index = (lower_index + upper_index) // 2
            middle_time = torch.gather(
                observer_times,
                1,
                middle_index,
            )
            move_lower = active & (middle_time <= target_times)
            move_upper = active & ~move_lower
            lower_index = torch.where(
                move_lower,
                middle_index,
                lower_index,
            )
            upper_index = torch.where(
                move_upper,
                middle_index,
                upper_index,
            )

        time_0 = torch.gather(observer_times, 1, lower_index)
        time_1 = torch.gather(observer_times, 1, upper_index)
        pressure_index_0 = lower_index[..., None].expand(
            -1, -1, -1, -1, n_components
        )
        pressure_index_1 = upper_index[..., None].expand(
            -1, -1, -1, -1, n_components
        )
        pressure_0 = torch.gather(source_pressure, 1, pressure_index_0)
        pressure_1 = torch.gather(source_pressure, 1, pressure_index_1)

        weight = (target_times - time_0) / (time_1 - time_0 + 1.0e-12)
        interpolated = pressure_0 + weight[..., None] * (
            pressure_1 - pressure_0
        )
        return interpolated.sum(dim=(2, 3))
