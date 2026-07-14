from __future__ import annotations

from collections.abc import Mapping
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
        loadings: Optional path to the supported VPM ``.pt`` dictionary
            containing direct global-frame force per unit span and section
            geometry. The required format is
            ``{"loadings": tensor, "sections": dict}``, where ``loadings`` has
            shape ``(T, B, S, 4)`` with source time in channel 0 and force
            components in channels 1:4, and ``sections`` contains ``r_mid_m``
            and ``dr_m``.
        last_rotations: Optional number of final rotor revolutions to keep from
            direct loading files. Use this to discard startup transients before
            F1A builds kinematics and differentiates the load history.
    Observer-dependent source tensors use dimension order
    ``(O, T, B, S, ...)``.
    """

    def __init__(
        self,
        propeller: Propeller,
        bemt: BEMT | None = None,
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
        using_vpm_loadings = loadings is not None
        if using_vpm_loadings == (bemt is not None):
            raise ValueError("Specify exactly one of bemt or VPM loadings.")

        if using_vpm_loadings:
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

            loadings_path = Path(loadings)
            direct_loadings, direct_source_times, direct_sections = (
                self._load_direct_loadings(loadings_path)
            )
            direct_geometry = self._direct_loading_geometry(direct_sections)
            direct_loadings, direct_source_times = (
                self._trim_direct_loading_history(
                    direct_loadings,
                    direct_source_times,
                    last_rotations,
                )
            )
            loads = -direct_loadings

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
            self.n_source_times = self.kinematics.n_source_times
            self.n_blades = self.kinematics.n_blades

            self._selected_sections = torch.ones(
                direct_geometry["r"].shape,
                dtype=torch.bool,
                device=self.device,
            )
            self.section_radius = torch.as_tensor(
                direct_geometry["r"],
                dtype=self.dtype,
                device=self.device,
            )
            self.section_width = torch.as_tensor(
                direct_geometry["dr"],
                dtype=self.dtype,
                device=self.device,
            )
            self.section_area = torch.as_tensor(
                direct_geometry["area"],
                dtype=self.dtype,
                device=self.device,
            )
            self.section_chord = torch.as_tensor(
                direct_geometry["chord"],
                dtype=self.dtype,
                device=self.device,
            )
            self.section_twist_rad = torch.deg2rad(
                torch.as_tensor(
                    direct_geometry["twist"],
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.section_sweep = torch.as_tensor(
                direct_geometry["sweep"],
                dtype=self.dtype,
                device=self.device,
            )
            self.section_rake = torch.as_tensor(
                direct_geometry["rake"],
                dtype=self.dtype,
                device=self.device,
            )
            self.thickness_strength = (
                self.rho * self.section_area * self.section_width / (4.0 * np.pi)
            )
            self.dipole_strength = self.section_width / (4.0 * np.pi * self.a_inf)
        else:
            if last_rotations is not None:
                raise ValueError(
                    "last_rotations is only supported for direct F1A loadings."
                )
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
            self.n_source_times = self.kinematics.n_source_times
            self.n_blades = self.kinematics.n_blades

            section_mask = propeller.aerodynamic_section_mask.copy()
            section_thrust = np.array(
                solution["section_thrust"].values,
                dtype=np.float64,
                copy=True,
            )
            section_torque = np.array(
                solution["section_torque"].values,
                dtype=np.float64,
                copy=True,
            )
            section_mask &= np.isfinite(section_thrust) & np.isfinite(section_torque)
            selected_sections = torch.as_tensor(
                section_mask,
                dtype=torch.bool,
                device=self.device,
            )
            section_thrust_tensor = torch.as_tensor(
                section_thrust,
                dtype=self.dtype,
                device=self.device,
            )
            section_torque_tensor = torch.as_tensor(
                section_torque,
                dtype=self.dtype,
                device=self.device,
            )
            section_width = propeller.section_width[selected_sections]
            section_radius = propeller.section_radius[selected_sections]
            axial_load = (
                section_thrust_tensor[selected_sections]
                / section_width
                / self.n_blades
            )
            tangential_load = (
                section_torque_tensor[selected_sections]
                / section_width
                / section_radius
                / self.n_blades
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
            self.section_radius = propeller.section_radius[selected_sections]
            self.section_width = propeller.section_width[selected_sections]
            self.section_area = propeller.section_area[selected_sections]
            self.section_chord = propeller.section_chord[selected_sections]
            self.section_twist_rad = propeller.section_twist_rad[selected_sections]
            self.section_sweep = propeller.section_sweep[selected_sections]
            self.section_rake = propeller.section_rake[selected_sections]
            self.thickness_strength = propeller.f1a_thickness_strength[
                selected_sections
            ]
            self.dipole_strength = propeller.f1a_dipole_strength[
                selected_sections
            ]
        self.n_sections = int(self.section_radius.shape[0])
        self.loads = loads

        self._initialize_loading()

        self.observers: torch.Tensor | None = None
        self.source_reception_times: torch.Tensor | None = None
        self.observer_times: torch.Tensor | None = None
        self.observer_rotations: int | None = None
        self.observer_duration: float | None = None
        self.n_observer_times: int | None = None
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

        if source_times is None:
            if kinematics.uses_custom_source_times:
                return False
        else:
            if not kinematics.uses_custom_source_times:
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

        if section_radius is None:
            return not kinematics.uses_custom_section_geometry

        section_radius_tensor = torch.as_tensor(
            section_radius,
            dtype=self.dtype,
            device=self.device,
        )
        if kinematics.section_radius.shape != section_radius_tensor.shape:
            return False
        return torch.allclose(
            kinematics.section_radius,
            section_radius_tensor,
            rtol=1.0e-6,
            atol=1.0e-9,
        )

    def _load_direct_loadings(
        self,
        path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, ArrayLike]]:
        """Load the supported VPM direct-loading dictionary from ``.pt``."""
        loaded = torch.load(path, map_location="cpu")
        loadings = torch.as_tensor(
            loaded["loadings"],
            dtype=self.dtype,
            device=self.device,
        )
        return (
            loadings[..., 1:4].contiguous(),
            loadings[:, 0, 0, 0].contiguous(),
            loaded["sections"],
        )

    def _trim_direct_loading_history(
        self,
        loadings: torch.Tensor,
        source_times: torch.Tensor,
        last_rotations: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep only the final direct-loading rotations, when requested."""
        if last_rotations is None:
            return loadings, source_times

        rotation_period = 60.0 / self.rpm
        window = torch.as_tensor(
            float(last_rotations) * rotation_period,
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
        trimmed_times = source_times[keep].contiguous()
        if trimmed_times.numel() < 2:
            raise ValueError(
                "Direct VPM loadings must retain at least two source-time "
                "samples after last_rotations trimming."
            )
        loadings = loadings[keep, ...].contiguous()
        return loadings, trimmed_times

    def _direct_loading_geometry(
        self,
        sections: Mapping[str, ArrayLike],
    ) -> dict[str, np.ndarray]:
        """Build direct-loading section geometry from embedded section data."""
        r = np.asarray(sections["r_mid_m"], dtype=np.float64)
        dr = np.asarray(sections["dr_m"], dtype=np.float64)
        source = self.propeller.section_geometry_np
        source_r = np.asarray(source["r"], dtype=np.float64)
        direct_geometry = {
            "r": r.copy(),
            "dr": dr.copy(),
        }
        # VPM owns the radial bins. The propeller supplies missing geometry
        # values at those radii, including the acoustic thickness area.
        for name in (
            "chord",
            "twist",
            "area",
            "sweep",
            "rake",
        ):
            direct_geometry[name] = np.interp(
                r,
                source_r,
                np.asarray(source[name], dtype=np.float64),
            )
        return direct_geometry

    def _initialize_loading(self) -> None:
        """Form global-frame load vectors and source-time derivatives."""
        if self.bemt is None:
            self._initialize_unsteady_global_frame_loading()
            return
        self._initialize_steady_blade_frame_loading()

    def _initialize_steady_blade_frame_loading(self) -> None:
        """Rotate steady BEMT blade-frame loads and differentiate by rotation."""
        self.global_loads = torch.einsum(
            "tbij,sj->tbsi",
            self.kinematics.blade_to_global_rotation_matrix,
            self.loads,
        ).contiguous()
        self.global_load_derivative = torch.linalg.cross(
            self.kinematics.omega_vec,
            self.global_loads,
            dim=-1,
        ).contiguous()

    def _initialize_unsteady_global_frame_loading(self) -> None:
        """Use unsteady VPM global-frame loads and differentiate them directly."""
        self.global_loads = self.loads.contiguous()
        self.global_load_derivative = torch.gradient(
            self.loads,
            spacing=(self.kinematics.source_times,),
            dim=(0,),
            edge_order=2,
        )[0].contiguous()

    @torch.inference_mode()
    def run(
        self,
        observers: ArrayLike,
        observer_duration: float,
        n_observer_times: int | None = None,
        *,
        observer_batch_size: int | None = None,
        retain_source_terms: bool = False,
    ) -> None:
        """Calculate and interpolate compact F1A pressure to observer time.

        Args:
            observers: Stationary observer coordinates, shape ``(O, 3)``.
            observer_duration: Maximum duration of the observer-time grid.
                The actual duration is the largest whole number of rotor
                revolutions supported by this request and the source data.
            n_observer_times: Requested number of output samples. The
                implied samples per revolution are preserved. Defaults to
                ``T``.
            observer_batch_size: Optional number of observers to process at a
                time. Results are merged onto this object in observer order.
            retain_source_terms: Keep the uncombined ``(O, T, B, S)`` source
                pressure tensors for diagnostics.

        Results are stored in ``p_m``, ``p_d``, and ``p_tot`` with shape
        ``(O, T_observer)``. ``p_tot`` has its observer-wise mean removed.
        The actual grid is described by ``observer_rotations``,
        ``observer_duration``, ``n_observer_times``, and
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
            self.n_source_times
            if n_observer_times is None
            else int(n_observer_times)
        )
        if requested_observer_count <= 0:
            raise ValueError("n_observer_times must be greater than zero.")
        requested_duration = float(observer_duration)
        if not np.isfinite(requested_duration) or requested_duration <= 0.0:
            raise ValueError("observer_duration must be finite and positive.")

        batch_size = normalize_observer_batch_size(
            observer_batch_size,
            observer_count=int(observer_values.shape[0]),
        )
        if batch_size is None:
            self._run_observer_batch(
                observer_values,
                requested_duration=requested_duration,
                requested_observer_count=requested_observer_count,
                retain_source_terms=retain_source_terms,
            )
            return

        observer_rotations = self._observer_rotations_for_batches(
            observer_values,
            requested_duration=requested_duration,
            requested_observer_count=requested_observer_count,
            batch_size=batch_size,
        )
        batch_results = []
        for start in range(0, int(observer_values.shape[0]), batch_size):
            observer_batch = observer_values[start : start + batch_size]
            self._run_observer_batch(
                observer_batch,
                requested_duration=requested_duration,
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
        self.source_reception_times = None
        self.observer_times = None
        self.observer_rotations = None
        self.observer_duration = None
        self.n_observer_times = None
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
        requested_duration: float,
        requested_observer_count: int,
        batch_size: int,
    ) -> int:
        observer_rotations = None
        for start in range(0, int(observers.shape[0]), batch_size):
            observer_batch = observers[start : start + batch_size]
            batch_reception_times = self._calculate_source_reception_times(
                observer_batch
            )
            batch_interpolation_times = (
                self._extend_periodic_steady_reception_times(
                    batch_reception_times,
                )
            )
            batch_rotations, _, _, _ = self._resolve_observer_time_grid(
                batch_reception_times,
                batch_interpolation_times,
                requested_duration=requested_duration,
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
        requested_duration: float,
        requested_observer_count: int,
        retain_source_terms: bool,
        forced_observer_rotations: int | None = None,
    ) -> None:
        self._reset_results()
        self.observers = observers

        source_p_m, source_p_d = self._calculate_source_pressures(self.observers)
        self.source_reception_times = self._calculate_source_reception_times(
            self.observers
        )
        source_pressure = torch.stack((source_p_m, source_p_d), dim=-1)
        interpolation_times, interpolation_pressure = (
            self._extend_periodic_steady_sources(
                self.source_reception_times,
                source_pressure,
            )
        )

        (
            self.observer_rotations,
            self.observer_duration,
            self.n_observer_times,
            self.sample_spacing,
        ) = self._resolve_observer_time_grid(
            self.source_reception_times,
            interpolation_times,
            requested_duration=requested_duration,
            requested_observer_count=requested_observer_count,
            forced_observer_rotations=forced_observer_rotations,
        )
        latest_reception_start = torch.amax(
            self.source_reception_times[:, 0, :, :],
            dim=(1, 2),
        )
        observer_time_offsets = (
            torch.arange(
                self.n_observer_times,
                dtype=self.dtype,
                device=self.device,
            )
            * self.sample_spacing
        )
        self.observer_times = (
            latest_reception_start[:, None] + observer_time_offsets[None, :]
        )

        observer_pressure = self._interpolate_sources(
            self.observer_times,
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
        source_reception_times: torch.Tensor,
        interpolation_times: torch.Tensor,
        *,
        requested_duration: float,
        requested_observer_count: int,
        forced_observer_rotations: int | None = None,
    ) -> tuple[int, float, int, float]:
        latest_reception_start = torch.amax(
            source_reception_times[:, 0, :, :],
            dim=(1, 2),
        )
        earliest_reception_end = torch.amin(
            interpolation_times[:, -1, :, :],
            dim=(1, 2),
        )

        rotation_period = 60.0 / self.rpm
        requested_rotations = int(
            np.floor(requested_duration / rotation_period + 1.0e-7)
        )
        if requested_rotations < 1:
            raise ValueError(
                "observer_duration must contain at least one complete "
                f"rotor revolution ({rotation_period:.9g} s)."
            )

        requested_spacing = requested_duration / requested_observer_count
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

        observer_duration = observer_rotations * rotation_period
        n_observer_times = observer_rotations * samples_per_rotation
        return (
            observer_rotations,
            observer_duration,
            n_observer_times,
            sample_spacing,
        )

    def _snapshot_batch_results(
        self,
        *,
        retain_source_terms: bool,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "source_reception_times": self.source_reception_times,
            "observer_times": self.observer_times,
            "observer_rotations": self.observer_rotations,
            "observer_duration": self.observer_duration,
            "n_observer_times": self.n_observer_times,
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
        self.observer_duration = float(first["observer_duration"])
        self.n_observer_times = int(first["n_observer_times"])
        self.sample_spacing = float(first["sample_spacing"])
        self.frequencies = first["frequencies"]

        self.source_reception_times = torch.cat(
            [result["source_reception_times"] for result in batch_results],
            dim=0,
        )
        self.observer_times = torch.cat(
            [result["observer_times"] for result in batch_results],
            dim=0,
        )
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

    def _extend_periodic_steady_reception_times(
        self,
        source_reception_times: torch.Tensor,
    ) -> torch.Tensor:
        if self.bemt is None:
            return source_reception_times

        simulation = self.propeller.simulation
        samples_per_rotation = simulation.timesteps_per_revolution
        source_duration = simulation.duration

        periodic_slice = slice(0, samples_per_rotation)
        return torch.cat(
            (
                source_reception_times,
                source_reception_times[:, periodic_slice] + source_duration,
            ),
            dim=1,
        )

    def _extend_periodic_steady_sources(
        self,
        source_reception_times: torch.Tensor,
        source_pressure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one periodic source revolution for steady BEMT section loads."""
        if self.bemt is None:
            return source_reception_times, source_pressure

        simulation = self.propeller.simulation
        samples_per_rotation = simulation.timesteps_per_revolution

        periodic_slice = slice(0, samples_per_rotation)
        extended_times = self._extend_periodic_steady_reception_times(
            source_reception_times
        )
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
        y0dot = self.kinematics.section_position_global_frame[
            :, :, self._selected_sections
        ].unsqueeze(0)
        y1dot = self.kinematics.section_velocity_global_frame[
            :, :, self._selected_sections
        ].unsqueeze(0)
        y2dot = self.kinematics.section_acceleration_global_frame[
            :, :, self._selected_sections
        ].unsqueeze(0)
        y3dot = self.kinematics.section_jerk_global_frame[
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

        global_loads = self.global_loads.unsqueeze(0)
        global_load_derivative = self.global_load_derivative.unsqueeze(0)
        p_d = (
            torch.sum(global_load_derivative * D1A, dim=-1)
            + torch.sum(global_loads * E1A, dim=-1)
        ) * self.dipole_strength[None, None, None, :]

        return p_m, p_d

    def _calculate_source_reception_times(
        self,
        observers: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate source-to-observer arrival times in ``(O, T, B, S)``."""
        x_obs = observers[:, None, None, None, :]
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
        source_reception_times: torch.Tensor,
        source_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate and sum ``(O, T, B, S, C)`` source terms."""
        _, n_source_times, n_blades, n_sections = source_reception_times.shape
        n_components = source_pressure.shape[-1]

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
                source_reception_times,
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

        time_0 = torch.gather(source_reception_times, 1, lower_index)
        time_1 = torch.gather(source_reception_times, 1, upper_index)
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
