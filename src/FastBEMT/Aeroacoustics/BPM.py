from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ._common import ArrayLike, normalize_observer_batch_size, observer_tensor
from ._bpm_components import lbl, tbl, teb, ti, tv
from .Utils import (
    a_weighting_db,
    power_ratio_to_spl,
    spl_spectrum_to_overall_level,
)
from ..Kinematics import Kinematics

if TYPE_CHECKING:
    from ..Aerodynamics.BEMT import BEMT
    from ..Propeller import Propeller


class BPM:
    """Brooks-Pope-Marcolini broadband noise model.

    The object consumes a Propeller, BEMT solution, and optional shared
    Kinematics object. Calling ``run`` stores third-octave component spectra,
    total SPL, and total OSPL/OASPL on the instance.

    Args:
        propeller: Propeller containing geometry and simulation settings.
        bemt: BEMT analysis supplying the aerodynamic section solution.
        kinematics: Optional matching rotating-section kinematics.
        rpm: Operating speed used to select a multi-point BEMT solution.
        v_inf: Freestream velocity used with ``rpm`` for solution selection.
        turbulence_length_scale: Turbulent inflow length scale in meters.
        turbulence_intensity: Turbulent inflow intensity as a fraction.
        trailing_edge_offset: Distance from each acoustic section reference
            point to its trailing-edge source in meters. Defaults to half the
            local chord.
    """

    def __init__(
        self,
        propeller: Propeller,
        bemt: BEMT,
        *,
        kinematics: Kinematics | None = None,
        rpm: float | None = None,
        v_inf: float | None = None,
        turbulence_length_scale: float = 1.0,
        turbulence_intensity: float = 0.01,
        trailing_edge_offset: ArrayLike | None = None,
    ) -> None:
        self.propeller = propeller
        self.environment = propeller.environment
        self.bemt = bemt
        self.device = torch.device(propeller.device)
        self.dtype = propeller.dtype
        self.turbulence_length_scale = float(turbulence_length_scale)
        self.turbulence_intensity = float(turbulence_intensity)
        self.rpm, self.v_inf = bemt.resolve_operating_point(rpm, v_inf)
        self.rotation_period = 60.0 / self.rpm

        if kinematics is not None:
            if not self._kinematics_matches(kinematics):
                raise ValueError(
                    "The supplied Kinematics object must belong to this propeller "
                    "and operating RPM."
                )
            self.kinematics = kinematics
        elif self._kinematics_matches(propeller.kinematics):
            self.kinematics = propeller.kinematics
        else:
            self.kinematics = Kinematics(propeller, rpm=self.rpm)
        propeller.kinematics = self.kinematics

        solution = bemt.solution_for(self.rpm, self.v_inf)
        self.section_mask = self._valid_section_mask(solution)
        self._selected_sections = torch.as_tensor(
            self.section_mask,
            dtype=torch.bool,
            device=self.device,
        )
        self.section_indices = np.flatnonzero(self.section_mask)
        self.n_sections = int(self.section_indices.shape[0])
        self.n_blades = self.kinematics.n_blades
        self.n_source_times = self._source_time_count()
        self.source_times = self.kinematics.source_times[
            : self.n_source_times
        ].to(dtype=self.dtype, device=self.device)

        geometry = propeller.section_geometry_np
        self.frequencies = propeller.third_octave_freqs.to(
            dtype=self.dtype,
            device=self.device,
        )
        self.section_radius = self._section_tensor(geometry["r"])
        self.section_width = self._section_tensor(geometry["dr"])
        self.section_chord = self._section_tensor(geometry["chord"])
        self.angle_of_attack_deg = self._solution_tensor(
            solution,
            "angle_of_attack_deg",
        )
        self.relative_velocity = self._solution_tensor(
            solution,
            "relative_velocity",
        )
        self.reynolds_number = self._solution_tensor(solution, "reynolds_number")
        self.mach_number = self._solution_tensor(solution, "mach_number")
        self.upper_displacement_thickness = self._solution_tensor(
            solution,
            "upper_displacement_thickness",
        )
        self.lower_displacement_thickness = self._solution_tensor(
            solution,
            "lower_displacement_thickness",
        )
        self.boat_tail_angle_deg = self._section_tensor(
            geometry["boat_tail_angle"]
        )
        self.a_inf = propeller.a_inf_tensor
        self.rho = propeller.rho_tensor
        self.trailing_edge_offset = self._trailing_edge_offset(
            trailing_edge_offset,
            geometry["chord"],
        )

        self.observers: torch.Tensor | None = None
        self.source_reception_times: torch.Tensor | None = None
        self.observer_times: torch.Tensor | None = None
        self.observer_rotations: int | None = None
        self.observer_duration: float | None = None
        self.n_observer_times: int | None = None
        self.sample_spacing: float | None = None
        self.source_rotation_repetitions: int | None = None
        self.base_val_te: torch.Tensor | None = None
        self.base_val_le: torch.Tensor | None = None
        self.base_val_low: torch.Tensor | None = None
        self.component_power_ratio: dict[str, torch.Tensor] = {}
        self.component_spl: dict[str, torch.Tensor] = {}
        self.spl: torch.Tensor | None = None
        self.spl_a: torch.Tensor | None = None
        self.ospl: torch.Tensor | None = None
        self.oaspl: torch.Tensor | None = None
        self.source_component_power_ratio: dict[str, torch.Tensor] | None = None

    def _kinematics_matches(self, kinematics: Kinematics | None) -> bool:
        if kinematics is None:
            return False
        if kinematics.propeller is not self.propeller:
            return False
        return np.isclose(float(kinematics.rpm), float(self.rpm))

    def _valid_section_mask(self, solution: object) -> np.ndarray:
        finite_solution = np.ones(self.propeller.n_sections, dtype=bool)
        for column in (
            "angle_of_attack_deg",
            "relative_velocity",
            "reynolds_number",
            "mach_number",
            "upper_displacement_thickness",
            "lower_displacement_thickness",
        ):
            finite_solution &= np.isfinite(
                np.asarray(solution[column].to_numpy(), dtype=np.float64)
            )
        return self.propeller.aerodynamic_section_mask & finite_solution

    def _source_time_count(self) -> int:
        return int(
            self.propeller.simulation.source_times_one_revolution.shape[0]
        )

    def _section_tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            values[self.section_mask],
            dtype=self.dtype,
            device=self.device,
        )

    def _solution_tensor(self, solution: object, column: str) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(solution[column].to_numpy(), dtype=np.float64)[
                self.section_mask
            ],
            dtype=self.dtype,
            device=self.device,
        )

    @staticmethod
    def _to_numpy(value: ArrayLike) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    def _selected_optional_values(
        self,
        value: ArrayLike,
        *,
        name: str,
        full_chord: np.ndarray,
    ) -> np.ndarray:
        selected_chord = full_chord[self.section_mask]
        values = self._to_numpy(value)
        if values.ndim == 0:
            return np.full_like(selected_chord, float(values), dtype=np.float64)
        if values.shape == full_chord.shape:
            return values[self.section_mask].astype(np.float64, copy=False)
        if values.shape == selected_chord.shape:
            return values.astype(np.float64, copy=False)
        raise ValueError(
            f"{name} must be scalar, full-section shape {full_chord.shape}, "
            f"or selected-section shape {selected_chord.shape}; got {values.shape}."
        )

    def _trailing_edge_offset(
        self,
        trailing_edge_offset: ArrayLike | None,
        full_chord: np.ndarray,
    ) -> torch.Tensor:
        selected_chord = full_chord[self.section_mask].astype(np.float64, copy=False)
        if trailing_edge_offset is not None:
            values = self._selected_optional_values(
                trailing_edge_offset,
                name="trailing_edge_offset",
                full_chord=full_chord,
            )
        else:
            values = 0.5 * selected_chord
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    @torch.inference_mode()
    def run(
        self,
        observers: ArrayLike,
        observer_duration: float | None = None,
        n_observer_times: int | None = None,
        *,
        turbulence_length_scale: float | None = None,
        turbulence_intensity: float | None = None,
        observer_batch_size: int | None = None,
        retain_source_terms: bool = False,
    ) -> None:
        """Compute BPM component and total third-octave spectra.

        Args:
            observers: Stationary observer coordinates, shape ``(O, 3)``.
            observer_duration: Optional requested observer-time duration.
            n_observer_times: Optional requested observer sample count.
            turbulence_length_scale: Override the turbulent length scale.
            turbulence_intensity: Override the turbulence intensity.
            observer_batch_size: Optional number of observers to process at a
                time. Results are merged onto this object in observer order.
            retain_source_terms: Keep uncombined source component powers.
        """
        observer_values = observer_tensor(
            observers,
            dtype=self.dtype,
            device=self.device,
        )
        batch_size = normalize_observer_batch_size(
            observer_batch_size,
            observer_count=int(observer_values.shape[0]),
        )
        if batch_size is None:
            self._run_observer_batch(
                observer_values,
                observer_duration=observer_duration,
                n_observer_times=n_observer_times,
                turbulence_length_scale=turbulence_length_scale,
                turbulence_intensity=turbulence_intensity,
                retain_source_terms=retain_source_terms,
            )
            return

        batch_results = []
        for start in range(0, int(observer_values.shape[0]), batch_size):
            observer_batch = observer_values[start : start + batch_size]
            self._run_observer_batch(
                observer_batch,
                observer_duration=observer_duration,
                n_observer_times=n_observer_times,
                turbulence_length_scale=turbulence_length_scale,
                turbulence_intensity=turbulence_intensity,
                retain_source_terms=retain_source_terms,
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

    def _run_observer_batch(
        self,
        observers: torch.Tensor,
        *,
        observer_duration: float | None,
        n_observer_times: int | None,
        turbulence_length_scale: float | None,
        turbulence_intensity: float | None,
        retain_source_terms: bool,
    ) -> None:
        self._reset_results()
        self.observers = observers

        source_positions, source_beta = self._source_trajectory()
        source_reception_times = self._calculate_source_reception_times(
            self.observers,
            source_positions,
        )
        self.source_reception_times = source_reception_times.permute(
            3,
            0,
            2,
            1,
        ).contiguous()

        (
            interpolation_times,
            interpolation_positions,
            interpolation_beta,
        ) = self._extend_periodic_sources(
            source_reception_times,
            source_positions,
            source_beta,
        )
        output_times = self._observer_output_times(
            interpolation_times,
            observer_duration,
            n_observer_times,
        )
        self.observer_times = output_times.T.contiguous()

        positions_at_observer = self._interpolate_positions(
            output_times,
            interpolation_times,
            interpolation_positions,
        )
        beta_at_observer = self._interpolate_positions(
            output_times,
            interpolation_times,
            interpolation_beta,
        )
        self._set_base_values(
            self.observers,
            positions_at_observer,
            beta_at_observer,
        )

        source_components = self.compute_noise_components(
            turbulence_length_scale=(
                self.turbulence_length_scale
                if turbulence_length_scale is None
                else float(turbulence_length_scale)
            ),
            turbulence_intensity=(
                self.turbulence_intensity
                if turbulence_intensity is None
                else float(turbulence_intensity)
            ),
        )
        if retain_source_terms:
            self.source_component_power_ratio = source_components

        self.component_power_ratio = {
            name: values.sum(dim=(2, 3))
            for name, values in source_components.items()
        }
        band_power = {
            name: values.mean(dim=1).T.contiguous()
            for name, values in self.component_power_ratio.items()
        }
        self.component_spl = {
            name: power_ratio_to_spl(values)
            for name, values in band_power.items()
        }

        total_band_power = torch.stack(tuple(band_power.values())).sum(dim=0)

        self.spl = power_ratio_to_spl(total_band_power)
        self.spl_a = self.spl + a_weighting_db(self.frequencies)[None, :]
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
            "source_rotation_repetitions": self.source_rotation_repetitions,
            "base_val_te": self.base_val_te,
            "base_val_le": self.base_val_le,
            "base_val_low": self.base_val_low,
            "component_power_ratio": self.component_power_ratio,
            "component_spl": self.component_spl,
            "spl": self.spl,
            "spl_a": self.spl_a,
            "ospl": self.ospl,
            "oaspl": self.oaspl,
            "frequencies": self.frequencies,
        }
        if retain_source_terms:
            result["source_component_power_ratio"] = (
                self.source_component_power_ratio
            )
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
        self.source_rotation_repetitions = max(
            int(result["source_rotation_repetitions"])
            for result in batch_results
        )
        self.frequencies = first["frequencies"]

        self.source_reception_times = torch.cat(
            [result["source_reception_times"] for result in batch_results],
            dim=0,
        )
        self.observer_times = torch.cat(
            [result["observer_times"] for result in batch_results],
            dim=0,
        )
        self.base_val_te = torch.cat(
            [result["base_val_te"] for result in batch_results],
            dim=3,
        )
        self.base_val_le = torch.cat(
            [result["base_val_le"] for result in batch_results],
            dim=3,
        )
        self.base_val_low = torch.cat(
            [result["base_val_low"] for result in batch_results],
            dim=3,
        )

        component_names = tuple(first["component_power_ratio"])
        self.component_power_ratio = {
            name: torch.cat(
                [
                    result["component_power_ratio"][name]
                    for result in batch_results
                ],
                dim=2,
            )
            for name in component_names
        }
        self.component_spl = {
            name: torch.cat(
                [result["component_spl"][name] for result in batch_results],
                dim=0,
            )
            for name in component_names
        }
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
            self.source_component_power_ratio = {
                name: torch.cat(
                    [
                        result["source_component_power_ratio"][name]
                        for result in batch_results
                    ],
                    dim=4,
                )
                for name in component_names
            }
        else:
            self.source_component_power_ratio = None

    def _reset_results(self) -> None:
        self.observers = None
        self.source_reception_times = None
        self.observer_times = None
        self.observer_rotations = None
        self.observer_duration = None
        self.n_observer_times = None
        self.sample_spacing = None
        self.source_rotation_repetitions = None
        self.base_val_te = None
        self.base_val_le = None
        self.base_val_low = None
        self.component_power_ratio = {}
        self.component_spl = {}
        self.spl = None
        self.spl_a = None
        self.ospl = None
        self.oaspl = None
        self.source_component_power_ratio = None

    def _source_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        reference_position = self.kinematics.section_position_global_frame[
            : self.n_source_times,
            :,
            self._selected_sections,
            :,
        ]
        blade_to_global = self.kinematics.blade_to_global_rotation_matrix[
            : self.n_source_times
        ]
        airfoil_to_blade = self.kinematics.airfoil_to_blade_rotation_matrix[
            self._selected_sections
        ]

        offset_airfoil = torch.zeros(
            (self.n_sections, 3),
            dtype=self.dtype,
            device=self.device,
        )
        offset_airfoil[:, 0] = self.trailing_edge_offset
        offset_blade = torch.einsum(
            "sij,sj->si",
            airfoil_to_blade,
            offset_airfoil,
        )
        offset_global = torch.einsum(
            "tbij,sj->tbsi",
            blade_to_global,
            offset_blade,
        )
        source_position = (reference_position + offset_global).permute(
            0,
            2,
            1,
            3,
        ).contiguous()
        source_beta = self.kinematics.blade_angles_rad[
            : self.n_source_times,
            None,
            :,
            None,
        ]
        source_beta = source_beta.expand(
            -1,
            self.n_sections,
            -1,
            -1,
        ).contiguous()
        return source_position, source_beta

    def _calculate_source_reception_times(
        self,
        observers: torch.Tensor,
        source_positions: torch.Tensor,
    ) -> torch.Tensor:
        distance = torch.linalg.vector_norm(
            observers[None, None, None, :, :] - source_positions[:, :, :, None, :],
            dim=-1,
        )
        return self.source_times[:, None, None, None] + distance / self.a_inf

    def _extend_periodic_sources(
        self,
        source_reception_times: torch.Tensor,
        source_positions: torch.Tensor,
        source_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.source_rotation_repetitions = 2
        return (
            torch.cat(
                (
                    source_reception_times,
                    source_reception_times + self.rotation_period,
                ),
                dim=0,
            ),
            torch.cat((source_positions, source_positions), dim=0),
            torch.cat((source_beta, source_beta + 2.0 * np.pi), dim=0),
        )

    def _observer_output_times(
        self,
        source_reception_times: torch.Tensor,
        observer_duration: float | None,
        n_observer_times: int | None,
    ) -> torch.Tensor:
        requested_duration = (
            self.propeller.simulation.observer_duration
            if observer_duration is None
            else float(observer_duration)
        )
        if requested_duration is None:
            requested_duration = self.rotation_period

        if n_observer_times is None:
            samples_per_rotation = int(
                self.propeller.simulation.timesteps_per_revolution
            )
        else:
            requested_observer_count = int(n_observer_times)
            requested_spacing = requested_duration / requested_observer_count
            samples_per_rotation = max(
                1,
                int(np.rint(self.rotation_period / requested_spacing)),
            )
        sample_spacing = self.rotation_period / samples_per_rotation

        latest_reception_start = torch.amax(
            source_reception_times[0],
            dim=(0, 1),
        )
        self.observer_rotations = 1
        self.observer_duration = self.rotation_period
        self.n_observer_times = samples_per_rotation
        self.sample_spacing = sample_spacing
        offsets = (
            torch.arange(
                self.n_observer_times,
                dtype=self.dtype,
                device=self.device,
            )
            * self.sample_spacing
        )
        return latest_reception_start[None, :] + offsets[:, None]

    def _interpolate_positions(
        self,
        x_new: torch.Tensor,
        x_old: torch.Tensor,
        y_old: torch.Tensor,
    ) -> torch.Tensor:
        n_source_times, n_sections, n_blades, _ = x_old.shape
        n_steps, n_observers = x_new.shape

        x_old_p = x_old.permute(3, 1, 2, 0).contiguous()
        x_new_p = (
            x_new.T.view(n_observers, 1, 1, n_steps)
            .expand(-1, n_sections, n_blades, -1)
            .contiguous()
        )
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, n_source_times - 1)

        coord_count = y_old.shape[-1]
        y_old_p = y_old.permute(3, 1, 2, 0).contiguous()
        y_old_exp = y_old_p.unsqueeze(1).expand(
            -1,
            n_observers,
            -1,
            -1,
            -1,
        )
        idx_exp = idx.unsqueeze(0).expand(coord_count, -1, -1, -1, -1)

        y0 = torch.gather(y_old_exp, 4, idx_exp - 1)
        y1 = torch.gather(y_old_exp, 4, idx_exp)
        x0 = torch.gather(x_old_p, 3, idx - 1)
        x1 = torch.gather(x_old_p, 3, idx)
        t = (x_new_p - x0) / (x1 - x0 + 1.0e-12)
        return (y0 + t.unsqueeze(0) * (y1 - y0)).permute(4, 2, 3, 1, 0)

    def _global_to_blade_rotation(self, beta: torch.Tensor) -> torch.Tensor:
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        zeros = torch.zeros_like(cos_beta)
        ones = torch.ones_like(cos_beta)
        return torch.stack(
            [
                torch.stack([ones, zeros, zeros], dim=-1),
                torch.stack([zeros, cos_beta, sin_beta], dim=-1),
                torch.stack([zeros, -sin_beta, cos_beta], dim=-1),
            ],
            dim=-2,
        )

    def _emission_geometry(
        self,
        observers: torch.Tensor,
        source_positions: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_vec = observers[None, None, None, :, :] - source_positions
        r_mag = torch.linalg.vector_norm(r_vec, dim=-1)
        unit_r = r_vec / r_mag[..., None]

        global_to_blade = self._global_to_blade_rotation(beta.squeeze(-1))
        blade_to_airfoil = self.kinematics.blade_to_airfoil_rotation_matrix[
            self._selected_sections
        ].to(dtype=self.dtype, device=self.device)
        global_to_airfoil = torch.einsum(
            "sij,tsbojk->tsboik",
            blade_to_airfoil,
            global_to_blade,
        )
        unit_r_airfoil = torch.einsum(
            "tsboij,tsboj->tsboi",
            global_to_airfoil,
            unit_r,
        )

        r_x = unit_r_airfoil[..., 0]
        r_y = unit_r_airfoil[..., 1]
        r_z = unit_r_airfoil[..., 2]
        phi = torch.atan2(r_z, r_y)
        theta = torch.atan2(torch.sqrt(r_y**2 + r_z**2), r_x)

        phi_abs = torch.abs(phi)
        phi_sign = torch.where(
            phi >= 0.0,
            torch.ones_like(phi),
            -torch.ones_like(phi),
        )
        phi_abs_deg = torch.rad2deg(phi_abs)
        phi_small = phi_sign * torch.deg2rad(0.1 * phi_abs_deg**2 + 2.5)
        phi_large = phi_sign * torch.deg2rad(
            -0.1 * (phi_abs_deg - 180.0) ** 2 + 177.5
        )
        five_deg = torch.deg2rad(
            torch.tensor(5.0, dtype=self.dtype, device=self.device)
        )
        one_seventy_five_deg = torch.deg2rad(
            torch.tensor(175.0, dtype=self.dtype, device=self.device)
        )
        phi = torch.where(
            phi_abs < five_deg,
            phi_small,
            torch.where(phi_abs > one_seventy_five_deg, phi_large, phi),
        )

        mach = self.mach_number[None, :, None, None]
        dh_te = (2.0 * torch.sin(theta / 2.0) ** 2 * torch.sin(phi) ** 2) / (
            (1.0 + mach * torch.cos(theta))
            * (1.0 + 0.2 * mach * torch.cos(theta)) ** 2
        )
        dl = (torch.sin(theta) ** 2 * torch.sin(phi) ** 2) / (
            1.0 + mach * torch.cos(theta)
        ) ** 4
        return r_mag, dh_te, dh_te, dl

    def _set_base_values(
        self,
        observers: torch.Tensor,
        source_positions: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        r_mag, dh_te, dh_le, dl = self._emission_geometry(
            observers,
            source_positions,
            beta,
        )
        m5_dr = (self.mach_number**5 * self.section_width)[None, :, None, None]
        r_mag_sq = r_mag**2
        self.base_val_te = (m5_dr * dh_te) / r_mag_sq
        self.base_val_le = (m5_dr * dh_le) / r_mag_sq
        self.base_val_low = (m5_dr * dl) / r_mag_sq

    def compute_noise_components(
        self,
        *,
        turbulence_length_scale: float,
        turbulence_intensity: float,
    ) -> dict[str, torch.Tensor]:
        if (
            self.base_val_te is None
            or self.base_val_le is None
            or self.base_val_low is None
        ):
            raise RuntimeError("BPM base directivity values have not been computed.")

        # The standard positive-angle-of-attack propeller contract makes the
        # lower surface the pressure side and the upper surface the suction side.
        pressure_displacement_thickness = self.lower_displacement_thickness
        suction_displacement_thickness = self.upper_displacement_thickness

        return {
            "tbl": tbl.compute_tbl_noise(
                frequencies=self.frequencies,
                chord=self.section_chord,
                alpha=self.angle_of_attack_deg,
                u=self.relative_velocity,
                re_c=self.reynolds_number,
                m=self.mach_number,
                delta_p=pressure_displacement_thickness,
                delta_s=suction_displacement_thickness,
                base_val_te=self.base_val_te,
                base_val_low=self.base_val_low,
            ),
            "lbl": lbl.compute_lbl_noise(
                frequencies=self.frequencies,
                alpha=self.angle_of_attack_deg,
                u=self.relative_velocity,
                re_c=self.reynolds_number,
                delta_p=pressure_displacement_thickness,
                base_val_le=self.base_val_le,
            ),
            "teb": teb.compute_teb_noise(
                frequencies=self.frequencies,
                chord=self.section_chord,
                u=self.relative_velocity,
                m=self.mach_number,
                delta_p=pressure_displacement_thickness,
                delta_s=suction_displacement_thickness,
                psi=self.boat_tail_angle_deg,
                base_val_te=self.base_val_te,
            ),
            "ti": ti.compute_ti_noise(
                frequencies=self.frequencies,
                chord=self.section_chord,
                alpha=self.angle_of_attack_deg,
                u=self.relative_velocity,
                m=self.mach_number,
                rho=self.rho,
                a_inf=self.a_inf,
                base_val_le=self.base_val_le,
                base_val_low=self.base_val_low,
                turbulence_length_scale=turbulence_length_scale,
                turbulence_intensity=turbulence_intensity,
            ),
            "tv": tv.compute_tv_noise(
                frequencies=self.frequencies,
                r=self.section_radius,
                dr=self.section_width,
                chord=self.section_chord,
                alpha=self.angle_of_attack_deg,
                m=self.mach_number,
                a_inf=self.a_inf,
                base_val_te=self.base_val_te,
            ),
        }


__all__ = ["BPM"]
