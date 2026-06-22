from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import torch

from ._common import ArrayLike, normalize_observer_batch_size, observer_tensor
from .Utils import (
    a_weighting_db,
    power_ratio_to_spl,
    third_octave_spectrum_to_overall_level,
)
from ..Kinematics import Kinematics

if TYPE_CHECKING:
    from ..Aerodynamics.BEMT import BEMT
    from ..Propeller import Propeller


def _load_bpm_component(name: str) -> ModuleType:
    """Load a BPM component module from the sibling BPM directory."""
    component_path = Path(__file__).with_suffix("") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{__name__}_{name}", component_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load BPM component module: {component_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tbl_component = _load_bpm_component("tbl")
_lbl_component = _load_bpm_component("lbl")
_teb_component = _load_bpm_component("teb")
_ti_component = _load_bpm_component("ti")
_tv_component = _load_bpm_component("tv")


class BPM:
    """Brooks-Pope-Marcolini broadband noise model.

    The object consumes a Propeller, BEMT solution, and optional shared
    Kinematics object. Calling ``run`` stores third-octave component spectra,
    total SPL, and total OSPL/OASPL on the instance.
    """

    def __init__(
        self,
        propeller: Propeller,
        bemt: BEMT,
        *,
        kinematics: Kinematics | None = None,
        rpm: float | None = None,
        v_inf: float | None = None,
        lt: float = 1.0,
        i: float = 0.01,
        alpha_stall: float = 15.0,
        trailing_edge_offset: ArrayLike | None = None,
        c1: ArrayLike | None = None,
    ) -> None:
        from ..Aerodynamics.BEMT import BEMT

        if not isinstance(bemt, BEMT):
            raise TypeError("bemt must be a FastBEMT Aerodynamics.BEMT object.")
        if bemt.propeller is not propeller:
            raise ValueError("The BEMT analysis belongs to a different propeller.")
        if bemt.environment is not propeller.environment:
            raise ValueError("BPM and BEMT must use the same environment.")
        if trailing_edge_offset is not None and c1 is not None:
            raise ValueError("Specify either trailing_edge_offset or c1, not both.")

        self.propeller = propeller
        self.environment = propeller.environment
        self.bemt = bemt
        self.device = torch.device(propeller.device)
        self.dtype = propeller.dtype
        self.lt = float(lt)
        self.i = float(i)
        self.alpha_stall = float(alpha_stall)
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
        if not np.any(self.section_mask):
            raise ValueError("No valid blade sections available for BPM.")
        self._selected_sections = torch.as_tensor(
            self.section_mask,
            dtype=torch.bool,
            device=self.device,
        )
        self.section_indices = np.flatnonzero(self.section_mask)
        self.ns = int(self.section_indices.shape[0])
        self.nb = self.kinematics.nb
        self.source_rotation_count = self._source_rotation_count()
        self.nt = self.source_rotation_count
        self.source_times = self.kinematics.source_times[
            : self.source_rotation_count
        ].to(dtype=self.dtype, device=self.device)

        geometry = propeller.section_geometry_np
        self.frequencies = propeller.third_octave_freqs.to(
            dtype=self.dtype,
            device=self.device,
        )
        self.r = self._section_tensor(geometry["r"])
        self.dr = self._section_tensor(geometry["dr"])
        self.chord = self._section_tensor(geometry["chord"])
        self.alpha = self._solution_tensor(solution, "alpha")
        self.vi = self._solution_tensor(solution, "u")
        self.u = self._solution_tensor(solution, "W")
        self.re_c = self._solution_tensor(solution, "Re")
        self.m = self._solution_tensor(solution, "Ma")
        self.delta_p = self._solution_tensor(solution, "dp")
        self.delta_s = self._solution_tensor(solution, "ds")
        self.psi = self._section_tensor(geometry["boat_tail_angle"])
        self.a_inf = torch.tensor(
            self.environment.a_inf,
            dtype=self.dtype,
            device=self.device,
        )
        self.rho = torch.tensor(
            self.environment.rho,
            dtype=self.dtype,
            device=self.device,
        )
        self.trailing_edge_offset = self._trailing_edge_offset(
            trailing_edge_offset,
            c1,
            geometry["chord"],
        )

        self.observers: torch.Tensor | None = None
        self.observer_times: torch.Tensor | None = None
        self.t: torch.Tensor | None = None
        self.observer_rotations: int | None = None
        self.observer_time_range: float | None = None
        self.num_observer_times: int | None = None
        self.sample_spacing: float | None = None
        self.source_rotation_repetitions: int | None = None
        self.base_val_te: torch.Tensor | None = None
        self.base_val_le: torch.Tensor | None = None
        self.base_val_low: torch.Tensor | None = None
        self.component_p2: dict[str, torch.Tensor] = {}
        self.component_spl: dict[str, torch.Tensor] = {}
        self.spl: torch.Tensor | None = None
        self.spl_a: torch.Tensor | None = None
        self.ospl: torch.Tensor | None = None
        self.oaspl: torch.Tensor | None = None
        self.source_component_p2: dict[str, torch.Tensor] | None = None

    def _kinematics_matches(self, kinematics: Kinematics | None) -> bool:
        if kinematics is None:
            return False
        if kinematics.propeller is not self.propeller:
            return False
        return np.isclose(float(kinematics.rpm), float(self.rpm))

    def _valid_section_mask(self, solution: object) -> np.ndarray:
        finite_solution = np.ones(self.propeller.n_sections, dtype=bool)
        for column in ("alpha", "u", "W", "Re", "Ma", "dp", "ds"):
            finite_solution &= np.isfinite(
                np.asarray(solution[column].to_numpy(), dtype=np.float64)
            )
        return self.propeller.bpm_geometry_mask & finite_solution

    def _source_rotation_count(self) -> int:
        simulation_times = self.propeller.simulation.src_times_one_rotation
        if simulation_times is not None:
            count = int(torch.as_tensor(simulation_times).shape[0])
            if 0 < count <= self.kinematics.nt:
                return count

        if self.kinematics.nt == 1:
            return 1

        source_times = self.kinematics.source_times
        median_dt = torch.median(torch.diff(source_times))
        first_rotation_end = source_times[0] + self.rotation_period
        mask = source_times < (first_rotation_end - 0.5 * median_dt)
        count = int(mask.sum().item())
        if count > 0:
            return count

        estimated_count = int(round(self.rotation_period / float(median_dt.item())))
        return max(1, min(estimated_count, self.kinematics.nt))

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
        c1: ArrayLike | None,
        full_chord: np.ndarray,
    ) -> torch.Tensor:
        selected_chord = full_chord[self.section_mask].astype(np.float64, copy=False)
        if trailing_edge_offset is not None:
            values = self._selected_optional_values(
                trailing_edge_offset,
                name="trailing_edge_offset",
                full_chord=full_chord,
            )
        elif c1 is not None:
            c1_values = self._selected_optional_values(
                c1,
                name="c1",
                full_chord=full_chord,
            )
            values = selected_chord - c1_values
        else:
            values = 0.5 * selected_chord
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    @torch.inference_mode()
    def run(
        self,
        observers: ArrayLike,
        observer_time_range: float | None = None,
        num_observer_times: int | None = None,
        *,
        lt: float | None = None,
        i: float | None = None,
        alpha_stall: float | None = None,
        observer_batch_size: int | None = None,
        retain_source_terms: bool = False,
    ) -> None:
        """Compute BPM component and total third-octave spectra.

        Args:
            observers: Stationary observer coordinates, shape ``(O, 3)``.
            observer_time_range: Optional requested observer-time duration.
            num_observer_times: Optional requested observer sample count.
            lt: Override the turbulent length scale for this run.
            i: Override the turbulence intensity for this run.
            alpha_stall: Override the stall angle in degrees for this run.
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
                observer_time_range=observer_time_range,
                num_observer_times=num_observer_times,
                lt=lt,
                i=i,
                alpha_stall=alpha_stall,
                retain_source_terms=retain_source_terms,
            )
            return

        batch_results = []
        for start in range(0, int(observer_values.shape[0]), batch_size):
            observer_batch = observer_values[start : start + batch_size]
            self._run_observer_batch(
                observer_batch,
                observer_time_range=observer_time_range,
                num_observer_times=num_observer_times,
                lt=lt,
                i=i,
                alpha_stall=alpha_stall,
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
        observer_time_range: float | None,
        num_observer_times: int | None,
        lt: float | None,
        i: float | None,
        alpha_stall: float | None,
        retain_source_terms: bool,
    ) -> None:
        self._reset_results()
        self.observers = observers

        source_positions, source_beta = self._source_trajectory()
        source_observer_times = self._source_observer_times(
            self.observers,
            source_positions,
        )
        self.observer_times = source_observer_times.permute(
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
            source_observer_times,
            source_positions,
            source_beta,
        )
        output_times = self._observer_output_times(
            interpolation_times,
            observer_time_range,
            num_observer_times,
        )
        self.t = output_times.T.contiguous()

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
            lt=self.lt if lt is None else float(lt),
            i=self.i if i is None else float(i),
            alpha_stall=(
                self.alpha_stall if alpha_stall is None else float(alpha_stall)
            ),
        )
        if retain_source_terms:
            self.source_component_p2 = source_components

        self.component_p2 = {
            name: values.sum(dim=(2, 3))
            for name, values in source_components.items()
        }
        band_power = {
            name: values.mean(dim=1).T.contiguous()
            for name, values in self.component_p2.items()
        }
        self.component_spl = {
            name: power_ratio_to_spl(values)
            for name, values in band_power.items()
        }

        total_band_power: torch.Tensor | None = None
        for values in band_power.values():
            total_band_power = (
                values
                if total_band_power is None
                else total_band_power + values
            )
        if total_band_power is None:
            raise RuntimeError("BPM did not compute any noise components.")

        self.spl = power_ratio_to_spl(total_band_power)
        self.spl_a = self.spl + a_weighting_db(self.frequencies)[None, :]
        self.ospl = third_octave_spectrum_to_overall_level(
            self.spl,
            self.frequencies,
            frequency_dim=1,
        )
        self.oaspl = third_octave_spectrum_to_overall_level(
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
            "observer_times": self.observer_times,
            "t": self.t,
            "observer_rotations": self.observer_rotations,
            "observer_time_range": self.observer_time_range,
            "num_observer_times": self.num_observer_times,
            "sample_spacing": self.sample_spacing,
            "source_rotation_repetitions": self.source_rotation_repetitions,
            "base_val_te": self.base_val_te,
            "base_val_le": self.base_val_le,
            "base_val_low": self.base_val_low,
            "component_p2": self.component_p2,
            "component_spl": self.component_spl,
            "spl": self.spl,
            "spl_a": self.spl_a,
            "ospl": self.ospl,
            "oaspl": self.oaspl,
            "frequencies": self.frequencies,
        }
        if retain_source_terms:
            result["source_component_p2"] = self.source_component_p2
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
        self.source_rotation_repetitions = max(
            int(result["source_rotation_repetitions"])
            for result in batch_results
        )
        self.frequencies = first["frequencies"]

        for result in batch_results[1:]:
            if int(result["observer_rotations"]) != self.observer_rotations:
                raise RuntimeError("BPM observer batches used different durations.")
            if int(result["num_observer_times"]) != self.num_observer_times:
                raise RuntimeError(
                    "BPM observer batches used different sample counts."
                )
            if not np.isclose(
                float(result["sample_spacing"]),
                self.sample_spacing,
                rtol=1.0e-7,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "BPM observer batches used different sample spacing."
                )
            if not torch.allclose(result["frequencies"], self.frequencies):
                raise RuntimeError(
                    "BPM observer batches produced different frequency grids."
                )

        self.observer_times = torch.cat(
            [result["observer_times"] for result in batch_results],
            dim=0,
        )
        self.t = torch.cat([result["t"] for result in batch_results], dim=0)
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

        component_names = tuple(first["component_p2"])
        self.component_p2 = {
            name: torch.cat(
                [result["component_p2"][name] for result in batch_results],
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
            self.source_component_p2 = {
                name: torch.cat(
                    [
                        result["source_component_p2"][name]
                        for result in batch_results
                    ],
                    dim=4,
                )
                for name in component_names
            }
        else:
            self.source_component_p2 = None

    def _reset_results(self) -> None:
        self.observers = None
        self.observer_times = None
        self.t = None
        self.observer_rotations = None
        self.observer_time_range = None
        self.num_observer_times = None
        self.sample_spacing = None
        self.source_rotation_repetitions = None
        self.base_val_te = None
        self.base_val_le = None
        self.base_val_low = None
        self.component_p2 = {}
        self.component_spl = {}
        self.spl = None
        self.spl_a = None
        self.ospl = None
        self.oaspl = None
        self.source_component_p2 = None

    def _source_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        reference_position = self.kinematics.section_position_global_frame[
            : self.source_rotation_count,
            :,
            self._selected_sections,
            :,
        ].to(dtype=self.dtype, device=self.device)
        blade_to_global = self.kinematics.blade_to_global_rotation_matrix[
            : self.source_rotation_count
        ].to(dtype=self.dtype, device=self.device)
        airfoil_to_blade = self.kinematics.airfoil_to_blade_rotation_matrix[
            self._selected_sections
        ].to(dtype=self.dtype, device=self.device)

        offset_airfoil = torch.zeros(
            (self.ns, 3),
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
        source_beta = self.kinematics.blade_angles[
            : self.source_rotation_count,
            None,
            :,
            None,
        ].to(dtype=self.dtype, device=self.device)
        source_beta = source_beta.expand(-1, self.ns, -1, -1).contiguous()
        return source_position, source_beta

    def _source_observer_times(
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
        observer_times: torch.Tensor,
        source_positions: torch.Tensor,
        source_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latest_reception_start = torch.amax(observer_times[0], dim=(0, 1))
        earliest_reception_end = torch.amin(observer_times[-1], dim=(0, 1))
        available_time_range = torch.amin(
            earliest_reception_end - latest_reception_start
        ).item()
        extra_rotations = max(
            1,
            int(
                np.ceil(
                    max(0.0, self.rotation_period - available_time_range)
                    / self.rotation_period
                )
            ),
        )

        time_blocks = []
        position_blocks = []
        beta_blocks = []
        two_pi = torch.tensor(
            2.0 * np.pi,
            dtype=self.dtype,
            device=self.device,
        )
        for repeat_index in range(extra_rotations + 1):
            time_shift = repeat_index * self.rotation_period
            time_blocks.append(observer_times + time_shift)
            position_blocks.append(source_positions)
            beta_blocks.append(source_beta + repeat_index * two_pi)
        self.source_rotation_repetitions = extra_rotations + 1
        return (
            torch.cat(time_blocks, dim=0),
            torch.cat(position_blocks, dim=0),
            torch.cat(beta_blocks, dim=0),
        )

    def _observer_output_times(
        self,
        observer_times: torch.Tensor,
        observer_time_range: float | None,
        num_observer_times: int | None,
    ) -> torch.Tensor:
        requested_time_range = (
            self.propeller.simulation.observer_time_range
            if observer_time_range is None
            else float(observer_time_range)
        )
        if requested_time_range is None:
            requested_time_range = self.rotation_period
        requested_time_range = float(requested_time_range)
        if not np.isfinite(requested_time_range) or requested_time_range <= 0.0:
            raise ValueError("observer_time_range must be finite and positive.")

        requested_rotations = int(
            np.floor(requested_time_range / self.rotation_period + 1.0e-7)
        )
        if requested_rotations < 1:
            raise ValueError(
                "observer_time_range must contain at least one complete "
                f"rotor revolution ({self.rotation_period:.9g} s)."
            )

        if num_observer_times is None:
            samples_per_rotation = int(self.propeller.simulation.num_obs_times_per_rev)
        else:
            requested_observer_count = int(num_observer_times)
            if requested_observer_count <= 0:
                raise ValueError("num_observer_times must be greater than zero.")
            requested_spacing = requested_time_range / requested_observer_count
            samples_per_rotation = max(
                1,
                int(np.rint(self.rotation_period / requested_spacing)),
            )
        if samples_per_rotation <= 0:
            raise ValueError("num_observer_times must imply at least one sample.")
        sample_spacing = self.rotation_period / samples_per_rotation

        latest_reception_start = torch.amax(observer_times[0], dim=(0, 1))
        earliest_reception_end = torch.amin(observer_times[-1], dim=(0, 1))
        available_time_range = torch.amin(
            earliest_reception_end - latest_reception_start
        ).item()
        available_rotations = int(
            np.floor(
                (
                    available_time_range
                    + sample_spacing
                    + 1.0e-7 * self.rotation_period
                )
                / self.rotation_period
            )
        )
        if available_rotations < 1:
            raise ValueError(
                "The source-time data do not cover one complete observer "
                "revolution after the latest initial reception time."
            )

        self.observer_rotations = 1
        self.observer_time_range = self.rotation_period
        self.num_observer_times = samples_per_rotation
        self.sample_spacing = sample_spacing
        offsets = (
            torch.arange(
                self.num_observer_times,
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
        nt, n_sec, n_b, _ = x_old.shape
        n_steps, n_obs = x_new.shape

        x_old_p = x_old.permute(3, 1, 2, 0).contiguous()
        x_new_p = (
            x_new.T.view(n_obs, 1, 1, n_steps)
            .expand(-1, n_sec, n_b, -1)
            .contiguous()
        )
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, nt - 1)

        coord_count = y_old.shape[-1]
        y_old_p = y_old.permute(3, 1, 2, 0).contiguous()
        y_old_exp = y_old_p.unsqueeze(1).expand(-1, n_obs, -1, -1, -1)
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

        mach = self.m[None, :, None, None]
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
        m5_dr = (self.m**5 * self.dr)[None, :, None, None]
        r_mag_sq = r_mag**2
        self.base_val_te = (m5_dr * dh_te) / r_mag_sq
        self.base_val_le = (m5_dr * dh_le) / r_mag_sq
        self.base_val_low = (m5_dr * dl) / r_mag_sq

    def compute_noise_components(
        self,
        *,
        lt: float,
        i: float,
        alpha_stall: float,
    ) -> dict[str, torch.Tensor]:
        if (
            self.base_val_te is None
            or self.base_val_le is None
            or self.base_val_low is None
        ):
            raise RuntimeError("BPM base directivity values have not been computed.")

        return {
            "tbl": _tbl_component.compute_tbl_noise(
                frequencies=self.frequencies,
                chord=self.chord,
                alpha=self.alpha,
                u=self.u,
                re_c=self.re_c,
                m=self.m,
                delta_p=self.delta_p,
                delta_s=self.delta_s,
                base_val_te=self.base_val_te,
                base_val_low=self.base_val_low,
                alpha_stall=alpha_stall,
            ),
            "lbl": _lbl_component.compute_lbl_noise(
                frequencies=self.frequencies,
                alpha=self.alpha,
                u=self.u,
                re_c=self.re_c,
                delta_p=self.delta_p,
                base_val_le=self.base_val_le,
            ),
            "teb": _teb_component.compute_teb_noise(
                frequencies=self.frequencies,
                chord=self.chord,
                u=self.u,
                m=self.m,
                delta_p=self.delta_p,
                delta_s=self.delta_s,
                psi=self.psi,
                base_val_te=self.base_val_te,
            ),
            "ti": _ti_component.compute_ti_noise(
                frequencies=self.frequencies,
                chord=self.chord,
                alpha=self.alpha,
                u=self.u,
                m=self.m,
                rho=self.rho,
                a_inf=self.a_inf,
                base_val_le=self.base_val_le,
                base_val_low=self.base_val_low,
                lt=lt,
                i=i,
            ),
            "tv": _tv_component.compute_tv_noise(
                frequencies=self.frequencies,
                r=self.r,
                dr=self.dr,
                chord=self.chord,
                alpha=self.alpha,
                m=self.m,
                a_inf=self.a_inf,
                base_val_te=self.base_val_te,
            ),
        }


__all__ = ["BPM"]
