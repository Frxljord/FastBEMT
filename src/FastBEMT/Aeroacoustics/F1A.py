from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from .Utils import (
    a_weighting_db,
    spl_spectrum_to_overall_level,
    time_domain_to_spl_spectrum,
)
from ..Kinematics import Kinematics

if TYPE_CHECKING:
    from ..Aerodynamics.BEMT import BEMT
    from ..Propeller import Propeller
    from ..Utils.Environment import Environment


ArrayLike = Union[np.ndarray, torch.Tensor]


class F1A:
    """Farassat formulation 1A for compact rotating sources.

    Args:
        propeller: Propeller containing geometry and simulation settings.
        environment: Acoustic environment stored by ``propeller``.
        loadings: Either a BEMT analysis or direct blade-frame force per
            unit span on the fluid with shape ``(S, 3)`` or
            ``(T, B, S, 3)`` and ``(axial, radial, tangential)``
            components. ``S`` must match the full Propeller geometry.
        rpm: Operating RPM. Optional for a single-point BEMT analysis and
            required for direct loadings.
        v_inf: BEMT freestream velocity. Required with ``rpm`` when
            selecting from a multi-point BEMT analysis.

    Observer-dependent source tensors use dimension order
    ``(O, T, B, S, ...)``.
    """

    def __init__(
        self,
        propeller: Propeller,
        environment: Environment,
        loadings: BEMT | ArrayLike,
        *,
        rpm: float | None = None,
        v_inf: float | None = None,
    ) -> None:
        self.propeller = propeller
        self.environment = environment
        if environment is not propeller.environment:
            raise ValueError(
                "F1A environment must be the environment stored by the propeller."
            )

        self.device = torch.device(propeller.device)
        self.dtype = propeller.dtype
        self.rho = propeller.rho_tensor
        self.a_inf = propeller.a_inf_tensor

        from ..Aerodynamics.BEMT import BEMT

        if isinstance(loadings, BEMT):
            if loadings.propeller is not propeller:
                raise ValueError(
                    "The BEMT analysis belongs to a different propeller."
                )
            if loadings.environment is not environment:
                raise ValueError(
                    "F1A and BEMT must use the same environment."
                )
            self.bemt: BEMT | None = loadings
            self.rpm, self.v_inf = loadings.resolve_operating_point(rpm, v_inf)
            solution = loadings.solution_for(self.rpm, self.v_inf)
        else:
            self.bemt = None
            if rpm is None:
                raise ValueError("rpm is required for direct F1A loadings.")
            self.rpm = float(rpm)
            if not np.isfinite(self.rpm) or self.rpm <= 0.0:
                raise ValueError("rpm must be finite and greater than zero.")
            if v_inf is not None:
                raise ValueError(
                    "v_inf is only used when loadings is a BEMT analysis."
                )
            self.v_inf = None
            solution = None

        existing_kinematics = propeller.kinematics
        if (
            existing_kinematics is not None
            and existing_kinematics.rpm == self.rpm
        ):
            self.kinematics = existing_kinematics
        else:
            self.kinematics = Kinematics(propeller, rpm=self.rpm)
        propeller.kinematics = self.kinematics
        self.nt = self.kinematics.nt
        self.nb = self.kinematics.nb

        section_mask = propeller.f1a_geometry_mask.copy()
        section_count = propeller.n_sections

        if solution is not None:
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
        else:
            loads_full = torch.as_tensor(
                loadings,
                dtype=self.dtype,
                device=self.device,
            )
            steady_shape = (section_count, 3)
            unsteady_shape = (
                self.nt,
                self.nb,
                section_count,
                3,
            )
            if tuple(loads_full.shape) not in (steady_shape, unsteady_shape):
                raise ValueError(
                    "Direct F1A loadings must have shape "
                    f"{steady_shape} or {unsteady_shape}; "
                    f"got {tuple(loads_full.shape)}."
                )
            if loads_full.ndim == 2:
                finite_load_sections = torch.isfinite(loads_full).all(
                    dim=1
                ).cpu().numpy()
            else:
                finite_load_sections = torch.isfinite(loads_full).all(
                    dim=(0, 1, 3)
                ).cpu().numpy()
            section_mask &= finite_load_sections
            if not np.any(section_mask):
                raise ValueError("No valid blade sections available for F1A.")
            all_sections_selected = bool(np.all(section_mask))
            if all_sections_selected:
                selected_sections = propeller.f1a_geometry_mask_tensor
                loads = loads_full
            else:
                selected_sections = torch.as_tensor(
                    section_mask,
                    dtype=torch.bool,
                    device=self.device,
                )
                if loads_full.ndim == 2:
                    loads = loads_full[selected_sections, :]
                else:
                    loads = loads_full[:, :, selected_sections, :]

        self.section_mask = section_mask
        self._selected_sections = selected_sections
        self._uses_all_sections = all_sections_selected
        if all_sections_selected:
            self.r = propeller.section_radius
            self.dr = propeller.section_width
            self.area = propeller.section_area
            self.chord = propeller.section_chord
            self.twist_rad = propeller.section_twist_rad
            self.com_shift_forward = propeller.section_com_shift_forward
            self.com_shift_up = propeller.section_com_shift_up
            self.thickness_strength = propeller.f1a_thickness_strength
            self.dipole_strength = propeller.f1a_dipole_strength
        else:
            self.r = propeller.section_radius[selected_sections]
            self.dr = propeller.section_width[selected_sections]
            self.area = propeller.section_area[selected_sections]
            self.chord = propeller.section_chord[selected_sections]
            self.twist_rad = propeller.section_twist_rad[selected_sections]
            self.com_shift_forward = propeller.section_com_shift_forward[
                selected_sections
            ]
            self.com_shift_up = propeller.section_com_shift_up[
                selected_sections
            ]
            self.thickness_strength = propeller.f1a_thickness_strength[
                selected_sections
            ]
            self.dipole_strength = propeller.f1a_dipole_strength[
                selected_sections
            ]
        self.ns = int(self.r.shape[0])
        self.loads = loads
        self.loadings = self.loads

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

    def _initialize_loading(self) -> None:
        """Rotate loads and form their complete source-time derivative."""
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

        # Compatibility aliases for callers that inspect the transformed loads.
        self.force_fixed = self.f0dot
        self.force_der_fixed = self.f1dot

    @torch.inference_mode()
    def run(
        self,
        observers: ArrayLike,
        observer_time_range: float,
        num_observer_times: int | None = None,
        *,
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

        self.observers = torch.as_tensor(
            observers,
            dtype=self.dtype,
            device=self.device,
        )
        if self.observers.ndim == 1:
            self.observers = self.observers.unsqueeze(0)
        if self.observers.ndim != 2 or self.observers.shape[1] != 3:
            raise ValueError("observers must have shape (O, 3).")

        source_p_m, source_p_d = self._calculate_source_pressures(self.observers)
        self.observer_times = self._calculate_observer_times(self.observers)
        source_pressure = torch.stack((source_p_m, source_p_d), dim=-1)
        interpolation_times, interpolation_pressure = (
            self._extend_periodic_steady_sources(
                self.observer_times,
                source_pressure,
            )
        )

        latest_reception_start = torch.amax(
            self.observer_times[:, 0, :, :],
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

        requested_spacing = (
            requested_time_range / requested_observer_count
        )
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
        self.observer_rotations = min(
            requested_rotations,
            available_rotations,
        )
        if self.observer_rotations < 1:
            raise ValueError(
                "The source-time data do not cover one complete observer "
                "revolution after the latest initial reception time. "
                "Provide at least one additional source revolution."
            )

        self.observer_time_range = (
            self.observer_rotations * rotation_period
        )
        self.num_observer_times = (
            self.observer_rotations * samples_per_rotation
        )
        self.sample_spacing = sample_spacing
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
        extended_times = torch.cat(
            (
                observer_times,
                observer_times[:, periodic_slice] + source_duration,
            ),
            dim=1,
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
        """Calculate uncombined source pressure in Julia's notation."""
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

        # Source-time derivatives used directly by AcousticAnalogies.jl.
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
