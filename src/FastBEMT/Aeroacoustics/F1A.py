from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import torch

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
        self.rho = torch.tensor(
            environment.rho,
            dtype=self.dtype,
            device=self.device,
        )
        self.a_inf = torch.tensor(
            environment.a_inf,
            dtype=self.dtype,
            device=self.device,
        )

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

        geometry = propeller.geometry
        r_all = np.asarray(geometry["r"], dtype=np.float64)
        dr_all = np.asarray(geometry["dr"], dtype=np.float64)
        chord_all = np.asarray(geometry["chord"], dtype=np.float64)
        twist_all = np.asarray(geometry["twist"], dtype=np.float64)
        area_all = np.asarray(geometry["cross_section"], dtype=np.float64)
        com_forward_all = np.asarray(
            propeller.com_shift_forward,
            dtype=np.float64,
        )
        com_up_all = np.asarray(propeller.com_shift_up, dtype=np.float64)

        section_mask = (
            np.isfinite(r_all)
            & (np.abs(r_all) > 1.0e-12)
            & np.isfinite(dr_all)
            & (dr_all > 0.0)
            & np.isfinite(chord_all)
            & np.isfinite(twist_all)
            & np.isfinite(area_all)
            & np.isfinite(com_forward_all)
            & np.isfinite(com_up_all)
        )

        if solution is not None:
            d_t = np.asarray(solution["d_t"].values, dtype=np.float64)
            d_q = np.asarray(solution["d_q"].values, dtype=np.float64)
            section_mask &= np.isfinite(d_t) & np.isfinite(d_q)
            if not np.any(section_mask):
                raise ValueError("No valid blade sections available for F1A.")
            axial_load = d_t[section_mask] / dr_all[section_mask] / self.nb
            tangential_load = (
                d_q[section_mask]
                / dr_all[section_mask]
                / r_all[section_mask]
                / self.nb
            )
            loads = torch.as_tensor(
                np.stack(
                    (
                        axial_load,
                        np.zeros_like(axial_load),
                        -tangential_load,
                    ),
                    axis=-1,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            loads_full = torch.as_tensor(
                loadings,
                dtype=self.dtype,
                device=self.device,
            )
            steady_shape = (int(r_all.shape[0]), 3)
            unsteady_shape = (
                self.nt,
                self.nb,
                int(r_all.shape[0]),
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
            if np.all(section_mask):
                loads = loads_full
            else:
                section_mask_tensor = torch.as_tensor(
                    section_mask,
                    dtype=torch.bool,
                    device=self.device,
                )
                if loads_full.ndim == 2:
                    loads = loads_full[section_mask_tensor, :]
                else:
                    loads = loads_full[:, :, section_mask_tensor, :]

        self.section_mask = section_mask
        self.r = torch.as_tensor(
            r_all[section_mask],
            dtype=self.dtype,
            device=self.device,
        )
        self.dr = torch.as_tensor(
            dr_all[section_mask],
            dtype=self.dtype,
            device=self.device,
        )
        self.area = torch.as_tensor(
            area_all[section_mask],
            dtype=self.dtype,
            device=self.device,
        )
        self.chord = torch.as_tensor(
            chord_all[section_mask],
            dtype=self.dtype,
            device=self.device,
        )
        self.twist_rad = torch.deg2rad(
            torch.as_tensor(
                twist_all[section_mask],
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.com_shift_forward = torch.as_tensor(
            com_forward_all[section_mask],
            dtype=self.dtype,
            device=self.device,
        )
        self.com_shift_up = torch.as_tensor(
            com_up_all[section_mask],
            dtype=self.dtype,
            device=self.device,
        )
        self.ns = int(self.r.shape[0])
        self._selected_sections = torch.as_tensor(
            section_mask,
            dtype=torch.bool,
            device=self.device,
        )
        self.loads = self._validate_loads(loads)
        self.loadings = self.loads

        # Section-only factors broadcast over (O, T, B, S).
        self.thickness_strength = self.rho * self.area * self.dr / (4.0 * np.pi)
        self.dipole_strength = self.dr / (4.0 * np.pi * self.a_inf)

        self._initialize_loading()

        self.observers: torch.Tensor | None = None
        self.observer_times: torch.Tensor | None = None
        self.t: torch.Tensor | None = None
        self.p_m: torch.Tensor | None = None
        self.p_d: torch.Tensor | None = None
        self.p_tot: torch.Tensor | None = None
        self.source_p_m: torch.Tensor | None = None
        self.source_p_d: torch.Tensor | None = None

    def _validate_loads(self, loads: ArrayLike) -> torch.Tensor:
        """Validate steady ``(S, 3)`` or unsteady ``(T, B, S, 3)`` loads."""
        load_tensor = torch.as_tensor(
            loads,
            dtype=self.dtype,
            device=self.device,
        )
        steady_shape = (self.ns, 3)
        unsteady_shape = (self.nt, self.nb, self.ns, 3)
        if tuple(load_tensor.shape) not in (steady_shape, unsteady_shape):
            raise ValueError(
                "loads must have shape "
                f"{steady_shape} or {unsteady_shape}; got {tuple(load_tensor.shape)}."
            )
        return load_tensor

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
            observer_time_range: Duration of the uniform observer-time grid.
            num_observer_times: Number of output samples. Defaults to ``T``.
            retain_source_terms: Keep the uncombined ``(O, T, B, S)`` source
                pressure tensors for diagnostics.

        Results are stored in ``p_m``, ``p_d``, and ``p_tot`` with shape
        ``(O, T_observer)``. ``p_tot`` has its observer-wise mean removed.
        """
        observer_count = self.nt if num_observer_times is None else int(
            num_observer_times
        )
        if observer_count <= 0:
            raise ValueError("num_observer_times must be greater than zero.")
        observer_time_range = float(observer_time_range)
        if not np.isfinite(observer_time_range) or observer_time_range <= 0.0:
            raise ValueError("observer_time_range must be finite and positive.")

        self.observer_times = None
        self.t = None
        self.p_m = None
        self.p_d = None
        self.p_tot = None
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

        latest_reception_start = torch.amax(
            self.observer_times[:, 0, :, :],
            dim=(1, 2),
        )
        observer_time_offsets = (
            torch.arange(
                observer_count,
                dtype=self.dtype,
                device=self.device,
            )
            * (observer_time_range / observer_count)
        )
        self.t = latest_reception_start[:, None] + observer_time_offsets[None, :]

        source_pressure = torch.stack((source_p_m, source_p_d), dim=-1)
        observer_pressure = self._interpolate_sources(
            self.t,
            self.observer_times,
            source_pressure,
        )
        self.p_m = observer_pressure[..., 0]
        self.p_d = observer_pressure[..., 1]
        pressure_total = self.p_m + self.p_d
        self.p_tot = pressure_total - torch.mean(
            pressure_total,
            dim=1,
            keepdim=True,
        )

        if retain_source_terms:
            self.source_p_m = source_p_m
            self.source_p_d = source_p_d
        else:
            self.source_p_m = None
            self.source_p_d = None

    def _calculate_source_pressures(
        self,
        observers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate uncombined source pressure in Julia's notation."""
        x_obs = observers[:, None, None, None, :]

        # Kinematics are stored as (T, B, S, 3). Adding the observer axis
        # gives every source term the native order (O, T, B, S, ...).
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
