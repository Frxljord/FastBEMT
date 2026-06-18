from __future__ import annotations

from os import PathLike
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import pyfar as pf
import torch

from .Aeroacoustics.F1A import F1A
from .Aeroacoustics.BPM import BPM
from .Aeroacoustics.Utils import a_weighting_db, spl_spectrum_to_overall_level
from .Kinematics import Kinematics
from .Utils.Environment import Environment
from .Utils.Simulation import Simulation

if TYPE_CHECKING:
    from .Aerodynamics.BEMT import BEMT


class Propeller:
    '''Propeller geometry, environment, and aeroacoustic analysis.

    Aerodynamic operating-point calculations are handled by
    :class:`FastBEMT.Aerodynamics.BEMT`. This class retains the geometry and
    environment needed by aerodynamic, acoustic, and structural analyses.
    '''

    def __init__(
        self,
        geometry: Dict[str, Union[np.ndarray, int, float, list]],
        environment: Environment,
        simulation: Simulation,
        use_cuda_timing: bool = False,
    ) -> None:
        '''Initialize propeller analysis.

        Args:
            geometry: Propeller geometry dictionary with keys:
                'r': radial stations (m), shape (n_sections,)
                'dr': radial widths (m), shape (n_sections,)
                'chord': chord lengths (m), shape (n_sections,)
                'twist': twist angles (deg), shape (n_sections,)
                'airfoil': airfoil coordinates, list of shape (n_points, 2)
                'COM_shift': center of mass shift, list of (x, z) tuples
                'tip_radius': propeller tip radius (m)
                'hub_radius': propeller hub radius (m)
                'n_blades': number of blades
            environment: Fluid and acoustic reference properties.
            simulation: Numerical device and time-discretization settings.
            use_cuda_timing: Print GPU/CPU timing diagnostics if True.
        '''
        self.geometry = geometry
        self.environment = environment
        self.simulation = simulation
        self.dtype = torch.float32
        self.device = simulation.device
        self.use_cuda_timing = use_cuda_timing
        self.section_areas()
        self.calculate_boat_tail_angle()

        # COM shift: positive in forward direction (+x), positive in upward direction (+z)
        self.com_shift_up: List[float] = [s[1] for s in self.geometry["COM_shift"]]
        self.com_shift_forward: List[float] = [
            -s[0] for s in self.geometry["COM_shift"]
        ]
        self._initialize_geometry_cache()

        # Initialize third-octave band frequencies for BPM analysis
        octave_freqs = pf.dsp.filter.fractional_octave_frequencies(
            num_fractions=3, frequency_range=(20, 20000)
        )[0]
        self.f_low = octave_freqs / (2 ** (1 / 6))
        self.f_high = octave_freqs * (2 ** (1 / 6))
        self.third_octave_freqs = torch.as_tensor(
            octave_freqs, dtype=self.dtype, device=self.device
        )

        # Acoustic results containers
        self.p_m: Optional[torch.Tensor] = None
        self.p_d: Optional[torch.Tensor] = None
        self.p_tot: Optional[torch.Tensor] = None
        self.freq: Optional[torch.Tensor] = None
        self.spl: Optional[torch.Tensor] = None
        self.spl_a: Optional[torch.Tensor] = None
        self.ospl: Optional[np.ndarray] = None
        self.oaspl: Optional[np.ndarray] = None
        self.fft_amp: Optional[torch.Tensor] = None
        self.observer_positions: Optional[np.ndarray] = None
        self.spl_breakdown: Dict[str, np.ndarray] = {}
        self.kinematics: Optional[Kinematics] = None
        self.f1a: Optional[F1A] = None

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
            "com_shift_forward": np.array(
                self.com_shift_forward,
                dtype=np.float64,
                copy=True,
            ),
            "com_shift_up": np.array(
                self.com_shift_up,
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
            & np.isfinite(section_arrays["com_shift_forward"])
            & np.isfinite(section_arrays["com_shift_up"])
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
        self.section_com_shift_forward = torch.tensor(
            section_arrays["com_shift_forward"],
            dtype=self.dtype,
            device=self.device,
        )
        self.section_com_shift_up = torch.tensor(
            section_arrays["com_shift_up"],
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
        '''Calculate cross-sectional areas using Shoelace formula.

        Computes area of each airfoil section scaled by chord squared.
        Updates geometry['cross_section'] with areas in m².
        '''
        areas: List[float] = []
        for idx, coords in enumerate(self.geometry["airfoil"]):
            x = coords[:, 0]
            y = coords[:, 1]
            # Shoelace formula for polygon area
            a_norm = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
            # Scale by chord squared
            areas.append(a_norm * self.geometry["chord"][idx] ** 2)
        self.geometry["cross_section"] = np.array(areas)

    def calculate_boat_tail_angle(self) -> None:
        '''Calculate trailing edge boat tail angle.

        Fits linear regressions to upper and lower surfaces in trailing edge
        region (x > 0.95) and computes angle between slopes.
        Updates geometry['boat_tail_angle'] with angles in degrees.
        '''
        angles: List[float] = []
        for idx, coords in enumerate(self.geometry["airfoil"]):
            # Find leading edge index
            le_idx = np.argmin(coords[:, 0])
            upper = coords[: le_idx + 1]
            lower = coords[le_idx:]

            # Extract trailing edge points (x > 0.95)
            u_pts = upper[upper[:, 0] > 0.95]
            l_pts = lower[lower[:, 0] > 0.95]

            # Need at least 2 points on each surface
            if len(u_pts) < 2 or len(l_pts) < 2:
                angles.append(0.0)
                continue

            # Fit line to trailing edge and compute slope
            m_u, _ = np.polyfit(u_pts[:, 0], u_pts[:, 1], 1)
            m_l, _ = np.polyfit(l_pts[:, 0], l_pts[:, 1], 1)

            # Boat tail angle is angle between upper and lower slopes
            angle = np.degrees(np.abs(np.arctan(m_u) - np.arctan(m_l)))
            angles.append(angle)

        self.geometry["boat_tail_angle"] = np.array(angles)

    def run_aeroacoustics(
        self,
        observer_positions: np.ndarray,
        bemt: BEMT,
        loads: Optional[Union[str, PathLike[str]]] = None,
        load_source_times: Optional[Union[str, PathLike[str]]] = None,
        lt: int = 1,
        i: float = 0.01,
        alpha_stall: float = 15.0,
        keep_bpm_components: bool = True,
        rpm: Optional[float] = None,
        v_inf: Optional[float] = None,
    ) -> None:
        '''Compute F1A and BPM acoustic predictions.

        Performs Farassat 1A thickness and loading noise propagation followed by
        Brooks-Pope-Marcolini broadband noise prediction using GPU acceleration.

        Args:
            observer_positions: Observer coordinates, shape (n_observers, 3) in meters.
            bemt: Aerodynamic analysis providing the section loads.
            loads: Optional path to a ``.pt`` file containing blade-frame
                loading per unit span on the fluid in
                ``(axial, radial, tangential)`` components with shape
                ``(n_sections, 3)`` or
                ``(n_times, n_blades, n_sections, 3)``. BEMT loading is used
                when None.
            load_source_times: Path to a CSV timestamp table for ``loads``.
                Required when ``loads`` is provided.
            lt: Turbulent length scale
            i: Turbulence intensity for BPM (dimensionless).
            alpha_stall: Stall angle for BPM separation noise (degrees).
            keep_bpm_components: Store individual BPM component SPL if True.
            rpm: RPM of the BEMT case to use. Required for multi-point BEMT.
            v_inf: Freestream velocity of the BEMT case to use. Required for
                multi-point BEMT.
        '''
        if bemt.propeller is not self:
            raise ValueError("The BEMT analysis belongs to a different propeller.")
        if loads is None and load_source_times is not None:
            raise ValueError("load_source_times is only used when loads is provided.")
        if loads is not None and load_source_times is None:
            raise ValueError("load_source_times is required when loads is provided.")

        operating_rpm, operating_v_inf = bemt.resolve_operating_point(
            rpm,
            v_inf,
        )
        solution_data = bemt.solution_for(operating_rpm, operating_v_inf)

        with torch.inference_mode():
            self.f1a = self._time_cuda(
                F1A,
                propeller=self,
                environment=self.environment,
                loadings=bemt if loads is None else loads,
                rpm=operating_rpm,
                v_inf=operating_v_inf if loads is None else None,
                source_times=None if loads is None else load_source_times,
                label="F1A.__init__",
            )
        self.kinematics = self.f1a.kinematics

        self.observer_positions = np.atleast_2d(observer_positions)
        self.third_octave_total_oaspl = None

        # Immutable geometry was validated and converted when Propeller was built.
        geometry = self.section_geometry_np
        geom_r = geometry["r"]
        geom_dr = geometry["dr"]
        geom_chord = geometry["chord"]
        geom_twist = geometry["twist"]
        geom_boat_tail = geometry["boat_tail_angle"]

        sol_alpha = np.asarray(solution_data['alpha'].values)
        sol_u = np.asarray(solution_data['u'].values)
        sol_w = np.asarray(solution_data['W'].values)
        sol_re = np.asarray(solution_data['Re'].values)
        sol_ma = np.asarray(solution_data['Ma'].values)
        sol_dp = np.asarray(solution_data['dp'].values)
        sol_ds = np.asarray(solution_data['ds'].values)

        section_mask = (
            self.bpm_geometry_mask
            & np.isfinite(sol_alpha)
            & np.isfinite(sol_u)
            & np.isfinite(sol_w)
            & np.isfinite(sol_re)
            & np.isfinite(sol_ma)
            & np.isfinite(sol_dp)
            & np.isfinite(sol_ds)
            & self.f1a.section_mask
        )
        if not np.any(section_mask):
            raise ValueError('No valid blade sections available for aeroacoustics.')
        if not np.all(section_mask):
            n_invalid = int((~section_mask).sum())
            print(
                f'[WARN] run_aeroacoustics dropped {n_invalid} invalid section(s) '
                'with non-finite aerodynamic/geometry data.'
            )

        r = geom_r[section_mask]
        dr = geom_dr[section_mask]
        chord = geom_chord[section_mask]
        twist = geom_twist[section_mask]
        boat_tail_angle = geom_boat_tail[section_mask]
        com_shift_forward = geometry["com_shift_forward"][section_mask]
        com_shift_up = geometry["com_shift_up"][section_mask]

        alpha = sol_alpha[section_mask]
        vi = sol_u[section_mask]
        u = sol_w[section_mask]
        re_c = sol_re[section_mask]
        m = sol_ma[section_mask]
        delta_p = sol_dp[section_mask]
        delta_s = sol_ds[section_mask]

        with torch.inference_mode():
            # Initialize BPM noise source model
            bpm = self._time_cuda(
                BPM,
                frequencies=self.third_octave_freqs,
                r=r,
                dr=dr,
                chord=chord,
                alpha=alpha,
                vi=vi,
                u=u,
                re_c=re_c,
                m=m,
                delta_p=delta_p,
                delta_s=delta_s,
                boat_tail_angle=boat_tail_angle,
                src_times=self.simulation.src_times_one_rotation,
                a_inf=self.environment.a_inf,
                rho=self.environment.rho,
                omega=self.simulation.omega,
                blade_angles=self.simulation.blade_angles,
                twist=twist,
                com_shift_forward=com_shift_forward,
                com_shift_up=com_shift_up,
                observer_time_range=self.simulation.observer_time_range
                / self.simulation.revolutions,
                num_obs_times=self.simulation.num_obs_times
                // self.simulation.revolutions,
                device=self.device,
                kinematics=self.f1a.kinematics,
                section_indices=np.flatnonzero(section_mask),
                label="BPM.__init__",
            )

            self._time_cuda(
                self.f1a.run,
                self.observer_positions,
                observer_time_range=self.simulation.observer_time_range,
                num_observer_times=self.simulation.num_obs_times,
                label="F1A.run",
            )
            if (
                self.f1a.p_m is None
                or self.f1a.p_d is None
                or self.f1a.p_tot is None
                or self.f1a.observer_times is None
                or self.f1a.t is None
                or self.f1a.frequencies is None
                or self.f1a.spl is None
                or self.f1a.spl_a is None
                or self.f1a.ospl is None
                or self.f1a.oaspl is None
            ):
                raise RuntimeError("F1A did not populate its acoustic results.")

            # Retain the established Propeller result API as aliases.
            self.p_m = self.f1a.p_m
            self.p_d = self.f1a.p_d
            self.p_tot = self.f1a.p_tot
            self.t = self.f1a.t

            # Retain the established Propeller spectral result API as aliases.
            observer_count = int(self.f1a.p_tot.shape[0])
            self.freq = self.f1a.frequencies[None, :].expand(
                observer_count,
                -1,
            )
            self.spl = self.f1a.spl
            self.spl_a = self.f1a.spl_a
            self.ospl = self.f1a.ospl.cpu().numpy()
            self.oaspl = self.f1a.oaspl.cpu().numpy()
            self.fft_amp = self.environment.p_ref * torch.pow(
                10.0,
                self.f1a.spl / 20.0,
            )

            bpm_sections_in_f1a = torch.as_tensor(
                section_mask[self.f1a.section_mask],
                dtype=torch.bool,
                device=self.device,
            )
            bpm_obs_times = self.f1a.observer_times[
                :,
                : self.simulation.num_obs_times_per_rev,
                :,
                bpm_sections_in_f1a,
            ].permute(1, 3, 2, 0).contiguous()
            bpm_output_times = self.f1a.t[
                :,
                : self.simulation.num_obs_times_per_rev,
            ].T.contiguous()

            # Compute BPM noise contributions
            bpm_output = self._time_cuda(
                bpm.run_forward_bpm,
                self.observer_positions,
                bpm_obs_times,
                bpm_output_times,
                lt,
                i,
                alpha_stall,
                label="BPM.run_forward_bpm",
            )

            self._time_cuda(
                self.combine_sources_bpm,
                output_times=bpm_output_times,
                bpm_output=bpm_output,
                keep_components=keep_bpm_components,
                label="combine_sources_bpm",
            )
            
            torch.cuda.empty_cache()

    def combine_sources_bpm(
        self,
        output_times: Union[torch.Tensor, np.ndarray],
        bpm_output: Dict[str, torch.Tensor],
        keep_components: bool = True,
    ) -> None:
        self.spl_breakdown = {}

        if not keep_components:
            # Sum all BPM component tensors
            spp_interp_total: Optional[torch.Tensor] = None
            for v in bpm_output.values():
                if spp_interp_total is None:
                    spp_interp_total = v.clone()
                else:
                    spp_interp_total = spp_interp_total + v
        else:
            # Interpolate each BPM component separately
            spp_interp_total = torch.zeros(
                (
                    self.third_octave_freqs.shape[0],
                    output_times.shape[0],
                    self.observer_positions.shape[0],
                ),
                device=self.device,
            )

            for name, spp_raw in bpm_output.items():
                # Accumulate for total
                spp_interp_total += spp_raw
                # Calculate and store individual component SPL
                self.spl_breakdown[name] = (
                    10 * torch.log10(spp_raw.mean(dim=1) + 1e-12).cpu().numpy()
                )

        # Final Total SPL (acoustic pressure squared level)
        self.third_octave_bpm_spl = 10 * torch.log10(spp_interp_total.mean(dim=1) + 1e-12)

    def _time_cuda(
        self,
        func,
        *args,
        label: Optional[str] = None,
        **kwargs,
    ) -> Any:
        '''Execute function with GPU/CPU timing diagnostics.

        Args:
            func: Callable to execute and time.
            *args: Positional arguments for func.
            label: Optional label for timing output.
            **kwargs: Keyword arguments for func.

        Returns:
            Result from func(*args, **kwargs).
        '''
        if not self.use_cuda_timing:
            return func(*args, **kwargs)

        name = label or getattr(func, "__name__", "call")
        use_cuda = torch.cuda.is_available() and ("cuda" in str(self.device))

        if use_cuda:
            # Use CUDA events for precise GPU timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            result = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            print(f"[CUDA TIMING] {name}: {ms:.3f} ms")
            return result

        # Fall back to wall-clock timing on CPU
        import time

        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[CPU TIMING] {name}: {(t1 - t0) * 1000.0:.3f} ms")
        return result

    def get_a_weighting(self, f: Union[torch.Tensor, float]) -> torch.Tensor:
        '''Calculate A-weighting offset per ISO 61672-1.
        
        Args:
            f: Frequency or frequencies (Hz).
            
        Returns:
            A-weighting offset in dB.
        '''

        f_tensor = torch.as_tensor(f, dtype=self.dtype, device=self.device)
        return a_weighting_db(f_tensor)

    def calc_oaspl(
        self,
        spl_tensor: torch.Tensor,
        freqs: torch.Tensor,
        weighted: bool = False,
    ) -> torch.Tensor:
        '''Calculate overall sound pressure level.

        Args:
            spl_tensor: SPL values, shape (n_freqs, ...).
            freqs: Frequency values, shape (n_freqs,).
            weighted: Apply A-weighting if True.
            
        Returns:
            OASPL or OASPL-A, shape (...).
        '''
        return spl_spectrum_to_overall_level(
            spl_tensor,
            freqs,
            weighted=weighted,
            frequency_dim=0,
        )

    def postprocess(self) -> None:
        '''Aggregate to third-octave bands and compute OASPL.
        
        Converts narrowband F1A and BPM spectra to third-octave bands and
        computes overall A-weighted sound pressure levels.
        '''
        f_low = self.third_octave_freqs / (2 ** (1 / 6))
        f_high = self.third_octave_freqs * (2 ** (1 / 6))
        mask = (self.freq[0:1, :] >= f_low.unsqueeze(1)) & (
            self.freq[0:1, :] < f_high.unsqueeze(1)
        )

        p_raw = 10 ** (self.spl / 10.0)
        p_band = torch.einsum("kf,of->ko", mask.float(), p_raw)
        third_octave_f1a_spl = 10.0 * torch.log10(p_band.clamp(min=1e-12))
        third_octave_f1a_spl[p_band == 0] = float("-inf")
        self.third_octave_f1a_spl = third_octave_f1a_spl

        self.third_octave_total_spl = 10.0 * torch.log10(
            10 ** (self.third_octave_f1a_spl / 10.0)
            + 10 ** (self.third_octave_bpm_spl / 10.0)
        )

        self.third_octave_f1a_oaspl = self.calc_oaspl(
            self.third_octave_f1a_spl, self.third_octave_freqs, weighted=True
        ).cpu().numpy()
        self.third_octave_bpm_oaspl = self.calc_oaspl(
            self.third_octave_bpm_spl, self.third_octave_freqs, weighted=True
        ).cpu().numpy()
        self.third_octave_total_oaspl = self.calc_oaspl(
            self.third_octave_total_spl, self.third_octave_freqs, weighted=True
        ).cpu().numpy()
