import numpy as np
import pandas as pd
import aerosandbox as asb
import torch
from typing import Any, Tuple, Dict, List, Optional, Union
import pyfar as pf

from .Section import SectionForces
from .TorchCompactSource import F1A
from .JobParameters import LowFidelityParameters
from .TorchBPM import BPM


class Propeller:
    """High-level propeller BEMT and F1A acoustic analysis controller.

    Handles blade element momentum theory (BEMT) aerodynamic analysis and
    Farassat 1A (F1A) acoustic source computation for propeller noise prediction.
    """

    def __init__(
        self,
        propeller_geometry: Dict[str, Union[np.ndarray, int, float, list]],
        params: LowFidelityParameters,
    ) -> None:
        """Initialize propeller analysis with geometry and parameter objects.

        Args:
            propeller_geometry: Dictionary containing propeller geometry data including
                airfoil coordinates, radial stations, chord, twist, and COM shift.
                Expected keys: 'r', 'dr', 'chord', 'twist', 'airfoil', 'COM_shift',
                'tip_radius', 'hub_radius', 'n_blades'.
            params: Aerodynamic and acoustic parameters including RPM, diameter,
                density, and reference pressure.
            dtype: PyTorch data type for tensor operations (e.g., torch.float32).
        """
        self.geometry = propeller_geometry
        self.params = params
        self.dtype = torch.float32
        self.device = params.device
        self.section_areas()
        self.calculate_boat_tail_angle()

        # Initialize third-octave band frequencies for BPM analysis
        octave_freqs = pf.dsp.filter.fractional_octave_frequencies(
            num_fractions=3, frequency_range=(20, 20000)
        )[0]
        self.f_low = octave_freqs / (2 ** (1 / 6))
        self.f_high = octave_freqs * (2 ** (1 / 6))
        self.third_octave_freqs = torch.as_tensor(
            octave_freqs, dtype=self.dtype, device=self.device
        )

        # COM shift: positive in forward direction (+x), positive in upward direction (+z)
        self.com_shift_up: List[float] = [s[1] for s in self.geometry["COM_shift"]]
        self.com_shift_forward: List[float] = [
            -s[0] for s in self.geometry["COM_shift"]
        ]

        # BEMT solution storage
        self.solution_data: pd.DataFrame = pd.DataFrame(
            columns=[
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
            ]
        )

        # Initialize airfoil objects using aerosandbox
        self._section_airfoils: List[asb.Airfoil] = [
            asb.Airfoil(coordinates=af) for af in self.geometry["airfoil"]
        ]

        # Initialize section force solvers for each radial station
        self._sections: List[SectionForces] = [
            SectionForces(
                airfoil=self._section_airfoils[idx],
                r=self.geometry["r"][idx],
                dr=self.geometry["dr"][idx],
                chord=self.geometry["chord"][idx],
                theta=np.radians(self.geometry["twist"][idx]),
                params=self.params,
                prop_radius=self.geometry["tip_radius"],
                hub_radius=self.geometry["hub_radius"],
                n_blades=self.geometry["n_blades"],
            )
            for idx in range(len(self.geometry["r"]))
        ]

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
        self.v_inf: Optional[np.ndarray] = None
        self.observer_positions: Optional[np.ndarray] = None
        self.spl_breakdown: Dict[str, np.ndarray] = {}

    def process_section(
        self,
        section_index: int,
        v_inf: float,
        prev_phi: Optional[float] = None,
    ) -> List[float]:
        """Run BEMT solver for a single radial section.

        Args:
            section_index: Index of the radial station to process.
            v_inf: Freestream velocity at this section (m/s).
            prev_phi: Flow angle from previous section for convergence (radians).
                Defaults to None for the first section.

        Returns:
            List containing 18 elements:
            [r, dr, chord, twist, phi, alpha, c_l, c_d, u, a_prime,
            d_t, d_q, f_factor, w, reynolds, mach, delta_star_upper, delta_star_lower]
            Returns list of NaN values if solver fails.
        """
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
                f_factor,
                w,
                reynolds,
                mach,
                delta_star_upper,
                delta_star_lower,
            ) = self._sections[section_index].solve(v_inf, prev_phi=prev_phi)

            return [
                self.geometry["r"][section_index],
                self.geometry["dr"][section_index],
                self.geometry["chord"][section_index],
                self.geometry["twist"][section_index],
                np.degrees(phi),
                alpha,
                c_l,
                c_d,
                u,
                a_prime,
                d_t,
                d_q,
                f_factor,
                w,
                reynolds,
                mach,
                delta_star_upper,
                delta_star_lower,
            ]
        except RuntimeError as e:
            print(f"Error in section at r={self.geometry['r'][section_index]}: {e}")
            return [np.nan] * 18

    def run_bemt(self, v_inf: Union[float, np.ndarray]) -> None:
        """Execute the BEMT radial sweep sequentially.

        Solves blade element momentum theory equations at each radial station
        in order, using the flow angle from the previous station for convergence.

        Args:
            v_inf: Freestream velocity (m/s). Can be a scalar applied to all sections
                or an array of shape (n_radial_stations,) with velocity at each station.
        """
        self.v_inf = v_inf
        prev_phi: Optional[float] = None
        rows: List[List[float]] = []

        # Ensure v_inf is an array
        if isinstance(v_inf, (int, float)):
            v_inf = np.full(len(self.geometry["r"]), v_inf)

        # Process each radial section
        for i, (r, dr, chord, twist, airfoil_asb) in enumerate(
            zip(
                self.geometry["r"],
                self.geometry["dr"],
                self.geometry["chord"],
                self.geometry["twist"],
                self._section_airfoils,
            )
        ):
            res = self.process_section(i, v_inf[i], prev_phi=prev_phi)
            if not np.isnan(res[4]):  # Index 4 is phi in degrees
                prev_phi = np.radians(res[4])
            rows.append(res)

        self.solution_data = pd.DataFrame(rows, columns=self.solution_data.columns)

    def compute_total_forces(self) -> Tuple[float, float, float, float]:
        """Compute total thrust, torque, and nondimensional force coefficients.

        Integrates local blade element forces across all radial stations and
        computes dimensionless performance coefficients.

        Returns:
            Tuple of (total_thrust, total_torque, thrust_coefficient, power_coefficient)
            where:
            - total_thrust: Total thrust force (N)
            - total_torque: Total torque (N·m)
            - thrust_coefficient: Dimensionless thrust coefficient
            - power_coefficient: Dimensionless power coefficient
        """
        total_thrust: float = self.solution_data["d_t"].sum()
        total_torque: float = self.solution_data["d_q"].sum()

        n_rev_s: float = self.params.rpm / 60.0
        d: float = 2 * self.geometry["tip_radius"]
        rho: float = self.params.rho

        c_t = total_thrust / (rho * n_rev_s**2 * d**4)
        c_p = (2 * np.pi * total_torque) / (rho * n_rev_s**2 * d**5)

        return total_thrust, total_torque, c_t, c_p

    def section_areas(self) -> None:
        """Calculate cross-sectional areas (m²) using the Shoelace formula.

        Computes the area of each airfoil section scaled by chord length squared.
        Updates self.geometry["cross_section"] with computed areas.
        """
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
        """Calculate boat tail angle for each airfoil section.

        Computes the trailing edge boat tail angle by fitting linear regressions
        to the upper and lower surface trailing edge regions (x > 0.95 chord).
        The boat tail angle is the angle between the fitted upper and lower slopes.

        Updates self.geometry["boat_tail_angle"] with angles in degrees for each section.
        """
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
        local_dt: Optional[np.ndarray] = None,
        local_dq: Optional[np.ndarray] = None,
        lt: int = 1,
        i: float = 0.01,
        alpha_stall: float = 15.0,
        keep_bpm_components: bool = True,
    ) -> None:
        """Initialize acoustic array and compute total pressure at observer locations.

        Performs Farassat 1A (F1A) and BPM acoustic predictions for sources on
        the propeller blade surface. Utilizes GPU acceleration via PyTorch.

        Args:
            observer_positions: Observer location coordinates in Cartesian space.
                Shape: (num_observers, 3) in meters.
            local_dt: Local thrust distribution (override computed values).
                If None, uses solution_data['d_t']. Shape: (n_times, n_blades, n_sections).
            local_dq: Local torque distribution (override computed values).
                If None, uses solution_data['d_q']. Shape: (n_times, n_blades, n_sections).
            lt: Trailing edge noise model switch (BPM parameter).
            i: Turbulence intensity (BPM parameter).
            alpha_stall: Stall angle in degrees for BPM modeling.
            keep_bpm_components: If True, preserve individual BPM component SPL.
                If False, sum components before interpolation (faster).
        """
        self.observer_positions = observer_positions

        # Prepare local force distributions per unit span and per blade
        if local_dt is None or local_dq is None:
            local_dt = (
                self.solution_data["d_t"].values
                / self.geometry["dr"]
                / self.geometry["n_blades"]
            )
            local_dq = (
                self.solution_data["d_q"].values
                / self.geometry["dr"]
                / self.geometry["r"]
                / self.geometry["n_blades"]
            )
            # Broadcast to source time and blade dimensions
            local_dt = np.broadcast_to(
                local_dt[None, None, :],
                (
                    self.params.num_src_times,
                    self.geometry["n_blades"],
                    len(self.geometry["r"]),
                ),
            )
            local_dq = np.broadcast_to(
                local_dq[None, None, :],
                (
                    self.params.num_src_times,
                    self.geometry["n_blades"],
                    len(self.geometry["r"]),
                ),
            )
        else:
            local_dt = local_dt / self.geometry["dr"]
            local_dq = local_dq / self.geometry["dr"] / self.geometry["r"]

        with torch.inference_mode():
            # Initialize F1A compact source acoustic array
            acoustic_array = self._time_cuda(
                F1A,
                rho=self.params.rho,
                a_inf=self.params.a_inf,
                r=self.geometry["r"],
                dr=self.geometry["dr"],
                area=self.geometry["cross_section"],
                chord=self.geometry["chord"],
                twist=self.geometry["twist"],
                com_shift_forward=self.com_shift_forward,
                com_shift_up=self.com_shift_up,
                source_times=self.params.src_times,
                omega=self.params.omega,
                d_t=local_dt,
                d_q=local_dq,
                blade_angles=self.params.blade_angles,
                device=self.device,
                label="TorchCompactAcousticSourceArray.__init__",
            )

            # Initialize BPM noise source model
            bpm = self._time_cuda(
                BPM,
                frequencies=self.third_octave_freqs,
                r=self.geometry["r"],
                dr=self.geometry["dr"],
                chord=self.geometry["chord"],
                alpha=self.solution_data["alpha"].values,
                vi=self.solution_data["u"].values,
                u=self.solution_data["W"].values,
                re_c=self.solution_data["Re"].values,
                m=self.solution_data["Ma"].values,
                delta_p=self.solution_data["dp"].values,
                delta_s=self.solution_data["ds"].values,
                boat_tail_angle=self.geometry["boat_tail_angle"],
                src_times=self.params.src_times_one_rotation,
                a_inf=self.params.a_inf,
                rho=self.params.rho,
                omega=self.params.omega,
                blade_angles=self.params.blade_angles,
                twist=self.geometry["twist"],
                com_shift_forward=self.com_shift_forward,
                com_shift_up=self.com_shift_up,
                observer_time_range=self.params.observer_time_range
                / self.params.revolutions,
                num_obs_times=self.params.num_obs_times // self.params.revolutions,
                device=self.device,
                label="BPM.__init__",
            )

            # Compute retarded times for acoustic propagation
            obs_times = self._time_cuda(
                self.get_observer_times,
                pos_fixed=acoustic_array.pos_fixed,
                src_times=self.params.src_times,
                observers=observer_positions,
                label="get_observer_times",
            )

            # Calculate F1A pressure contributions
            f1a_output = self._time_cuda(
                acoustic_array.calculate_f1a_pressure,
                observer_positions,
                label="calculate_f1a_pressure",
            )

            # Interpolate F1A pressures to uniform observer time grid
            latest_reception_start_time = torch.amax(obs_times[0, :, :, :], dim=(0, 1))
            f1a_output_times = (
                latest_reception_start_time[None, :]
                + torch.linspace(
                    0,
                    self.params.observer_time_range,
                    self.params.num_obs_times,
                    device=self.device,
                )[:, None]
            )
            self._time_cuda(
                self.combine_sources,
                obs_times=obs_times,
                output_times=f1a_output_times,
                f1a_output=f1a_output,
                label="combine_sources (F1A)",
            )

            # Perform spectral analysis on combined F1A pressures
            self._time_cuda(
                self._perform_spectral_analysis_torch,
                label="_perform_spectral_analysis_torch",
            )

            bpm_obs_times = obs_times[: self.params.num_obs_times_per_rev]
            bpm_output_times = f1a_output_times[: self.params.num_obs_times_per_rev]

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

    def combine_sources_bpm(
        self,
        output_times: Union[torch.Tensor, np.ndarray],
        bpm_output: Dict[str, torch.Tensor],
        keep_components: bool = True,
    ) -> None:
        self.spl_breakdown: Dict[str, np.ndarray] = {}

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
        self.spl_total = 10 * torch.log10(spp_interp_total.mean(dim=1) + 1e-12)

    def combine_sources(
        self,
        obs_times: Union[torch.Tensor, np.ndarray],
        output_times: Union[torch.Tensor, np.ndarray],
        f1a_output: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Interpolate F1A source signals using GPU-accelerated PyTorch.

        Combines monopole and dipole contributions from all blade elements
        and interpolates to a uniform observer time grid.

        Args:
            obs_times: Retarded times at observer locations.
                Shape: (n_source_times, n_sources, n_observers).
            output_times: Uniform observer time grid for interpolation.
                Shape: (n_steps, n_observers).
            f1a_output: Pressure contributions from F1A sources.
                Shape: (n_times, n_sources, n_blades, n_observers, 2).
        """

        # Interpolate monopole and dipole contributions to uniform time grid
        self.p_m = self._interp_tensor_vectorized(
            output_times, obs_times, f1a_output[..., 0]
        )
        self.p_d = self._interp_tensor_vectorized(
            output_times, obs_times, f1a_output[..., 1]
        )

        # Compute total pressure and remove mean (DC offset)
        self.t = output_times
        self.p_tot = self.p_m + self.p_d
        self.p_tot -= torch.mean(self.p_tot, dim=0)

    def _interp_tensor_vectorized(
        self,
        x_new: torch.Tensor,
        x_old: torch.Tensor,
        y_old: torch.Tensor,
    ) -> torch.Tensor:
        """Perform vectorized linear interpolation on 4D GPU tensors.

        Args:
            x_new: (n_steps, n_observers)
            x_old: (n_src_times, n_sections, n_blades, n_observers)
            y_old: (n_src_times, n_sections, n_blades, n_observers)

        Returns:
            Shape: (n_steps, n_observers)
        """
        nt, n_sec, n_b, n_obs = x_old.shape
        n_steps = x_new.shape[0]

        # 1. Permute to put time (interpolation dim) at the end
        # From (nt, n_sec, n_b, no) -> (no, n_sec, n_b, nt)
        x_old_p = x_old.permute(3, 1, 2, 0).contiguous()
        y_old_p = y_old.permute(3, 1, 2, 0).contiguous()

        # 2. Prepare x_new: (n_steps, no) -> (no, n_sec, n_b, n_steps)
        # Transpose x_new to (no, n_steps), then expand to match dimensions
        x_new_p = (
            x_new.T.view(n_obs, 1, 1, n_steps).expand(-1, n_sec, n_b, -1).contiguous()
        )

        # 3. Find bracketing indices
        # searchsorted works on the last dimension (nt)
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, nt - 1)

        # 4. Gather bracketing points
        # Dim 3 is the time dimension we are interpolating within
        x0 = torch.gather(x_old_p, 3, idx - 1)
        x1 = torch.gather(x_old_p, 3, idx)
        y0 = torch.gather(y_old_p, 3, idx - 1)
        y1 = torch.gather(y_old_p, 3, idx)

        # 5. Linear interpolation formula
        # (y - y0) / (x - x0) = (y1 - y0) / (x1 - x0)
        weights = (x_new_p - x0) / (x1 - x0 + 1e-12)
        interp_vals = y0 + weights * (y1 - y0)

        # 6. Reduction
        # Sum across sections (dim 1) and blades (dim 2)
        # Result shape: (no, n_steps)
        summed = torch.sum(interp_vals, dim=(1, 2))

        # Transpose back to (n_steps, n_observers)
        return summed.T

    def _perform_spectral_analysis_torch(self) -> None:
        """Compute spectral analysis using GPU-accelerated PyTorch.

        Computes FFT, SPL, A-weighted SPL, OASPL, and OASPL-A metrics
        using GPU-accelerated PyTorch operations.
        """
        n: int = self.params.num_obs_times
        no: int = self.observer_positions.shape[0]

        # Frequency grid for spectral analysis
        dt: float = self.params.observer_time_range / self.params.num_obs_times
        f_single = torch.fft.rfftfreq(n, dt).to(self.device)
        self.freq = f_single.unsqueeze(1).expand(-1, no)

        # Compute FFT and convert to RMS amplitude
        fft_p = torch.fft.rfft(self.p_tot, dim=0)
        self.fft_amp = torch.abs(fft_p)
        self.fft_amp.mul_(np.sqrt(2) / n)
        self.fft_amp[0, :].div_(np.sqrt(2))  # DC component

        # Compute SPL: 20*log10(p_rms / p_ref)
        p_ref: float = self.params.p_ref
        self.spl = torch.clamp(self.fft_amp, min=1e-15)
        self.spl.div_(p_ref).log10_().mul_(20.0)

        def r_a_func_torch(f: torch.Tensor) -> torch.Tensor:
            """Compute A-weighting function per ISO 61672-1."""
            f_sq = f.square()
            c1 = 12194.0**2
            c2 = 20.6**2
            c3 = 107.7**2
            c4 = 737.9**2

            num = f_sq.square().mul_(c1**2)
            den = (
                f_sq.add(c2)
                .mul_(torch.sqrt(f_sq.add(c3).mul_(f_sq.add(c4))))
                .mul_(f_sq.add(c1))
            )
            return num.div_(den)

        # Normalize A-weighting to 1000 Hz
        r_a_1000 = r_a_func_torch(torch.tensor(1000.0, device=self.device))

        # Compute A-weighted SPL
        a_weight = r_a_func_torch(self.freq).div_(r_a_1000).log10_().mul_(20.0)

        self.spl_a = self.spl.clone()
        self.spl_a.add_(a_weight)

        # Compute Overall Sound Pressure Level (OSPL)
        p_rms_sq = self.fft_amp[0, :].square()
        p_rms_sq.add_(torch.sum(self.fft_amp[1:, :].square(), dim=0).mul_(2.0))
        self.ospl = torch.sqrt(p_rms_sq).div_(p_ref).log10_().mul_(20.0)
        self.ospl = self.ospl.cpu().numpy()

        # Compute Overall A-weighted Sound Pressure Level (OASPL)
        amp_a = torch.pow(10.0, self.spl_a.div_(20.0)).mul_(p_ref)

        p_rms_a_sq = amp_a[0, :].square()
        p_rms_a_sq.add_(torch.sum(amp_a[1:, :].square(), dim=0).mul_(2.0))
        self.oaspl = torch.sqrt(p_rms_a_sq).div_(p_ref).log10_().mul_(20.0)
        self.oaspl = self.oaspl.cpu().numpy()

    def get_observer_times(
        self,
        pos_fixed: torch.Tensor,
        observers: np.ndarray,
        src_times: torch.Tensor,
    ) -> torch.Tensor:
        """Compute retarded times for source-observer pairs.

        Calculates the time at which acoustic waves reach observers
        accounting for propagation delay: t_retarded = t_source + r / a_inf.

        Args:
            pos_fixed: Source positions on blade surface.
                Shape: (n_src_times, n_sections, n_blades, 3).
            observers: Observer positions in Cartesian coordinates.
                Shape: (n_observers, 3) in meters.
            src_times: Source time instants.
                Shape: (n_src_times,) in seconds.

        Returns:
            Retarded times tensor of shape (n_src_times, n_sources, n_observers)
            in seconds.
        """
        # Convert observer positions to tensor with batch dimensions
        obs = torch.as_tensor(observers, dtype=self.dtype, device=self.device)[
            None, None, None, :, :
        ]

        # Compute distance from each source to each observer
        dist = torch.linalg.norm(obs.sub(pos_fixed[..., None, :]), dim=-1)

        # Compute retarded time: t_retarded = t_source + distance / speed_of_sound
        src_times_tensor = torch.as_tensor(
            src_times, dtype=self.dtype, device=self.device
        )[:, None, None, None]
        speed_of_sound = torch.as_tensor(
            self.params.a_inf, dtype=self.dtype, device=self.device
        ).reciprocal()
        return src_times_tensor.add(dist.mul(speed_of_sound))

    def _time_cuda(
        self,
        func,
        *args,
        label: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Time a callable using CUDA events when available, otherwise wall-clock.

        Args:
            func: Callable to time.
            *args: Positional arguments to pass to func.
            label: Optional label for timing output.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Result from func(*args, **kwargs).
        """
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
        """Calculate A-weighting offset in dB for frequency values."""

        def ra_calc(freq: torch.Tensor) -> torch.Tensor:
            return (12194.0**2 * freq**4) / (
                (freq**2 + 20.6**2)
                * ((freq**2 + 107.7**2) * (freq**2 + 737.9**2)).sqrt()
                * (freq**2 + 12194.0**2)
            )

        f_tensor = torch.as_tensor(f, dtype=self.dtype, device=self.device)
        ra = ra_calc(f_tensor)
        ra_1000 = ra_calc(torch.tensor(1000.0, dtype=self.dtype, device=self.device))

        return 20.0 * torch.log10(ra) - 20.0 * torch.log10(ra_1000)

    def calc_oaspl(
        self,
        spl_tensor: torch.Tensor,
        freqs: torch.Tensor,
        weighted: bool = False,
    ) -> torch.Tensor:
        """Calculate OASPL for an arbitrary grid shape.

        Args:
            spl_tensor: SPL tensor with shape (F, ...) where F is the frequency bins.
            freqs: Frequency grid with shape (F,).
            weighted: Apply A-weighting if True.
        """
        if weighted:
            a_offsets = self.get_a_weighting(freqs).to(spl_tensor.device)

            # Align A-weighting to the frequency dimension.
            dims_to_add = spl_tensor.ndim - 1
            view_shape = (-1,) + (1,) * dims_to_add
            spl_tensor = spl_tensor + a_offsets.view(view_shape)

        power_ratio = 10 ** (spl_tensor / 10.0)
        summed_power = torch.sum(power_ratio, dim=0)
        return 10.0 * torch.log10(summed_power)

    def postprocess(self) -> None:
        """Aggregate third-octave bands and compute third-octave OASPL."""
        f_low = self.third_octave_freqs / (2 ** (1 / 6))
        f_high = self.third_octave_freqs * (2 ** (1 / 6))
        mask = (self.freq[:, 0:1].T >= f_low.unsqueeze(1)) & (
            self.freq[:, 0:1].T < f_high.unsqueeze(1)
        )

        p_raw = 10 ** (self.spl / 10.0)
        p_band = torch.matmul(mask.float(), p_raw)
        third_octave_f1a_spl = 10.0 * torch.log10(p_band.clamp(min=1e-12))
        third_octave_f1a_spl[p_band == 0] = float("-inf")
        self.third_octave_f1a_spl = third_octave_f1a_spl

        self.third_octave_total_spl = 10.0 * torch.log10(
            10 ** (third_octave_f1a_spl / 10.0)
            + 10 ** (self.spl_total / 10.0)
        )

        self.third_octave_f1a_oaspl = self.calc_oaspl(
            self.third_octave_f1a_spl, self.third_octave_freqs, weighted=True
        ).cpu().numpy()
        self.third_octave_total_oaspl = self.calc_oaspl(
            self.third_octave_total_spl, self.third_octave_freqs, weighted=True
        ).cpu().numpy()