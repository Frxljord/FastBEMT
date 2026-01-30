import numpy as np
import pandas as pd
import aerosandbox as asb
import torch
from typing import Tuple
import pyfar as pf

from .Section import SectionForces
from .TorchCompactSource import TorchCompactAcousticSourceArray
from .JobParameters import LowFidelityParameters
from .TorchBPM import BPM

class Propeller:
    """High-level propeller BEMT and F1A acoustic analysis controller.
    
    Handles blade element momentum theory (BEMT) aerodynamic analysis and
    Farassat 1A (F1A) acoustic source computation for propeller noise prediction.
    """

    def __init__(
        self,
        propeller_geometry: dict,
        params: LowFidelityParameters,
        dtype : torch.FloatTensor,
    ) -> None:
        """Initialize propeller analysis with geometry and parameter objects.
        
        Args:
            propeller_geometry: Dictionary containing propeller geometry data including
                airfoil coordinates, radial stations, chord, twist, and COM shift.
            aero_params: Aerodynamic parameters including RPM, diameter, and density.
            acoustic_params: Acoustic parameters including reference pressure and
                source/observer time grids.
        """
        self.geometry = propeller_geometry
        self.params = params
        self.dtype = dtype
        self.device = params.device
        self.section_areas()
        self.calculate_boat_tail_angle()
        self.third_octave_freqs = torch.as_tensor(pf.dsp.filter.fractional_octave_frequencies(num_fractions=3, frequency_range=(20,20000))[0], dtype=self.dtype, device=self.device)

        # COM shift: positive in forward direction (+ x), positive in upward direction (+ z)
        self.com_shift_up = [s[1] for s in self.geometry['COM_shift']]
        self.com_shift_forward = [-s[0] for s in self.geometry['COM_shift']]

        # BEMT solution storage
        self.solution_data = pd.DataFrame(
            columns=[
                "r", "dr", "chord", "twist", "phi", "alpha", 
                "Cl", "Cd", "u", "a_prime", "d_t", "d_q", 
                "F", "W", "Re", "Ma", "ds", "dp"
            ]
        )

        # Initialize airfoil objects using aerosandbox
        self._section_airfoils = [
            asb.Airfoil(coordinates=af) for af in self.geometry["airfoil"]
        ]

        # Initialize section force solvers for each radial station
        self._sections = [
            SectionForces(
                airfoil=self._section_airfoils[idx],
                r=self.geometry['r'][idx],
                dr=self.geometry['dr'][idx],
                chord=self.geometry['chord'][idx],
                theta=np.radians(self.geometry['twist'][idx]),
                params=self.params,
                prop_radius=self.geometry['tip_radius'],
                hub_radius=self.geometry['hub_radius'],
                n_blades=self.geometry['n_blades'],
            )
            for idx in range(len(self.geometry["r"]))
        ]

        # Acoustic results containers
        self.p_m: np.ndarray | None = None
        self.p_d: np.ndarray | None = None
        self.p_tot: np.ndarray | None = None
        self.freq: np.ndarray | None = None
        self.spl: np.ndarray | None = None
        self.spl_a: np.ndarray | None = None
        self.ospl: np.ndarray | None = None
        self.oaspl: np.ndarray | None = None
        self.fft_amp: np.ndarray | None = None
        self.v_inf: np.ndarray | None = None
        self.observer_positions: np.ndarray | None = None

    def process_section(
        self,
        section_index: int,
        v_inf: float,
        prev_phi: float | None = None,
    ) -> list:
        """Run BEMT solver for a single radial section.
        
        Args:
            section_index: Index of the radial station to process.
            v_inf: Freestream velocity at this section (m/s).
            prev_phi: Flow angle from previous section for convergence (radians).
            
        Returns:
            List containing [r, dr, chord, twist, phi, alpha, c_l, c_d, u, a_prime,
            d_t, d_q, f_factor, w, reynolds, mach, d_s, d_p] for the section,
            or list of NaN values if solver fails.
        """
        try:
            (
                phi, d_t, d_q, alpha, u, a_prime, c_l, c_d, f_factor, w,
                reynolds, mach, delta_star_upper, delta_star_lower
            ) = self._sections[section_index].solve(v_inf, prev_phi=prev_phi)

            return [
                self.geometry['r'][section_index],
                self.geometry['dr'][section_index],
                self.geometry['chord'][section_index],
                self.geometry['twist'][section_index],
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

    def run_bemt(self, v_inf: float | np.ndarray) -> None:
        """Execute the BEMT radial sweep sequentially.
        
        Args:
            v_inf: Freestream velocity (m/s). Can be a scalar applied to all sections
                or an array with velocity at each radial station.
        """
        self.v_inf = v_inf
        prev_phi = None
        rows = []

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
        
        Returns:
            Tuple of (total_thrust, total_torque, thrust_coefficient, power_coefficient)
            where total_thrust and total_torque are in SI units (N, N·m) and
            coefficients are dimensionless.
        """
        total_thrust = self.solution_data["d_t"].sum()
        total_torque = self.solution_data["d_q"].sum()

        n_rev_s = self.params.rpm / 60.0
        d = 2 * self.geometry['tip_radius']
        rho = self.params.rho

        c_t = total_thrust / (rho * n_rev_s**2 * d**4)
        c_p = (2 * np.pi * total_torque) / (rho * n_rev_s**2 * d**5)

        return total_thrust, total_torque, c_t, c_p

    def section_areas(self) -> None:
        """Calculate cross-sectional areas (m^2) using the Shoelace formula.
        
        Updates self.geometry["cross_section"] with computed areas for each
        airfoil section, accounting for chord scaling.
        """
        areas = []
        for idx, coords in enumerate(self.geometry["airfoil"]):
            x, y = coords[:, 0], coords[:, 1]
            # Shoelace formula for polygon area
            a_norm = 0.5 * np.abs(
                np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
            )
            # Scale by chord squared
            areas.append(a_norm * self.geometry['chord'][idx] ** 2)
        self.geometry["cross_section"] = np.array(areas)

    def calculate_boat_tail_angle(self) -> None:
        """Calculate boat tail angle for each airfoil section.
        
        Computes the trailing edge boat tail angle by fitting lines to the upper
        and lower surface trailing edge regions (x > 0.95 chord).
        
        Updates self.geometry["boat_tail_angle"] with angles in degrees for each section.
        """
        angles = []
        for idx, coords in enumerate(self.geometry["airfoil"]):
            # Find leading edge index
            le_idx = np.argmin(coords[:, 0])
            upper = coords[:le_idx + 1]
            lower = coords[le_idx:]

            # Extract trailing edge points (x > 0.95)
            u_pts = upper[upper[:, 0] > 0.95]
            l_pts = lower[lower[:, 0] > 0.95]

            # Need at least 2 points on each surface
            if len(u_pts) < 2 or len(l_pts) < 2:
                return 0.0

            # Fit line to trailing edge and compute slope
            m_u, _ = np.polyfit(u_pts[:, 0], u_pts[:, 1], 1)
            m_l, _ = np.polyfit(l_pts[:, 0], l_pts[:, 1], 1)

            # Boat tail angle is angle between upper and lower slopes
            angles.append(np.degrees(np.abs(np.arctan(m_u) - np.arctan(m_l))))

        self.geometry["boat_tail_angle"] = np.array(angles)
        
    def run_aeroacoustics(
        self,
        observer_positions: np.ndarray,
        local_dT: np.ndarray | None = None,
        local_dQ: np.ndarray | None = None,
        keep_bpm_components: bool = True,
    ) -> None:
        """Initialize acoustic array and compute total pressure at observer locations.
        
        Performs Farassat 1A (F1A) acoustic prediction for compact sources on the
        propeller blade surface. Can utilize GPU acceleration via PyTorch.
        
        Args:
            observer_positions: Observer location coordinates in Cartesian space.
                Shape: (num_observers, 3).
            local_d_t: Local thrust distribution to use instead of computed values.
                If None, uses solution_data['d_t'].
            local_d_q: Local torque distribution to use instead of computed values.
                If None, uses solution_data['d_q'].
            use_gpu: Whether to use GPU acceleration via PyTorch CUDA.
        """
        self.observer_positions = observer_positions

        # Prepare local force distributions
        if local_dT is None or local_dQ is None:
            local_dT = (
                self.solution_data['d_t'].values
                / self.geometry['dr']
                / self.geometry['n_blades']
            )
            local_dQ = (
                self.solution_data['d_q'].values
                / self.geometry['dr']
                / self.geometry['r']
                / self.geometry['n_blades']
            )
            # Broadcast to source time and blade dimensions
            local_dT = np.broadcast_to(
                local_dT[None, None, :],
                (
                    self.params.num_src_times,
                    self.geometry['n_blades'],
                    len(self.geometry['r']),
                ),
            )
            local_dQ = np.broadcast_to(
                local_dQ[None, None, :],
                (
                    self.params.num_src_times,
                    self.geometry['n_blades'],
                    len(self.geometry['r']),
                ),
            )
        else:
            local_dT = local_dT / self.geometry['dr']
            local_dQ = (
                local_dQ / self.geometry['dr'] / self.geometry['r']
            )

        with torch.inference_mode():
            acoustic_array = TorchCompactAcousticSourceArray(
                rho=self.params.rho,
                a_inf=self.params.a_inf,
                r=self.geometry['r'],
                dr=self.geometry['dr'],
                area=self.geometry['cross_section'],
                chord=self.geometry['chord'],
                twist=self.geometry['twist'],
                com_shift_forward=self.com_shift_forward,
                com_shift_up=self.com_shift_up,
                source_times=self.params.src_times,
                omega=self.params.omega,
                d_t=local_dT,
                d_q=local_dQ,
                blade_angles=self.params.blade_angles,
                device=self.device,
            )

            bpm = BPM(
                frequencies=self.third_octave_freqs,
                r=self.geometry['r'],
                dr=self.geometry['dr'],
                chord=self.geometry['chord'],
                alpha=self.solution_data['alpha'].values,
                vi=self.solution_data['u'].values,
                u=self.solution_data['W'].values,
                re_c=self.solution_data['Re'].values,
                m=self.solution_data['Ma'].values,
                delta_p=self.solution_data['dp'].values,
                delta_s=self.solution_data['ds'].values,
                boat_tail_angle=self.geometry['boat_tail_angle'],
                src_times=self.params.src_times,
                a_inf=self.params.a_inf,
                omega=self.params.omega,
                blade_angles=self.params.blade_angles,
                twist=self.geometry['twist'],
                com_shift_forward=self.com_shift_forward,
                com_shift_up=self.com_shift_up,
                observer_time_range=self.params.observer_time_range,
                num_obs_times=self.params.num_obs_times,
                device=self.device,
            )

            # Time object construction and heavy ops
            acoustic_array = self._time_cuda(
                TorchCompactAcousticSourceArray,
                rho=self.params.rho,
                a_inf=self.params.a_inf,
                r=self.geometry['r'],
                dr=self.geometry['dr'],
                area=self.geometry['cross_section'],
                chord=self.geometry['chord'],
                twist=self.geometry['twist'],
                com_shift_forward=self.com_shift_forward,
                com_shift_up=self.com_shift_up,
                source_times=self.params.src_times,
                omega=self.params.omega,
                d_t=local_dT,
                d_q=local_dQ,
                blade_angles=self.params.blade_angles,
                device=self.device,
                label="TorchCompactAcousticSourceArray.__init__",
            )

            bpm = self._time_cuda(
                BPM,
                frequencies=self.third_octave_freqs,
                r=self.geometry['r'],
                dr=self.geometry['dr'],
                chord=self.geometry['chord'],
                alpha=self.solution_data['alpha'].values,
                vi=self.solution_data['u'].values,
                u=self.solution_data['W'].values,
                re_c=self.solution_data['Re'].values,
                m=self.solution_data['Ma'].values,
                delta_p=self.solution_data['dp'].values,
                delta_s=self.solution_data['ds'].values,
                boat_tail_angle=self.geometry['boat_tail_angle'],
                src_times=self.params.src_times_one_rotation,
                a_inf=self.params.a_inf,
                omega=self.params.omega,
                blade_angles=self.params.blade_angles,
                twist=self.geometry['twist'],
                com_shift_forward=self.com_shift_forward,
                com_shift_up=self.com_shift_up,
                observer_time_range=self.params.observer_time_range/self.params.revolutions,
                num_obs_times=self.params.num_obs_times/self.params.revolutions,
                device=self.device,
                label="BPM.__init__",
            )

            # Compute acoustic pressures
            f1a_obs_times = self._time_cuda(
                self.get_observer_times,
                pos_fixed=acoustic_array.pos_fixed,
                src_times=self.params.src_times,
                observers=observer_positions,
                label="get_observer_times",
            )

            bpm_obs_times = self._time_cuda(
                self.get_observer_times,
                pos_fixed=acoustic_array.pos_fixed[:self.params.num_obs_times_per_rev, ...],
                src_times=self.params.src_times_one_rotation,
                observers=observer_positions,
                label="get_observer_times",
            )

            f1a_output = self._time_cuda(
                acoustic_array.calculate_f1a_pressure,
                observer_positions,
                label="calculate_f1a_pressure",
            )

            # Combine sources and perform spectral analysis (timed)
            self._time_cuda(self.combine_sources, obs_times=f1a_obs_times, output_times=torch.max(f1a_obs_times[0, :, :], dim=0)[0][None, :]
                + torch.linspace(0, self.params.observer_time_range, self.params.num_obs_times, device=self.device)[:, None], f1a_output=f1a_output, label="combine_sources")

            self._time_cuda(self._perform_spectral_analysis_torch, label="_perform_spectral_analysis_torch")

            #####################################
            bpm_output = self._time_cuda(bpm.run_forward_bpm, self.observer_positions, label="BPM.run_forward_bpm")

            self._time_cuda(
                self.combine_sources_bpm,
                obs_times=bpm_obs_times,
                output_times=torch.max(bpm_obs_times[0, :, :], dim=0)[0][None, :]
                + torch.linspace(0, self.params.observer_time_range/self.params.revolutions, self.params.num_obs_times//self.params.revolutions, device=self.device)[:, None],
                bpm_output=bpm_output,
                keep_components=keep_bpm_components,
                label="combine_sources_bpm",
            )


    def combine_sources_bpm(
        self,
        obs_times: torch.Tensor | np.ndarray,
        output_times: torch.Tensor | np.ndarray,
        bpm_output: dict,
        keep_components: bool = True,
    ):
        nf, nt, ns, nb, no = bpm_output['tbl'].shape
        self.spl_breakdown = {}

        t_raw_expanded = obs_times.repeat(1, nf, 1)

        # If the user requests to sum components before interpolation,
        # perform the sum first and interpolate only the combined signal.
        if not keep_components:
            # Sum all components into a single spp tensor of shape (nf, nt, ns, nb, no)
            combined = None
            for v in bpm_output.values():
                if combined is None:
                    combined = v.clone()
                else:
                    combined = combined + v

            # reshape combined for interpolation helper: (nt, n_freq * ns * nb, no)
            combined_flat = combined.permute(1, 0, 2, 3, 4).reshape(nt, -1, no)
            
            # interpolate only the summed signal
            spp_interp_total = self._interp_bpm_vectorized(output_times, t_raw_expanded, combined_flat, nf)

        else:
            spp_interp_total = torch.zeros((nf, output_times.shape[0], no), device=self.device)

            for name, spp_raw in bpm_output.items():
                # Reshape for interpolation helper: (nt, n_freq * ns * nb, no)
                spp_raw_flat = spp_raw.permute(1, 0, 2, 3, 4).reshape(nt, -1, no)

                # Vectorized Interpolation
                spp_interp = self._interp_bpm_vectorized(
                    output_times, t_raw_expanded, spp_raw_flat, nf
                )

                # Accumulate for total
                spp_interp_total += spp_interp

                # Calculate and store individual SPL (n_freq, no)
                self.spl_breakdown[name] = 10 * torch.log10(spp_interp.mean(dim=1) + 1e-12)

        # Final Total SPL
        self.spl_total = 10 * torch.log10(spp_interp_total.mean(dim=1) + 1e-12)

    def combine_sources(
        self,
        obs_times: torch.Tensor | np.ndarray,
        output_times: torch.Tensor | np.ndarray,
        f1a_output: torch.Tensor | np.ndarray,
    ) -> None:
        """Interpolate F1A source signals using GPU-accelerated PyTorch operations.
        
        Args:
            obs_times: Retarded times at observer locations.
            f1a_output: Pressure contributions from F1A sources.
            t_range: Total time span for observer grid (seconds).
            n_steps: Number of time steps in uniform observer grid.
        """

        nt, ns, nb, no, _ = f1a_output.shape

        # Reshape for processing
        p_m_raw = f1a_output[..., 0].reshape(nt, -1, no)
        p_d_raw = f1a_output[..., 1].reshape(nt, -1, no)

        # Interpolate monopole and dipole
        self.p_m = self._interp_tensor_vectorized(output_times, obs_times, p_m_raw)
        self.p_d = self._interp_tensor_vectorized(output_times, obs_times, p_d_raw)

        # Total pressure and remove mean
        self.p_tot = self.p_m + self.p_d
        self.p_tot -= torch.mean(self.p_tot, dim=0)

    def _interp_tensor_vectorized(
        self,
        x_new: torch.Tensor,
        x_old: torch.Tensor,
        y_old: torch.Tensor,
    ) -> torch.Tensor:
        """Perform full vectorized linear interpolation on GPU tensors.
        
        Efficiently interpolates multi-dimensional data using PyTorch's
        searchsorted and gather operations.
        
        Args:
            x_new: New x-coordinates for interpolation. Shape: (n_steps, noervers).
            x_old: Original x-coordinates. Shape: (n_src_times, n_sources, noervers).
            y_old: Original y-values. Shape: (n_src_times, n_sources, noervers).
            
        Returns:
            Interpolated values at x_new coordinates.
            Shape: (n_steps, noervers).
        """
        nt, n_src, _ = x_old.shape

        # Permute and reshape for searchsorted (searchsorted operates on last dim)
        # (No, N_src, Nt)
        x_old_p = x_old.permute(2, 1, 0).contiguous()
        y_old_p = y_old.permute(2, 1, 0).contiguous()

        # Prepare x_new for searchsorted: (No, N_src, N_steps)
        x_new_p = (
            x_new.T.unsqueeze(1).expand(-1, n_src, -1).contiguous()
        )

        # Find indices for searchsorted
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, nt - 1)

        # Linear interpolation: gather and compute weighted average
        x0 = torch.gather(x_old_p, 2, idx - 1)
        x1 = torch.gather(x_old_p, 2, idx)
        y0 = torch.gather(y_old_p, 2, idx - 1)
        y1 = torch.gather(y_old_p, 2, idx)

        weights = (x_new_p - x0) / (x1 - x0 + 1e-12)
        interp_vals = y0 + weights * (y1 - y0)

        # Sum across sources (dim=1) and return as (n_steps, noervers)
        return torch.sum(interp_vals, dim=1).T
    
    def _perform_spectral_analysis_torch(self) -> None:
        """Compute spectral analysis using GPU-accelerated PyTorch operations.
        
        GPU-accelerated computing FFT, SPL,
        and acoustic metrics with PyTorch tensors.
        """
        n, no = self.params.num_obs_times, self.observer_positions.shape[0]

        # Frequency calculation
        # dt = (self.t[1, 0] - self.t[0, 0]).item()
        dt = self.params.observer_time_range/self.params.num_obs_times
        f_single = torch.fft.rfftfreq(n, dt).to(self.device)
        self.freq = f_single.unsqueeze(1).expand(-1, no)

        # FFT and Amplitude
        fft_p = torch.fft.rfft(self.p_tot, dim=0)
        self.fft_amp = torch.abs(fft_p)
        self.fft_amp.mul_(np.sqrt(2) / n)
        self.fft_amp[0, :].div_(np.sqrt(2))

        # SPL Calculation
        p_ref = self.params.p_ref
        self.spl = torch.clamp(self.fft_amp, min=1e-15)
        self.spl.div_(p_ref).log10_().mul_(20.0)
        
        def r_a_func_torch(f):
            f_sq = f.square()
            c1 = 12194.0**2
            c2 = 20.6**2
            c3 = 107.7**2
            c4 = 737.9**2
            
            num = f_sq.square().mul_(c1**2)
            den = f_sq.add(c2).mul_(torch.sqrt(f_sq.add(c3).mul_(f_sq.add(c4)))).mul_(f_sq.add(c1))
            return num.div_(den)

        r_a_1000 = r_a_func_torch(torch.tensor(1000.0, device=self.device))
        
        a_weight = r_a_func_torch(self.freq).div_(r_a_1000).log10_().mul_(20.0)
        
        self.spl_a = self.spl.clone()
        self.spl_a.add_(a_weight)

        # OSPL
        p_rms_sq = self.fft_amp[0, :].square()
        p_rms_sq.add_(torch.sum(self.fft_amp[1:, :].square(), dim=0).mul_(2.0))
        self.ospl = torch.sqrt(p_rms_sq).div_(p_ref).log10_().mul_(20.0)
        self.ospl = self.ospl.cpu().numpy()

        amp_a = torch.pow(10.0, self.spl_a.div_(20.0)).mul_(p_ref) 
        
        p_rms_a_sq = amp_a[0, :].square()
        p_rms_a_sq.add_(torch.sum(amp_a[1:, :].square(), dim=0).mul_(2.0))
        self.oaspl = torch.sqrt(p_rms_a_sq).div_(p_ref).log10_().mul_(20.0)
        self.oaspl = self.oaspl.cpu().numpy()

    def get_observer_times(self, pos_fixed: torch.Tensor, observers: np.ndarray, src_times: torch.Tensor) -> torch.Tensor:
        """Compute retarded times for source-observer pairs.
        
        Calculates the time at which acoustic waves reach observers accounting for
        wave propagation delay: t_obs = t_source + r / a_inf
        
        Args:
            observers: Observer positions (num_obs, 3) with coordinates (x, y, z) in meters
            
        Returns:
            Retarded times tensor of shape (time, 1, 1, num_obs) in seconds
        """
        # Convert observer positions to tensor and add batch dimensions
        obs = torch.as_tensor(observers, dtype=self.dtype, device=self.device)[None, None, None, :, :]
        # Compute distance from source to observer
        dist = torch.linalg.norm(obs.sub(pos_fixed[..., None, :]), dim=-1)
        # Compute retarded time: tau_obs = tau_source + distance / sound_speed
        return torch.as_tensor(src_times, dtype=self.dtype, device=self.device)[:, None, None, None].add(dist.mul(torch.as_tensor(self.params.a_inf, dtype=self.dtype, device=self.device).reciprocal())).reshape(src_times.shape[0], -1, observers.shape[0])

    def _time_cuda(self, func, *args, label: str | None = None, **kwargs):
        """Time a callable using CUDA events when available, otherwise wall-clock.

        Returns the callable's result and prints timing in milliseconds.
        """
        name = label or getattr(func, "__name__", "call")
        use_cuda = torch.cuda.is_available() and ("cuda" in str(self.device))

        if use_cuda:
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

        import time

        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[CPU TIMING] {name}: {(t1 - t0) * 1000.0:.3f} ms")
        return result
    
    def _interp_bpm_vectorized(
            self,
            x_new: torch.Tensor,
            x_old: torch.Tensor,
            y_old: torch.Tensor,
            n_freq: int
        ) -> torch.Tensor:
        """
        Vectorized interpolation for BPM that preserves the frequency dimension.
       
        Args:
            x_new: Uniform observer grid (n_steps, n_obs)
            x_old: Arrival times (nt, n_freq * ns * nb, n_obs)
            y_old: Source power SPP (nt, n_freq * ns * nb, n_obs)
            n_freq: Number of frequency bins to preserve

        Returns:
            Interpolated SPP (n_freq, n_steps, n_obs)
        """
        nt, n_total_sources, no = x_old.shape
        n_steps = x_new.shape[0]


        # 1. Permute to (n_obs, n_sources, nt) to use searchsorted on the last dim
        x_old_p = x_old.permute(2, 1, 0).contiguous()
        y_old_p = y_old.permute(2, 1, 0).contiguous()


        # 2. Prepare x_new: (n_obs, n_sources, n_steps)
        x_new_p = x_new.T.unsqueeze(1).expand(-1, n_total_sources, -1).contiguous()

        # 3. Find indices for the arrival time windows
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, nt - 1)

        # 4. Gather the neighbor points for linear interpolation
        # These are all (n_obs, n_total_sources, n_steps)
        x0 = torch.gather(x_old_p, 2, idx - 1)
        x1 = torch.gather(x_old_p, 2, idx)
        y0 = torch.gather(y_old_p, 2, idx - 1)
        y1 = torch.gather(y_old_p, 2, idx)


        # 5. Compute the interpolated values (Linear)
        weights = (x_new_p - x0) / (x1 - x0 + 1e-12)
        interp_vals = y0 + weights * (y1 - y0)

        # 6. Unpack Frequency vs. Physical Source
        # Current interp_vals: (n_obs, n_freq * ns * nb, n_steps)
        ns_nb = n_total_sources // n_freq

        # Reshape to: (n_obs, n_freq, ns_nb, n_steps)
        reshaped = interp_vals.view(no, n_freq, ns_nb, n_steps)

        # 7. Sum across physical sources (ns_nb) only
        # Result: (n_obs, n_freq, n_steps)
        spp_summed = torch.sum(reshaped, dim=2)
        # 8. Permute to final shape: (n_freq, n_steps, n_obs)
        return spp_summed.permute(1, 2, 0)