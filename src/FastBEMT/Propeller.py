import numpy as np
import pandas as pd
import aerosandbox as asb
import torch
from typing import List, Tuple 

from .Section import SectionForces
from .CompactSource import CompactAcousticSourceArray
from .TorchCompactSource import TorchCompactAcousticSourceArray
from .JobParameters import *

class Propeller:
    """High-level propeller BEMT and F1A acoustic analysis controller."""

    def __init__(self, propeller_geometry: dict, aero_params: AerodynamicParameters, acoustic_params: AcousticParameters):
        """Initialize analysis with geometry and global parameter objects."""
        self.geometry = propeller_geometry
        self.aero_params = aero_params
        self.acoustic_params = acoustic_params
        self.section_areas()
        self.calculate_boat_tail_angle()

        # com_shift_up positive in forward direction (+ x), com_shift_forward positive in upward direction (+ z)
        self.com_shift_up = [s[1] for s in self.geometry['COM_shift']]
        self.com_shift_forward = [-s[0] for s in self.geometry['COM_shift']]

        self.solution_data = pd.DataFrame(
            columns=[
                "r", "dr", "chord", "twist", "phi", "alpha", 
                "Cl", "Cd", "u", "a_prime", "dT", "dQ", 
                "F", "W", "Re", "Ma", "ds", "dp"
            ]
        )
        
        self._section_airfoils = [asb.Airfoil(coordinates=af) for af in self.geometry["airfoil"]]

        self._sections = [SectionForces(
            airfoil=self._section_airfoils[idx],
            r=self.geometry['r'][idx],
            dr=self.geometry['dr'][idx],
            chord=self.geometry['chord'][idx],
            theta=np.radians(self.geometry['twist'][idx]),
            propellerParams=self.aero_params,
        ) for idx in range(len(self.geometry["r"]))]

        # Acoustic results containers
        self.t, self.p_m, self.p_d, self.p_tot = None, None, None, None
        self.freq, self.spl, self.spl_a, self.ospl, self.oaspl = [None] * 5

    def process_section(self, section_index, v_inf, prev_phi=None):
        """Run the BEMT solver for a single radial section."""
        try:
            phi, dT, dQ, alpha, u, a_prime, cl, cd, f, w, re, ma, delta_star_upper, delta_star_lower = self._sections[section_index].solve(v_inf, prevPhi=prev_phi)
            return [self.geometry['r'][section_index], self.geometry['dr'][section_index], self.geometry['chord'][section_index], self.geometry['twist'][section_index], np.degrees(phi), alpha, cl, cd, u, a_prime, dT, dQ, f, w, re, ma, delta_star_upper[0], delta_star_lower[0]]
        except RuntimeError as e:
            print(f"Error in section at r={self.geometry['r'][section_index]}: {e}")
            return [np.nan] * 15

    def run_bemt(self, v_inf):
        """Execute the BEMT radial sweep sequentially."""
        self.v_inf = v_inf
        prev_phi = None
        rows = []
        if isinstance(v_inf, (int, float)):
            v_inf = np.full(len(self.geometry["r"]), v_inf)
        for i, (r, dr, chord, twist, af_asb) in enumerate(zip(
            self.geometry["r"], self.geometry["dr"], self.geometry["chord"], 
            self.geometry["twist"], self._section_airfoils
            )):
            res = self.process_section(i, v_inf[i], prev_phi=prev_phi)
            if not np.isnan(res[3]): # index 3 is phiDeg
                prev_phi = np.radians(res[3])
            
            rows.append(res)

        self.solution_data = pd.DataFrame(rows, columns=self.solution_data.columns)

    def compute_total_forces(self) -> Tuple[float, float, float, float]:
        """Compute total thrust, torque, and nondimensional coefficients Ct, Cp."""
        total_t = self.solution_data["dT"].sum()
        total_q = self.solution_data["dQ"].sum()
        
        n_rev_s = self.aero_params.rpm / 60.0
        d = self.aero_params.prop_diameter
        rho = self.aero_params.rho

        ct = total_t / (rho * n_rev_s**2 * d**4)
        cp = (2 * np.pi * total_q) / (rho * n_rev_s**2 * d**5)
        
        return total_t, total_q, ct, cp

    def section_areas(self):
        """Calculate cross-sectional areas (m^2) using the Shoelace formula."""
        areas = []
        for idx, coords in enumerate(self.geometry["airfoil"]):
            x, y = coords[:, 0], coords[:, 1]
            a_norm = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
            areas.append(a_norm * self.geometry['chord'][idx]**2)
        self.geometry["cross_section"] = np.array(areas)

    def calculate_boat_tail_angle(self):
        angles = []
        for idx, coords in enumerate(self.geometry["airfoil"]):
            le_idx = np.argmin(coords[:, 0])
            upper = coords[:le_idx + 1]
            lower = coords[le_idx:]
            
            u_pts = upper[upper[:, 0] > 0.95]
            l_pts = lower[lower[:, 0] > 0.95]
            
            if len(u_pts) < 2 or len(l_pts) < 2:
                return 0.0
                
            m_u, _ = np.polyfit(u_pts[:, 0], u_pts[:, 1], 1)
            m_l, _ = np.polyfit(l_pts[:, 0], l_pts[:, 1], 1)
            
            angles.append(np.degrees(np.abs(np.arctan(m_u) - np.arctan(m_l))))
        self.geometry["boat_tail_angle"] = np.array(angles)

    def combine_sources(self, obs_times, f1a_output, t_range, n_steps):
        """Interpolate source signals onto the uniform observer time grid."""
        nt, ns, nb, no, _ = f1a_output.shape
        t_raw = obs_times.reshape(nt, -1, no)
        pm_raw, pd_raw = f1a_output[..., 0].reshape(nt, -1, no), f1a_output[..., 1].reshape(nt, -1, no)

        t_start = np.max(t_raw[0, :, :], axis=0)
        self.t = t_start[None, :] + np.linspace(0, t_range, n_steps)[:, None]
        
        self.p_m, self.p_d = np.zeros((n_steps, no)), np.zeros((n_steps, no))

        for o in range(no):
            for s in range(ns * nb):
                self.p_m[:, o] += np.interp(self.t[:, o], t_raw[:, s, o], pm_raw[:, s, o])
                self.p_d[:, o] += np.interp(self.t[:, o], t_raw[:, s, o], pd_raw[:, s, o])

        self.p_tot = (self.p_m + self.p_d)
        self.p_tot -= np.mean(self.p_tot, axis=0)
        self._perform_spectral_analysis()

    def _perform_spectral_analysis(self):
        """Compute FFT, SPL, OSPL, and A-weighted metrics."""
        n, no = self.t.shape
        dt = self.t[1, :] - self.t[0, :]
        f_single = np.fft.rfftfreq(n, dt[0])
        self.freq = np.tile(f_single[:, None], (1, no))
        
        fft_p = np.fft.rfft(self.p_tot, axis=0)
        self.fft_amp = np.abs(fft_p) * np.sqrt(2) / n
        self.fft_amp[0, :] /= np.sqrt(2) 

        self.spl = 20 * np.log10(np.maximum(self.fft_amp, 1e-15) / self.acoustic_params.p_ref)
        
        def r_a_func(f):
            f_sq = f**2
            return (12194**2 * f_sq**2) / (
                (f_sq + 20.6**2) * np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + 12194**2)
            )
        
        a_weight = 20 * np.log10(r_a_func(self.freq)) - 20 * np.log10(r_a_func(1000))
        self.spl_a = self.spl + a_weight

        # RMS-based Overall levels
        p_rms = np.sqrt(self.fft_amp[0, :]**2 + 2 * np.sum(self.fft_amp[1:, :]**2, axis=0))
        self.ospl = 20 * np.log10(p_rms / self.acoustic_params.p_ref)

        amp_a = 10 ** (self.spl_a / 20) * self.acoustic_params.p_ref
        p_rms_a = np.sqrt(amp_a[0, :]**2 + 2 * np.sum(amp_a[1:, :]**2, axis=0))
        self.oaspl = 20 * np.log10(p_rms_a / self.acoustic_params.p_ref)
        
    def run_compact_f1a(self, observer_positions, local_dT=None, local_dQ=None, use_GPU=False):
        self.observer_positions = observer_positions
        """Initialize the acoustic array and compute total pressure at observers."""
        if local_dT is None or local_dQ is None:
            local_dT = self.solution_data['dT'].values / self.geometry['dr'] / self.aero_params.n_blades
            local_dQ = self.solution_data['dQ'].values / self.geometry['dr'] / self.geometry['r'] / self.aero_params.n_blades
            local_dT = np.broadcast_to(local_dT[None, None, :], (self.acoustic_params.num_src_times, self.aero_params.n_blades, len(self.geometry['r'])))
            local_dQ = np.broadcast_to(local_dQ[None, None, :], (self.acoustic_params.num_src_times, self.aero_params.n_blades, len(self.geometry['r'])))
        else:
            local_dT = local_dT / self.geometry['dr']
            local_dQ = local_dQ / self.geometry['dr'] / self.geometry['r']
        ArrayClass = TorchCompactAcousticSourceArray if use_GPU else CompactAcousticSourceArray
        acoustic_array = ArrayClass(
            self.aero_params.rho, 
            self.aero_params.a_inf,
            self.geometry['r'],
            self.geometry['dr'],
            self.geometry['cross_section'],
            self.geometry['chord'],
            self.geometry['twist'],
            self.com_shift_forward,
            self.com_shift_up,
            self.acoustic_params.src_times,
            self.acoustic_params.omega,
            local_dT,
            local_dQ,
            self.aero_params.blade_angles
        )
        obs_times = acoustic_array.get_observer_times(observer_positions)
        f1a_output = acoustic_array.calculate_f1a_pressure(observer_positions)
        if use_GPU:
            self.combine_sources_torch(
                obs_times=obs_times, 
                f1a_output=f1a_output, 
                t_range=self.acoustic_params.observer_time_range, 
                n_steps=self.acoustic_params.num_obs_times
            )
            self.t = self.t.cpu().numpy()
            self.p_d = self.p_d.cpu().numpy()
            self.p_m = self.p_m.cpu().numpy()
            self.p_tot = self.p_tot.cpu().numpy()
            self.freq = self.freq.cpu().numpy()
            self.spl = self.spl.cpu().numpy()
        else:
            self.combine_sources(
                obs_times=obs_times, 
                f1a_output=f1a_output, 
                t_range=self.acoustic_params.observer_time_range, 
                n_steps=self.acoustic_params.num_obs_times
            )

    def combine_sources_torch(self, obs_times, f1a_output, t_range, n_steps):
        if not torch.is_tensor(obs_times):
            obs_times = torch.tensor(obs_times, dtype=torch.float32, device="cuda")
        if not torch.is_tensor(f1a_output):
            f1a_output = torch.tensor(f1a_output, dtype=torch.float32, device="cuda")

        device = obs_times.device
        nt, ns, nb, no, _ = f1a_output.shape
        
        t_raw = obs_times.reshape(nt, -1, no) # (Nt, Ns*Nb, No)
        pm_raw = f1a_output[..., 0].reshape(nt, -1, no)
        pd_raw = f1a_output[..., 1].reshape(nt, -1, no)

        t_start, _ = torch.max(t_raw[0, :, :], dim=0) 
        
        self.t = t_start[None, :] + torch.linspace(0, t_range, n_steps, device=device)[:, None]
        
        self.p_m = self._interp_tensor_vectorized(self.t, t_raw, pm_raw)
        self.p_d = self._interp_tensor_vectorized(self.t, t_raw, pd_raw)

        self.p_tot = (self.p_m + self.p_d)
        self.p_tot -= torch.mean(self.p_tot, dim=0)
        
        self._perform_spectral_analysis_torch()

    def _interp_tensor_vectorized(self, x_new, x_old, y_old):
        """
        Full 3D vectorized interpolation.
        x_new: (N_steps, No)
        x_old: (Nt, N_sources, No)
        y_old: (Nt, N_sources, No)
        """
        nt, n_src, _ = x_old.shape

        # Permute and reshape to align for searchsorted: (No, N_src, Nt)
        # searchsorted operates on the last dimension
        x_old_p = x_old.permute(2, 1, 0).contiguous()
        y_old_p = y_old.permute(2, 1, 0).contiguous()
        
        # Prepare x_new for searchsorted: (No, N_src, N_steps)
        x_new_p = x_new.T.unsqueeze(1).expand(-1, n_src, -1).contiguous()
        
        # Find indices across all observers and sources at once
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, nt - 1)
        
        # Linear interpolation math
        # We gather from the Nt dimension (dim=2)
        x0 = torch.gather(x_old_p, 2, idx - 1)
        x1 = torch.gather(x_old_p, 2, idx)
        y0 = torch.gather(y_old_p, 2, idx - 1)
        y1 = torch.gather(y_old_p, 2, idx)
        
        weights = (x_new_p - x0) / (x1 - x0 + 1e-12)
        interp_vals = y0 + weights * (y1 - y0) # (No, N_src, N_steps)
        
        # Sum across sources (dim=1) and return as (N_steps, No)
        return torch.sum(interp_vals, dim=1).T
    
    def _perform_spectral_analysis_torch(self):
        n, no = self.t.shape
        device = self.t.device
        
        # Frequency calculation
        dt = (self.t[1, 0] - self.t[0, 0]).item()
        f_single = torch.fft.rfftfreq(n, dt).to(device)
        self.freq = f_single.unsqueeze(1).expand(-1, no)
        
        # FFT and Amplitude
        fft_p = torch.fft.rfft(self.p_tot, dim=0)
        self.fft_amp = torch.abs(fft_p)
        self.fft_amp.mul_(np.sqrt(2) / n) 
        self.fft_amp[0, :].div_(np.sqrt(2)) 

        # SPL Calculation
        p_ref = self.acoustic_params.p_ref
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

        r_a_1000 = r_a_func_torch(torch.tensor(1000.0, device=device))
        
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