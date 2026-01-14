import numpy as np
import pandas as pd
import aerosandbox as asb
from typing import List, Tuple 

from .Section import SectionForces
from .CompactSource import CompactAcousticSourceArray
from .JobParameters import *

class Propeller:
    """High-level propeller BEMT and F1A acoustic analysis controller."""

    def __init__(self, propeller_geometry: dict, aero_params: AerodynamicParameters, acoustic_params: AcousticParameters):
        """Initialize analysis with geometry and global parameter objects."""
        self.geometry = propeller_geometry
        self.aero_params = aero_params
        self.acoustic_params = acoustic_params
        
        self.set_section_areas()

        # com_shift_up positive in forward direction (+ x), com_shift_forward positive in upward direction (+ z)
        self.com_shift_up = [s[1] for s in self.geometry['COM_shift']]
        self.com_shift_forward = [-s[0] for s in self.geometry['COM_shift']]

        self.solution_data = pd.DataFrame(
            columns=[
                "radius", "chord", "twist", "phi", "alpha", 
                "Cl", "Cd", "u", "a_prime", "dT", "dQ", 
                "F", "W", "Re", "Ma"
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
            phi, dT, dQ, alpha, u, a_prime, cl, cd, f, w, re, ma = self._sections[section_index].solve(v_inf, prevPhi=prev_phi)
            return [self.geometry['r'][section_index], self.geometry['chord'][section_index], self.geometry['twist'][section_index], np.degrees(phi), alpha, cl, cd, u, a_prime, dT, dQ, f, w, re, ma]
        except RuntimeError as e:
            print(f"Error in section at r={self.geometry['r'][section_index]}: {e}")
            return [np.nan] * 15

    def run_bemt(self, v_inf):
        """Execute the BEMT radial sweep sequentially."""
        prev_phi = None
        rows = []

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

    def set_section_areas(self):
        """Calculate physical cross-sectional areas (m^2) using the Shoelace formula."""
        areas = []
        for idx, coords in enumerate(self.geometry["airfoil"]):
            x, y = coords[:, 0], coords[:, 1]
            # Normalized area * chord^2 to get physical area
            a_norm = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
            areas.append(a_norm * self.geometry['chord'][idx]**2)
        self.geometry["cross_section"] = areas

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
        
    def run_compact_f1a(self, observer_positions, local_dT=None, local_dQ=None):
        """Initialize the acoustic array and compute total pressure at observers."""
        # Force per unit span logic
        if local_dT is None or local_dQ is None:
            local_dT = self.solution_data['dT'].values / self.geometry['dr'] / self.aero_params.n_blades
            local_dQ = self.solution_data['dQ'].values / self.geometry['dr'] / self.geometry['r'] / self.aero_params.n_blades
            local_dT = np.broadcast_to(local_dT[None, None, :], (self.acoustic_params.num_src_times, self.aero_params.n_blades, len(self.geometry['r'])))
            local_dQ = np.broadcast_to(local_dQ[None, None, :], (self.acoustic_params.num_src_times, self.aero_params.n_blades, len(self.geometry['r'])))
        else:
            local_dT = local_dT / self.geometry['dr']
            local_dQ = local_dQ / self.geometry['dr'] / self.geometry['r']

        acoustic_array = CompactAcousticSourceArray(
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
        
        self.combine_sources(
            obs_times=obs_times, 
            f1a_output=f1a_output, 
            t_range=self.acoustic_params.observer_time_range, 
            n_steps=self.acoustic_params.num_obs_times
        )