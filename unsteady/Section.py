import numpy as np
import aerosandbox as asb
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class SectionForces:
    def __init__(self, airfoil, r, dr, chord, theta, v_inf, propellerParams):
        self.airfoil = airfoil
        self.r = r
        self.dr = dr
        self.chord = chord
        self.theta = np.radians(theta) if theta > np.pi else theta 
        self.v_inf = v_inf 
        self.propellerParams = propellerParams
        
        # --- Øye Model States (Axial Dynamics) ---
        self.v_int = 0.0      
        self.v_final = 0.0    
        self.v_qs_prev = 0.0  
        
        # --- Quasi-Steady Tangential State ---
        self.a_prime = 0.0    # Tangential induction factor
        
        self.Ma = 0.1
        self.Re = 1e5
        self._tables = {}
        self._buildPrandtlLossTable()

    @property
    def sigma(self):
        return self.propellerParams.n_blades * self.chord / (2 * np.pi * self.r)

    def _buildPrandtlLossTable(self):
        self._phi_grid = np.linspace(np.radians(0.1), np.radians(89.9), 100)
        B, R, Rh = self.propellerParams.n_blades, self.propellerParams.prop_radius, self.propellerParams.hub_radius
        sin_phi = np.sin(self._phi_grid)
        f_tip = B * (R - self.r) / (2 * self.r * sin_phi)
        f_hub = B * (self.r - Rh) / (2 * self.r * sin_phi)
        F_tip = 2 * np.arccos(np.exp(-np.clip(f_tip, 0, 500))) / np.pi
        F_hub = 2 * np.arccos(np.exp(-np.clip(f_hub, 0, 500))) / np.pi
        self._F_grid = F_tip * F_hub

    def prandtlLoss(self, phi):
        if phi <= 0: return 1.0
        return np.interp(phi, self._phi_grid, self._F_grid)

    def airfoilCoefficients(self, alpha, Re, Ma):
        ReBin, MaBin = round(Re, -4), round(Ma, 1)
        key = (ReBin, MaBin)
        if key not in self._tables:
            alpha_grid = np.linspace(-20.0, 30.0, 101)
            full_output = asb.Airfoil.get_aero_from_neuralfoil(self.airfoil, alpha=alpha_grid, Re=ReBin, mach=MaBin)
            self._tables[key] = (alpha_grid, np.asarray(full_output["CL"]), np.asarray(full_output["CD"]))
        grid, cl, cd = self._tables[key]
        return np.interp(alpha, grid, cl), np.interp(alpha, grid, cd)

    def step_forces(self, v_local_wind, azimuth_beta):
        p = self.propellerParams
        
        # 1. Axial Velocity (Lagged via Øye)
        v_axial = v_local_wind + self.v_final
        
        # 2. Tangential Velocity (Corrected by Quasi-Steady a_prime)
        v_tan = p.omega * self.r * (1 - self.a_prime)
        
        W = np.sqrt(v_axial**2 + v_tan**2)
        phi = np.arctan2(v_axial, v_tan)
        alpha = np.degrees(self.theta - phi)
        
        # Re/Ma lookup
        self.Re = p.rho * W * self.chord / p.mu
        self.Ma = W / p.a_inf
        cL, cD = self.airfoilCoefficients(alpha, self.Re, self.Ma)
        
        # Coefficients resolved to Disk Plane
        cL_p = cL * np.cos(phi) - cD * np.sin(phi)
        cD_p = cL * np.sin(phi) + cD * np.cos(phi)
        
        dT = 0.5 * p.rho * (W**2) * (cL_p * self.chord) * self.dr
        dQ = 0.5 * p.rho * (W**2) * (cD_p * self.chord) * self.r * self.dr
        
        return dT, dQ, W, phi, cD_p

    def update_oye_state(self, dt, dT_annulus, cDp_avg, W_avg, phi_avg):
        p = self.propellerParams
        R, rho = p.prop_radius, p.rho
        F = self.prandtlLoss(phi_avg)
        v_flow = max(self.v_inf + self.v_final, 0.1)
        
        # --- AXIAL DYNAMICS (Øye) ---
        v_qs = dT_annulus / (4 * np.pi * self.r * self.dr * rho * v_flow * F)
        
        a_axial = self.v_final / v_flow
        tau1 = (1.1 * R) / (max(W_avg, 0.1) * (1 - 1.3 * min(a_axial, 0.5)))
        tau2 = (0.39 * R) / (max(W_avg, 0.1) * (1 - 1.3 * min(a_axial, 0.5)))
        
        dv_qs_dt = (v_qs - self.v_qs_prev) / dt
        v_int_dot = (v_qs + 0.6 * tau1 * dv_qs_dt - self.v_int) / tau1
        self.v_int += v_int_dot * dt
        
        v_final_dot = (self.v_int - self.v_final) / tau2
        self.v_final += v_final_dot * dt
        self.v_qs_prev = v_qs

        # --- TANGENTIAL QUASI-STEADY UPDATE (a_prime) ---
        # Solving the steady relation: a' = 1 / [ (4 F sin phi cos phi) / (sigma cD') + 1 ]
        # This matches the 'aPrime' equation in your steady BEMT solver exactly.
        denom_a_prime = (4 * F * np.sin(phi_avg) * np.cos(phi_avg)) / (self.sigma * cDp_avg)
        self.a_prime = 1 / (denom_a_prime + 1)