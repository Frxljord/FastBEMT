import numpy as np
from .Propeller import Propeller

def St(f, L, U):
    return f[:, None]*L/U[None, :]

def St1(M):
    return 0.02*M**(-0.6)

def St2(M, alpha):    
    conditions = [
        alpha < 1.33,
        (alpha >= 1.33) & (alpha <= 12.5),
        alpha > 12.5
    ]
    choices = [
        1.0,
        10**(0.0054 * (alpha - 1.33)**2),
        4.72
    ]
    tmp = np.select(conditions, choices)
    return St1(M) * tmp

def St1_bar(M, alpha):
    return (St1(M)+St2(M,alpha))/2

def tbl_te_A_min(a):
    conds = [
        a < 0.204,
        (a >= 0.204) & (a <= 0.244),
        a > 0.244
    ]
    vals = [
        np.sqrt(np.maximum(0, 67.552 - 886.788 * a**2)) - 8.219,
        -32.665 * a + 3.981,
        -142.795 * a**3 + 103.656 * a**2 - 57.757 * a + 6.006
    ]
    return np.select(conds, vals)

def tbl_te_A_max(a):
    conds = [
        a < 0.13,
        (a >= 0.13) & (a <= 0.321),
        a > 0.321
    ]
    vals = [
        np.sqrt(67.552 - 886.788 * a**2) - 8.219,
        -15.901 * a + 1.098,
        -4.669 * a**3 + 3.491 * a**2 - 16.699 * a + 1.149
    ]
    return np.select(conds, vals)

def tbl_te_a0(Re_c):
    conds = [
        Re_c < 9.52e4,
        (Re_c >= 9.52e4) & (Re_c <= 8.57e5),
        Re_c > 8.57e5
    ]
    vals = [
        0.57,
        (-9.57e-13) * (Re_c - 8.57e5)**2 + 1.13,
        1.13
    ]
    return np.select(conds, vals)

def tbl_te_Ar(a0):
    return (-20 - tbl_te_A_min(a0)) / (tbl_te_A_max(a0) - tbl_te_A_min(a0))

def tbl_te_A(a, Re_c):
    a0 = tbl_te_a0(Re_c)
    return tbl_te_A_min(a) + tbl_te_Ar(a0) * (tbl_te_A_max(a) - tbl_te_A_min(a))

def tbl_te_B_min(b):
    conds = [
        b < 0.13,
        (b >= 0.13) & (b <= 0.145),
        b > 0.145
    ]
    vals = [
        np.sqrt(16.888 - 886.788 * b**2) - 4.109,
        -83.607 * b + 8.138,
        -817.810 * b**3 + 355.210 * b**2 - 135.024 * b + 10.619
    ]
    return np.select(conds, vals)

def tbl_te_B_max(b):
    conds = [
        b < 0.10,
        (b >= 0.10) & (b <= 0.187),
        b > 0.187
    ]
    vals = [
        np.sqrt(16.888 - 886.788 * b**2) - 4.109,
        -31.330 * b + 1.854,
        -80.541 * b**3 + 44.174 * b**2 - 39.381 * b + 2.344
    ]
    return np.select(conds, vals)

def tbl_te_b0(Re_c):
    conds = [
        Re_c < 9.52e4,
        (Re_c >= 9.52e4) & (Re_c <= 8.57e5),
        Re_c > 8.57e5
    ]
    vals = [
        0.30,
        (-4.48e-13) * (Re_c - 8.57e5)**2 + 0.56,
        0.56
    ]
    return np.select(conds, vals)

def tbl_te_Br(b0):
    return (-20 - tbl_te_B_min(b0)) / (tbl_te_B_max(b0) - tbl_te_B_min(b0))

def tbl_te_B(b, Re_c):
    b0 = tbl_te_b0(Re_c)
    return tbl_te_B_min(b) + tbl_te_Br(b0) * (tbl_te_B_max(b) - tbl_te_B_min(b))

def K1(Re_c):
    conds = [
        Re_c < 2.47e5,
        (Re_c >= 2.47e5) & (Re_c <= 8e5),
        Re_c > 8e5
    ]
    vals = [
        -4.31 * np.log10(Re_c) + 156.3,
        -9.0 * np.log10(Re_c) + 181.6,
        128.5
    ]
    return np.select(conds, vals)

def delta_K1(Re_dp, alpha):
    return np.where(Re_dp <= 5000, alpha * (1.43 * np.log10(Re_dp) - 5.29), 0.0)

def K2(Re_c, M, alpha):  
    gamma = 27.094 * M + 3.31
    gamma0 = 23.43 * M + 4.651
    beta = 72.65 * M + 10.74
    beta0 = -34.19 * M - 13.82
    
    conds = [
        alpha < gamma0 - gamma,
        (alpha >= gamma0 - gamma) & (alpha <= gamma0 + gamma),
        alpha > gamma0 + gamma
    ]
   
    vals = [
        -1000.0,
        np.sqrt(beta**2 - (beta/gamma)**2 * (alpha - gamma0)**2) + beta0,
        -12.0
    ]
    
    tmp = np.select(conds, vals)
    tmp = np.where((alpha > gamma0) & (tmp < -12), -12, tmp)
    return K1(Re_c) + tmp

import numpy as np

def calculate_directivity(r_element, r_obs, M, type):
    """
    Vectorized version of directivity calculation.
    
    Shapes:
    r_element: (Na, Nb, Nr, 3)
    r_obs:     (No, 3)
    M: (Nr,)
    """
    R_vec = r_obs[:, np.newaxis, np.newaxis, np.newaxis, :] - r_element[np.newaxis, ...]
    R_mag = np.linalg.norm(R_vec, axis=-1)
    x = R_vec[..., 0]
    y = R_vec[..., 1]
    z = R_vec[..., 2]
    phi = np.arctan2(x, y)
    num_theta = y * np.cos(phi) + x * np.sin(phi)
    theta = np.arctan2(num_theta, -z)
    cos_theta = np.cos(theta)
    if type == 'TE':
        term1 = 2 * (np.sin(theta / 2)**2) * (np.sin(phi)**2)
        term2 = (1 + M * cos_theta) * (1 + 0.2 * M * cos_theta)**2
        Dh = term1 / term2
    elif type == 'LE':
        term1 = 2 * (np.cos(theta / 2)**2) * (np.sin(phi)**2)
        term2 = (1 + M * cos_theta)**3
        Dh = term1 / term2
    elif type == 'low':
        term1 = np.sin(theta)**2 * np.sin(phi)**2
        term2 = (1 + M * cos_theta)**4
        Dh = term1 / term2
    return R_mag, Dh

def St1_prime(Re_c):
    conds = [
        Re_c <= 1.3e5,
        (Re_c > 1.3e5) & (Re_c <= 4e5),
        Re_c > 4e5
    ]
    vals = [
        0.18,
        0.001756 * Re_c**0.3931,
        0.28
    ]
    return np.select(conds, vals)

def St_peak_prime(Re_c, alpha):
    return St1_prime(Re_c) * 10**(-0.04 * alpha)

def G1(e):
    log_e = np.log10(e)
    conds = [
        e <= 0.5974,
        (e > 0.5974) & (e <= 0.8545),
        (e > 0.8545) & (e <= 1.17),
        (e > 1.17) & (e <= 1.674),
        e > 1.674
    ]
    vals = [
        39.8 * log_e - 11.12,
        98.409 * log_e + 2.0,
        -5.076 + np.sqrt(np.maximum(0, 2.484 - 506.25 * log_e**2)),
        -98.409 * log_e + 2.0,
        -39.8 * log_e - 11.12
    ]
    return np.select(conds, vals)

def G2(d):
    log_d = np.log10(d)
    conds = [
        d <= 0.3237,
        (d > 0.3237) & (d <= 0.5689),
        (d > 0.5689) & (d <= 1.7579),
        (d > 1.7579) & (d <= 3.0889),
        d > 3.0889
    ]
    vals = [
        77.852 * log_d + 15.328,
        65.188 * log_d + 9.125,
        -114.052 * log_d**2,
        -65.188 * log_d + 9.125,
        -77.852 * log_d + 15.328
    ]
    return np.select(conds, vals)

def Re_c0(alpha):
    conds = [
        alpha <= 3.0,
        alpha > 3.0
    ]
    vals = [
        10**(0.215 * alpha + 4.978),
        10**(0.120 * alpha + 5.263)
    ]
    return np.select(conds, vals)

def G3(alpha):
    return 171.04 - 3.03 * alpha

def calc_delta_avg(delta_p, delta_s):
    return (delta_p+delta_s)/2

def St_peak_3prime(q, psi):
    # psi in degrees
    conds = [
        q < 0.2,
        q >= 0.2
    ]
    vals = [
        0.1*q + 0.095 - 0.00243*psi,
        (0.212 - 0.0045*psi)/(1 + 0.235/q - 0.0132/q**2)
    ]
    return np.select(conds, vals)

def G4(q, psi):
    conds = [
        q <= 5,
        q > 5
    ]
    vals = [
        17.5*np.log10(q) + 157.5 - 1.114*psi,
        169.7 - 1.114*psi
    ]
    return np.select(conds, vals)


def calc_mu(q):
    conds = [
        q < 0.25,
        (q >= 0.25) & (q < 0.62),
        (q >= 0.62) & (q < 1.15),
        q >= 1.15
    ]
    vals = [
        0.1221,
        -0.2175*q + 0.1755,
        -0.0308*q + 0.0596,
        0.0242
    ]
    return np.select(conds, vals)

def calc_m(q):
    conds = [
        q <= 0.02,
        (q > 0.02) & (q <= 0.5),
        (q > 0.5) & (q <= 0.62),
        (q > 0.62) & (q <= 1.15),
        (q > 1.15) & (q <= 1.2),
        q > 1.2
    ]
    vals = [
        0,
        68.724*q - 1.35,
        308.475*q - 121.23,
        224.811*q - 69.35,
        1583.28*q - 1631.59,
        268.344
    ]
    return np.select(conds, vals)

def calc_eta0(m, mu):
    return -np.sqrt(m**2 * mu**4 / (6.25 + m**2 * mu**2))

def calc_k(m, mu, eta0):
    return 2.5*np.sqrt(1-(eta0/mu)**2) - 2.5 - m*eta0

def G5(m, mu, eta, eta0, k):
    conds = [
        eta < eta0,
        (eta >= eta0) & (eta < 0),
        (eta >= 0) & (eta < 0.03616),
        eta >= 0.03616
    ]
    vals = [
        m*eta + k,
        2.5*np.sqrt(1-(eta/mu)**2) - 2.5,
        np.sqrt(1.5625 - 1194.99*eta**2) - 1.25,
        -155.543*eta + 4.375
    ]
    return np.select(conds, vals)

def G5_0(q, eta):
    q_0 = 6.724*q**2 - 4.019*q + 1.107
    m = calc_m(q_0)
    mu = calc_mu(q_0)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    return G5(m, mu, eta, eta0, k)

def G5_tot(q, eta, psi):
    g5_0 = G5_0(q, eta)
    m = calc_m(q)
    mu = calc_mu(q)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    g5 = G5(m, mu, eta, eta0, k)
    return g5_0 + 0.0714*psi*np.where(g5 - g5_0 < 0, g5 - g5_0, 0)

def calc_l_tip(chord, alpha_tip):
    return chord * 0.008 * alpha_tip

def Phi_ww(f, U, sigma, L):
    """von Karman Turbulence Spectrum for vertical velocity fluctuations."""
    k1 = 2 * np.pi * f / U
    num = (2 * sigma**2 * L / np.pi)
    denom = (1 + (1.339 * L * k1)**2)**(5/6)
    return num/denom

def Amiet_L(omega, U, L, y, sigma, a_inf, xi):
    return 2 * L * (xi * omega / U) / ((xi * omega / U)**2 + (omega * y / a_inf / sigma)**2)

class BPM:
    def __init__(self, propeller : Propeller, frequencies, num_azim=360):
        self.frequencies = frequencies
        self.propeller = propeller
        self.aero_params = propeller.aero_params
        self.acoustic_params = propeller.acoustic_params
        self.v_inf = propeller.v_inf
        self.r = propeller.solution_data['r'].values
        self.dr = propeller.solution_data['dr'].values
        self.chord = propeller.solution_data['chord'].values
        self.alpha = propeller.solution_data['alpha'].values  # alpha in degrees
        self.vi = propeller.solution_data['u'].values
        self.U = propeller.solution_data['W'].values          # Local velocity
        self.Re_c = propeller.solution_data['Re'].values      # Reynolds number
        self.M = propeller.solution_data['Ma'].values         # Mach number
        self.delta_p = propeller.solution_data['dp'].values   # Boundary layer thickness (pressure side)
        self.delta_s = propeller.solution_data['ds'].values   # Boundary layer thickness (suction side)
        self.psi = propeller.geometry['boat_tail_angle']
        self.azimuth_positions = np.linspace(0, 2*np.pi, num_azim, endpoint=False)

        y_el = self.r[None, None, :] * np.cos(self.azimuth_positions[:, None, None] + propeller.aero_params.blade_angles[None, :, None])
        z_el = self.r[None, None, :] * np.sin(self.azimuth_positions[:, None, None] + propeller.aero_params.blade_angles[None, :, None])

        self.r_element = np.zeros((*y_el.shape, 3))
        self.r_element[..., 1] = y_el
        self.r_element[..., 2] = z_el

        self.R, self.Dh_TE = calculate_directivity(self.r_element, propeller.observer_positions, self.M, type = 'TE')
        _, self.Dh_LE = calculate_directivity(self.r_element, propeller.observer_positions, self.M, type = 'LE')
        _, self.Dl = calculate_directivity(self.r_element, propeller.observer_positions, self.M, type = 'low')

        self.base_val_TE = (self.M**5 * self.dr * self.Dh_TE) / (self.R**2)
        self.base_val_LE = (self.M**5 * self.dr * self.Dh_LE) / (self.R**2)
        self.base_val_low = (self.M**5 * self.dr * self.Dl) / (self.R**2)

    def run_BPM(self, z=1000, z0=1e-6, I=0.005, alpha_stall=15, h=None):
        self.TI_noise(z, z0, I)
        self.TBL_noise(alpha_stall)
        self.LBL_noise()
        self.TEB_noise(h)
        self.TV_noise()

    def TI_noise(self, z, z0, I):
        Lt = 25*z**0.35*z0**(-0.063)
        f_co = 10*self.U/np.pi/self.chord
        freq_4d = self.frequencies[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        f_co_4d = f_co[np.newaxis, np.newaxis, np.newaxis, :]
        Dh_avg = (self.Dh_TE + self.Dh_LE) / 2
        D = np.where(freq_4d < f_co_4d, self.Dl, Dh_avg).squeeze()
        k1 = 2*np.pi*self.frequencies[:, np.newaxis]/self.U[np.newaxis, :]
        k1_bar = k1*self.chord/2
        ke = 3/4/Lt
        k1_hat = k1/ke
        k1_hat = k1_hat[:, np.newaxis, np.newaxis, :]
        base_val_ti = (self.M**5 * self.dr * D) / (self.R**2)
        beta = 1 - self.M**2
        S_sq = 1/(2*np.pi*k1_bar/beta**2 + 1/(1+2.4*k1_bar/beta**2))
        LFC = 10*S_sq * self.M * k1_bar**(-2) * beta**(-2)
        LFC = LFC[:, np.newaxis, np.newaxis, :]

        SPL_TI = 10 * np.log10((self.aero_params.rho**2 * self.aero_params.a_inf**4 * Lt / 2 * I**2 * k1_hat**3 / (1 + k1_hat**2)**(7/3) * base_val_ti) + 1e-12) + 78.4 + 10 * np.log10(1 + 9*self.alpha[np.newaxis, np.newaxis, np.newaxis, :]**2) + 10 * np.log10(LFC/(1+LFC))
        SPP_TI = np.zeros_like(SPL_TI)
        SPP_TI += 10**(SPL_TI/10)
        self.SPL_TI = 10 * np.log10(SPP_TI.sum(axis=(2,3)).mean(axis=-1))

    def TEB_noise(self, h):
        if h is None:
            h = 1e-2*self.chord
        delta_avg = calc_delta_avg(self.delta_p, self.delta_s)
        q = h/delta_avg
        St_3prime = St(self.frequencies, h, self.U)
        St_3prime_peak = St_peak_3prime(q, self.psi)
        eta = np.log10(St_3prime/St_3prime_peak)
        m = calc_m(q)
        mu = calc_mu(q)
        eta0 = calc_eta0(m, mu)
        k = calc_k(m, mu, eta0)

        SPL_TEB = 10 * np.log10((h * self.base_val_TE * self.M**0.5) + 1e-12) + G4(q, self.psi)[np.newaxis, np.newaxis, np.newaxis, :] + G5_tot(q, eta, self.psi)[:, np.newaxis, np.newaxis, :]
        # SPL_TEB = 10 * np.log10((h * self.base_val_TE * self.M**0.5) + 1e-12) + G4(q, self.psi)[np.newaxis, np.newaxis, np.newaxis, :] + G5(m, mu, eta, eta0, k)[:, np.newaxis, np.newaxis, :]
        SPP_TEB = np.zeros_like(SPL_TEB)
        SPP_TEB += 10**(SPL_TEB/10)
        self.SPL_TEB = 10 * np.log10(SPP_TEB.sum(axis=(2,3)).mean(axis=-1))
    
    def LBL_noise(self):
        e = St(self.frequencies, self.delta_p, self.U)/St_peak_prime(self.Re_c, self.alpha)
        d = self.Re_c/Re_c0(self.alpha)

        SPL_LBL = 10 * np.log10((self.delta_p * self.base_val_LE) + 1e-12) + G1(e)[:, np.newaxis, np.newaxis, :] + G2(d)[np.newaxis, np.newaxis, np.newaxis, :] + G3(self.alpha)[np.newaxis, np.newaxis, np.newaxis, :]
        SPP_LBL = np.zeros_like(SPL_LBL)
        SPP_LBL += 10**(SPL_LBL/10)
        self.SPL_LBL = 10 * np.log10(SPP_LBL.sum(axis=(2,3)).mean(axis=-1))
    
    def TBL_noise(self, alpha_stall):
        st_1 = St1(self.M)
        st_2 = St2(self.M, self.alpha)
        K_1 = K1(self.Re_c)
        K_2 = K2(self.Re_c, self.M, self.alpha)
        delta_k1_val = delta_K1((self.Re_c/self.chord*self.delta_p), self.alpha)
        st_p = St(self.frequencies, self.delta_p, self.U)
        st_s = St(self.frequencies, self.delta_s, self.U)

        As = tbl_te_A(np.abs(np.log10(st_s/st_1)), self.Re_c)
        Ap = tbl_te_A(np.abs(np.log10(st_p/st_1)), self.Re_c)
        B = tbl_te_B(np.abs(np.log10(st_s/st_2)), self.Re_c)
        A_prime = tbl_te_A(np.abs(np.log10(st_s/st_2)), 3*self.Re_c)

        As_exp = As[:, np.newaxis, np.newaxis, :]
        Ap_exp = Ap[:, np.newaxis, np.newaxis, :]
        B_exp  = B[:, np.newaxis, np.newaxis, :]
        A_prime_exp = A_prime[:, np.newaxis, np.newaxis, :]

        SPL_S = np.where(self.alpha < alpha_stall, 10 * np.log10((self.delta_s * self.base_val_TE) + 1e-12) + As_exp + K_1 - 3, -np.inf)
        SPL_P = np.where(self.alpha < alpha_stall, 10 * np.log10((self.delta_p * self.base_val_TE) + 1e-12) + Ap_exp + K_1 - 3 + delta_k1_val, -np.inf)
        SPL_a = np.where(self.alpha < alpha_stall, 10 * np.log10((self.delta_s * self.base_val_TE) + 1e-12) + B_exp + K_2, 10 * np.log10((self.delta_s * self.base_val_low) + 1e-12) + A_prime_exp + K_2)
        SPP_TBL = np.zeros_like(SPL_a)
        SPP_TBL += (10**(SPL_S/10) + 10**(SPL_P/10) + 10**(SPL_a/10))
        self.SPL_TBL = 10 * np.log10(SPP_TBL.sum(axis=(2,3)).mean(axis=-1))
    
    def TV_noise(self):
        l_tip = calc_l_tip(self.chord[-1], self.alpha[-1])
        M_Max = self.M[-1] * (1 + 0.036*self.alpha[-1])
        St_2prime = St(self.frequencies, l_tip, np.atleast_1d(self.acoustic_params.a_inf * M_Max))
        SPL_TIP = 10 * np.log10(self.M[-1]**2 * M_Max ** 3 * l_tip ** 2 * self.Dh_TE[..., -1] / self.R[..., -1] ** 2) - 30.5 * (np.log10(St_2prime[:, :, np.newaxis]) + 0.3) ** 2 + 126
        SPP_TIP = np.zeros_like(SPL_TIP) 
        SPP_TIP += 10**(SPL_TIP/10)
        self.SPL_TIP = 10 * np.log10(SPP_TIP.sum(axis=-1).mean(axis=-1))

    # Inside your BPM Class:
    def BWI_noise(self, xi=1, b=1, sigma_turb=0.1, L_scale=0.125):
        """
        Computes Blade-Wake Interaction (BWI) noise.
        
        sigma_factor: Turbulence intensity as fraction of induced velocity (BEMT v_i)
        Lambda_factor: Integral length scale as fraction of chord
        h_miss: Vertical distance from blade to wake center (m)
        """
        vT = np.sqrt(self.U ** 2 - (self.v_inf + self.vi) ** 2)
        y_vT = vT[None, None, :] * np.sin(self.azimuth_positions[:, None, None] + self.propeller.aero_params.blade_angles[None, :, None])
        z_vT = - vT[None, None, :] * np.cos(self.azimuth_positions[:, None, None] + self.propeller.aero_params.blade_angles[None, :, None])
        vT = np.stack([np.zeros_like(y_vT), y_vT, z_vT], axis=-1)        
        R_vec = self.propeller.observer_positions[:, np.newaxis, np.newaxis, np.newaxis, :] - self.r_element[np.newaxis, ...]
        R_mag = np.linalg.norm(R_vec, axis=-1)
        sigma = R_mag ** 2 * (1 - self.M * np.sum(vT * R_vec, axis=-1))

        f = self.frequencies[:, np.newaxis, np.newaxis, np.newaxis]
        U = self.U[np.newaxis, np.newaxis, np.newaxis, :]

        omega = 2 * np.pi * f
        Phi = Phi_ww(f, U, sigma_turb, L_scale)
        L_sq = Amiet_L(omega, U, self.dr, R_vec[..., 1], sigma, self.aero_params.a_inf, xi)
        Spp_BWI = (omega * R_vec[..., 0] / 4 / np.pi / self.aero_params.a_inf / sigma**2)**2 * b * self.chord * Phi * L_sq
        self.SPL_BWI = 10 * np.log10(Spp_BWI.sum(axis=(2,3)).mean(axis=-1))