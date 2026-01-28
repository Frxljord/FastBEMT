import numpy as np
from .Propeller import Propeller


def st(f: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    return f[:, None] * l / u[None, :]


def st1(m: np.ndarray) -> np.ndarray:
    return 0.02 * m ** (-0.6)


def st2(m: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    conditions = [
        alpha < 1.33,
        (alpha >= 1.33) & (alpha <= 12.5),
        alpha > 12.5,
    ]
    choices = [
        1.0,
        10 ** (0.0054 * (alpha - 1.33) ** 2),
        4.72,
    ]
    tmp = np.select(conditions, choices)
    return st1(m) * tmp


def st1_bar(m: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (st1(m) + st2(m, alpha)) / 2

def tbl_te_a_min(a: np.ndarray) -> np.ndarray:
    conds = [
        a < 0.204,
        (a >= 0.204) & (a <= 0.244),
        a > 0.244,
    ]
    vals = [
        np.sqrt(np.maximum(0, 67.552 - 886.788 * a**2)) - 8.219,
        -32.665 * a + 3.981,
        -142.795 * a**3 + 103.656 * a**2 - 57.757 * a + 6.006,
    ]
    return np.select(conds, vals)


def tbl_te_a_max(a: np.ndarray) -> np.ndarray:
    conds = [
        a < 0.13,
        (a >= 0.13) & (a <= 0.321),
        a > 0.321,
    ]
    vals = [
        np.sqrt(67.552 - 886.788 * a**2) - 8.219,
        -15.901 * a + 1.098,
        -4.669 * a**3 + 3.491 * a**2 - 16.699 * a + 1.149,
    ]
    return np.select(conds, vals)


def tbl_te_a0(re_c: np.ndarray) -> np.ndarray:
    conds = [
        re_c < 9.52e4,
        (re_c >= 9.52e4) & (re_c <= 8.57e5),
        re_c > 8.57e5,
    ]
    vals = [
        0.57,
        (-9.57e-13) * (re_c - 8.57e5) ** 2 + 1.13,
        1.13,
    ]
    return np.select(conds, vals)


def tbl_te_ar(a0: np.ndarray) -> np.ndarray:
    return (-20 - tbl_te_a_min(a0)) / (tbl_te_a_max(a0) - tbl_te_a_min(a0))


def tbl_te_a(a: np.ndarray, re_c: np.ndarray) -> np.ndarray:
    a0 = tbl_te_a0(re_c)
    return tbl_te_a_min(a) + tbl_te_ar(a0) * (tbl_te_a_max(a) - tbl_te_a_min(a))

def tbl_te_b_min(b: np.ndarray) -> np.ndarray:
    conds = [
        b < 0.13,
        (b >= 0.13) & (b <= 0.145),
        b > 0.145,
    ]
    vals = [
        np.sqrt(16.888 - 886.788 * b**2) - 4.109,
        -83.607 * b + 8.138,
        -817.810 * b**3 + 355.210 * b**2 - 135.024 * b + 10.619,
    ]
    return np.select(conds, vals)


def tbl_te_b_max(b: np.ndarray) -> np.ndarray:
    conds = [
        b < 0.10,
        (b >= 0.10) & (b <= 0.187),
        b > 0.187,
    ]
    vals = [
        np.sqrt(16.888 - 886.788 * b**2) - 4.109,
        -31.330 * b + 1.854,
        -80.541 * b**3 + 44.174 * b**2 - 39.381 * b + 2.344,
    ]
    return np.select(conds, vals)


def tbl_te_b0(re_c: np.ndarray) -> np.ndarray:
    conds = [
        re_c < 9.52e4,
        (re_c >= 9.52e4) & (re_c <= 8.57e5),
        re_c > 8.57e5,
    ]
    vals = [
        0.30,
        (-4.48e-13) * (re_c - 8.57e5) ** 2 + 0.56,
        0.56,
    ]
    return np.select(conds, vals)


def tbl_te_br(b0: np.ndarray) -> np.ndarray:
    return (-20 - tbl_te_b_min(b0)) / (tbl_te_b_max(b0) - tbl_te_b_min(b0))


def tbl_te_b(b: np.ndarray, re_c: np.ndarray) -> np.ndarray:
    b0 = tbl_te_b0(re_c)
    return tbl_te_b_min(b) + tbl_te_br(b0) * (tbl_te_b_max(b) - tbl_te_b_min(b))

def k1(re_c: np.ndarray) -> np.ndarray:
    conds = [
        re_c < 2.47e5,
        (re_c >= 2.47e5) & (re_c <= 8e5),
        re_c > 8e5,
    ]
    vals = [
        -4.31 * np.log10(re_c) + 156.3,
        -9.0 * np.log10(re_c) + 181.6,
        128.5,
    ]
    return np.select(conds, vals)


def delta_k1(re_dp: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.where(
        re_dp <= 5000, alpha * (1.43 * np.log10(re_dp) - 5.29), 0.0
    )


def k2(re_c: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    gamma = 27.094 * m + 3.31
    gamma0 = 23.43 * m + 4.651
    beta = 72.65 * m + 10.74
    beta0 = -34.19 * m - 13.82

    conds = [
        alpha < gamma0 - gamma,
        (alpha >= gamma0 - gamma) & (alpha <= gamma0 + gamma),
        alpha > gamma0 + gamma,
    ]

    vals = [
        -1000.0,
        np.sqrt(beta**2 - (beta / gamma) ** 2 * (alpha - gamma0) ** 2) + beta0,
        -12.0,
    ]

    tmp = np.select(conds, vals)
    tmp = np.where((alpha > gamma0) & (tmp < -12), -12, tmp)
    return k1(re_c) + tmp

def calculate_directivity(
    r_element: np.ndarray,      # (azim, blade, radius, 3)
    r_obs: np.ndarray,          # (obs, 3)
    m: np.ndarray,              # (radius,)
    azimuths: np.ndarray,       # (azim,)
    blade_angles: np.ndarray,   # (blade,)
    directivity_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    
    # 1. Setup vectors from elements to observers
    # r_vec shape: (obs, azim, blade, radius, 3)
    r_vec = r_obs[:, None, None, None, :] - r_element[None, ...]
    r_mag = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    unit_r = r_vec / r_mag  # Directional unit vector to observer
    
    # 2. Define the Local Basis (Rotation Matrix)
    # psi_total is the instantaneous angle of the blade element in the YZ plane
    psi_total = azimuths[:, None, None] + blade_angles[None, :, None]
    
    # xl: Thrust Vector (Vertical/Axial)
    e_xl = np.array([1, 0, 0]) 
    
    # yl: Spanwise Vector (Radial)
    e_yl = np.stack([
        np.zeros_like(psi_total), 
        np.cos(psi_total), 
        np.sin(psi_total)
    ], axis=-1)
    
    # zl: Direction of Motion (Tangential) 
    # This is xl cross yl. For xl=[1,0,0], this yields [0, -sin(psi), cos(psi)]
    e_zl = np.cross(e_xl, e_yl)

    # 3. Project Observer Unit Vector onto Local Axes
    # Use dot products to find the observer's components in the blade's frame
    obs_xl = np.sum(unit_r * e_xl, axis=-1)
    obs_yl = np.sum(unit_r * e_yl, axis=-1)
    obs_zl = np.sum(unit_r * e_zl, axis=-1)

    # 4. Convert to BPM Angles (theta and phi)
    # In BPM, phi is the angle in the spanwise-normal plane
    # theta is the angle from the chord (xl)
    phi = np.arctan2(obs_zl, obs_yl)
    theta = np.arccos(np.clip(obs_xl, -1.0, 1.0))
    
    cos_theta = obs_xl
    sin_phi = np.sin(phi)
    sin_theta_2 = np.sin(theta / 2)
    cos_theta_2 = np.cos(theta / 2)

    # 5. Calculate Directivity Gain
    m_exp = m[None, None, None, :]
    
    if directivity_type == "TE":
        # 2 sin^2(theta/2) sin^2(phi) / (1+M cos theta)...
        term1 = 2 * (sin_theta_2**2) * (sin_phi**2)
        term2 = (1 + m_exp * cos_theta) * (1 + 0.2 * m_exp * cos_theta)**2
        dh = term1 / term2
    elif directivity_type == "LE":
        term1 = 2 * (cos_theta_2**2) * (sin_phi**2)
        term2 = (1 + m_exp * cos_theta)**3
        dh = term1 / term2
    elif directivity_type == "low":
        term1 = (np.sin(theta)**2) * (sin_phi**2)
        term2 = (1 + m_exp * cos_theta)**4
        dh = term1 / term2

    return np.squeeze(r_mag), dh

def st1_prime(re_c: np.ndarray) -> np.ndarray:
    conds = [
        re_c <= 1.3e5,
        (re_c > 1.3e5) & (re_c <= 4e5),
        re_c > 4e5,
    ]
    vals = [
        0.18,
        0.001756 * re_c ** 0.3931,
        0.28,
    ]
    return np.select(conds, vals)


def st_peak_prime(re_c: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return st1_prime(re_c) * 10 ** (-0.04 * alpha)


def g1(e: np.ndarray) -> np.ndarray:
    log_e = np.log10(e)
    conds = [
        e <= 0.5974,
        (e > 0.5974) & (e <= 0.8545),
        (e > 0.8545) & (e <= 1.17),
        (e > 1.17) & (e <= 1.674),
        e > 1.674,
    ]
    vals = [
        39.8 * log_e - 11.12,
        98.409 * log_e + 2.0,
        -5.076 + np.sqrt(np.maximum(0, 2.484 - 506.25 * log_e**2)),
        -98.409 * log_e + 2.0,
        -39.8 * log_e - 11.12,
    ]
    return np.select(conds, vals)


def g2(d: np.ndarray) -> np.ndarray:
    log_d = np.log10(d)
    conds = [
        d <= 0.3237,
        (d > 0.3237) & (d <= 0.5689),
        (d > 0.5689) & (d <= 1.7579),
        (d > 1.7579) & (d <= 3.0889),
        d > 3.0889,
    ]
    vals = [
        77.852 * log_d + 15.328,
        65.188 * log_d + 9.125,
        -114.052 * log_d**2,
        -65.188 * log_d + 9.125,
        -77.852 * log_d + 15.328,
    ]
    return np.select(conds, vals)


def re_c0(alpha: np.ndarray) -> np.ndarray:
    conds = [
        alpha <= 3.0,
        alpha > 3.0,
    ]
    vals = [
        10 ** (0.215 * alpha + 4.978),
        10 ** (0.120 * alpha + 5.263),
    ]
    return np.select(conds, vals)


def g3(alpha: np.ndarray) -> np.ndarray:
    return 171.04 - 3.03 * alpha

def calc_delta_avg(delta_p: np.ndarray, delta_s: np.ndarray) -> np.ndarray:
    return (delta_p + delta_s) / 2


def st_peak_3prime(q: np.ndarray, psi: np.ndarray) -> np.ndarray:
    conds = [
        q < 0.2,
        q >= 0.2,
    ]
    vals = [
        0.1 * q + 0.095 - 0.00243 * psi,
        (0.212 - 0.0045 * psi) / (1 + 0.235 / q - 0.0132 / q**2),
    ]
    return np.select(conds, vals)


def g4(q: np.ndarray, psi: np.ndarray) -> np.ndarray:
    conds = [
        q <= 5,
        q > 5,
    ]
    vals = [
        17.5 * np.log10(q) + 157.5 - 1.114 * psi,
        169.7 - 1.114 * psi,
    ]
    return np.select(conds, vals)


def calc_mu(q: np.ndarray) -> np.ndarray:
    conds = [
        q < 0.25,
        (q >= 0.25) & (q < 0.62),
        (q >= 0.62) & (q < 1.15),
        q >= 1.15,
    ]
    vals = [
        0.1221,
        -0.2175 * q + 0.1755,
        -0.0308 * q + 0.0596,
        0.0242,
    ]
    return np.select(conds, vals)


def calc_m(q: np.ndarray) -> np.ndarray:
    conds = [
        q <= 0.02,
        (q > 0.02) & (q <= 0.5),
        (q > 0.5) & (q <= 0.62),
        (q > 0.62) & (q <= 1.15),
        (q > 1.15) & (q <= 1.2),
        q > 1.2,
    ]
    vals = [
        0,
        68.724 * q - 1.35,
        308.475 * q - 121.23,
        224.811 * q - 69.35,
        1583.28 * q - 1631.59,
        268.344,
    ]
    return np.select(conds, vals)


def calc_eta0(m: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return -np.sqrt(m**2 * mu**4 / (6.25 + m**2 * mu**2))


def calc_k(m: np.ndarray, mu: np.ndarray, eta0: np.ndarray) -> np.ndarray:
    return 2.5 * np.sqrt(1 - (eta0 / mu) ** 2) - 2.5 - m * eta0


def g5(
    m: np.ndarray,
    mu: np.ndarray,
    eta: np.ndarray,
    eta0: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    conds = [
        eta < eta0,
        (eta >= eta0) & (eta < 0),
        (eta >= 0) & (eta < 0.03616),
        eta >= 0.03616,
    ]
    vals = [
        m * eta + k,
        2.5 * np.sqrt(1 - (eta / mu) ** 2) - 2.5,
        np.sqrt(1.5625 - 1194.99 * eta**2) - 1.25,
        -155.543 * eta + 4.375,
    ]
    return np.select(conds, vals)


def g5_0(q: np.ndarray, eta: np.ndarray) -> np.ndarray:
    q_0 = 6.724 * q**2 - 4.019 * q + 1.107
    m = calc_m(q_0)
    mu = calc_mu(q_0)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    return g5(m, mu, eta, eta0, k)


def g5_tot(q: np.ndarray, eta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    g5_0_val = g5_0(q, eta)
    m = calc_m(q)
    mu = calc_mu(q)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    g5_val = g5(m, mu, eta, eta0, k)
    return g5_0_val + 0.0714 * psi * np.where(g5_val - g5_0_val < 0, g5_val - g5_0_val, 0)

def calc_l_tip(chord: np.ndarray, alpha_tip: np.ndarray) -> np.ndarray:
    return chord * 0.008 * alpha_tip


def phi_ww(
    f: np.ndarray, u: np.ndarray, sigma: np.ndarray, l: np.ndarray
) -> np.ndarray:
    k1 = 2 * np.pi * f / u
    num = 2 * sigma**2 * l / np.pi
    denom = (1 + (1.339 * l * k1) ** 2) ** (5 / 6)
    return num / denom


def amiet_l(
    omega: np.ndarray,
    u: np.ndarray,
    l: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    a_inf: float,
    xi: float,
) -> np.ndarray:
    return (
        2
        * l
        * (xi * omega / u)
        / ((xi * omega / u) ** 2 + (omega * y / a_inf / sigma) ** 2)
    )

class BPM:
    """BPM broadband noise prediction model.
    
    Implements the BPM model for computing broadband noise sources from propeller
    blades including turbulent boundary layer (TBL), laminar boundary layer (LBL),
    trailing edge (TE), tip vortex (TV), and blade-wake interaction (BWI) noise.
    """

    def __init__(
        self,
        propeller: Propeller,
        frequencies: np.ndarray,
        num_azim: int = 360,
    ) -> None:
        """Initialize BPM model with propeller geometry and acoustic parameters.
        
        Args:
            propeller: Propeller object with solved aerodynamics and geometry.
            frequencies: Frequency array for acoustic analysis (Hz).
            num_azim: Number of azimuthal positions for blade discretization.
        """
        self.frequencies = frequencies
        self.propeller = propeller
        self.aero_params = propeller.params
        self.acoustic_params = propeller.params
        self.v_inf = propeller.v_inf
        self.r = propeller.solution_data['r'].values
        self.dr = propeller.solution_data['dr'].values
        self.chord = propeller.solution_data['chord'].values
        self.alpha = propeller.solution_data['alpha'].values
        self.vi = propeller.solution_data['u'].values
        self.u = propeller.solution_data['W'].values
        self.re_c = propeller.solution_data['Re'].values
        self.m = propeller.solution_data['Ma'].values
        self.delta_p = propeller.solution_data['dp'].values
        self.delta_s = propeller.solution_data['ds'].values
        self.psi = propeller.geometry['boat_tail_angle']
        self.azimuth_positions = np.linspace(0, 2 * np.pi, num_azim, endpoint=False)

        # Compute element positions in 3D space
        y_el = (
            self.r[None, None, :]
            * np.cos(
                self.azimuth_positions[:, None, None]
                + propeller.params.blade_angles[None, :, None]
            )
        )
        z_el = (
            self.r[None, None, :]
            * np.sin(
                self.azimuth_positions[:, None, None]
                + propeller.params.blade_angles[None, :, None]
            )
        )

        self.r_element = np.zeros((*y_el.shape, 3))
        self.r_element[..., 1] = y_el
        self.r_element[..., 2] = z_el

        # Compute directivity patterns
        self.r_dist, self.dh_te = calculate_directivity(
            self.r_element, propeller.observer_positions, self.m, self.azimuth_positions, propeller.params.blade_angles, directivity_type="TE"
        )
        _, self.dh_le = calculate_directivity(
            self.r_element, propeller.observer_positions, self.m, self.azimuth_positions, propeller.params.blade_angles, directivity_type="LE"
        )
        _, self.dl = calculate_directivity(
            self.r_element, propeller.observer_positions, self.m, self.azimuth_positions, propeller.params.blade_angles, directivity_type="low"
        )

        # Compute base directivity values
        self.base_val_te = (self.m**5 * self.dr * self.dh_te) / (self.r_dist**2)
        self.base_val_le = (self.m**5 * self.dr * self.dh_le) / (self.r_dist**2)
        self.base_val_low = (self.m**5 * self.dr * self.dl) / (self.r_dist**2)

    def run_bpm(
        self,
        lt: float = 1e6,
        i: float = 0.005,
        alpha_stall: float = 15,
        h: float | None = None,
    ) -> None:
        """Run all BPM noise mechanisms.
        
        Args:
            z: Roughness height or reference height for turbulence (m).
            z0: Surface roughness length (m).
            i: Turbulence intensity (fraction of flow velocity).
            alpha_stall: Stall angle of attack (degrees).
            h: Trailing edge thickness (m). If None, computed as 1% of chord.
        """
        self.ti_noise(lt, i)
        self.tbl_noise(alpha_stall)
        self.lbl_noise()
        self.teb_noise(h)
        self.tv_noise()

    def ti_noise(self, lt: float, i: float) -> None:
        """Compute turbulence ingestion (TI) noise.
        
        Args:
            lt: Turbulence length scale (m).
            i: Turbulence intensity.
        """
        f_co = 10 * self.u / np.pi / self.chord
        freq_5d = self.frequencies[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        f_co_4d = f_co[np.newaxis, np.newaxis, np.newaxis, :]
        dh_avg = (self.dh_te + self.dh_le) / 2
        d = np.where(freq_5d < f_co_4d, self.dl, dh_avg)
        k1_val = 2 * np.pi * self.frequencies[:, np.newaxis] / self.u[np.newaxis, :]
        k1_bar = k1_val * self.chord / 2
        ke = 3 / 4 / lt
        k1_hat = k1_val / ke
        k1_hat = k1_hat[:, np.newaxis, np.newaxis, np.newaxis, :]
        base_val_ti = (self.m**5 * self.dr * d) / (self.r_dist[np.newaxis, :, :, :, :]**2)
        beta = 1 - self.m**2
        s_sq = 1 / (
            2 * np.pi * k1_bar / beta**2 + 1 / (1 + 2.4 * k1_bar / beta**2)
        )
        lfc = 10 * s_sq * self.m * k1_bar**(-2) * beta**(-2)
        lfc = lfc[:, np.newaxis, np.newaxis, np.newaxis, :]
        spl_ti = (
            10
            * np.log10(
                (
                    self.aero_params.rho**2
                    * self.aero_params.a_inf**4
                    * lt
                    / 2
                    * i**2
                    * k1_hat**3
                    / (1 + k1_hat**2) ** (7 / 3)
                    * base_val_ti
                )
                + 1e-12
            )
            + 78.4
            + 10 * np.log10(1 + 9 * self.alpha[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :] ** 2)
            + 10 * np.log10(lfc / (1 + lfc))
        )
        spp_ti = np.zeros_like(spl_ti)
        spp_ti += 10 ** (spl_ti / 10)
        self.spl_ti = 10 * np.log10(spp_ti.sum(axis=(-1, -2)).mean(axis=-1))

    def teb_noise(self, h: float | None = None) -> None:
        """Compute trailing edge bluntness (TEB) noise.
        
        Args:
            h: Trailing edge thickness (m). If None, uses 1% of chord.
        """
        if h is None:
            h = 1e-2 * self.chord
        delta_avg = calc_delta_avg(self.delta_p, self.delta_s)
        q = h / delta_avg
        st_3prime = st(self.frequencies, h, self.u)
        st_3prime_peak = st_peak_3prime(q, self.psi)
        eta = np.log10(st_3prime / st_3prime_peak)

        spl_teb = (
            10 * np.log10((h * self.base_val_te[np.newaxis, :] * self.m**0.5) + 1e-12)
            + g4(q, self.psi)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
            + g5_tot(q, eta, self.psi)[:, np.newaxis, np.newaxis, np.newaxis, :]
        )
        spp_teb = np.zeros_like(spl_teb)
        spp_teb += 10 ** (spl_teb / 10)
        self.spl_teb = 10 * np.log10(spp_teb.sum(axis=(2, 3)).mean(axis=-1))
    
    def lbl_noise(self) -> None:
        """Compute laminar boundary layer (LBL) noise."""
        e = st(self.frequencies, self.delta_p, self.u) / st_peak_prime(self.re_c, self.alpha)
        d = self.re_c / re_c0(self.alpha)

        spl_lbl = (
            10 * np.log10((self.delta_p * self.base_val_le[np.newaxis, :]) + 1e-12)
            + g1(e)[:, np.newaxis, np.newaxis, np.newaxis, :]
            + g2(d)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
            + g3(self.alpha)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
        )
        spp_lbl = np.zeros_like(spl_lbl)
        spp_lbl += 10 ** (spl_lbl / 10)
        self.spl_lbl = 10 * np.log10(spp_lbl.sum(axis=(-1, -2)).mean(axis=-1))
    
    def tbl_noise(self, alpha_stall: float) -> None:
        """Compute turbulent boundary layer (TBL) noise.
        
        Args:
            alpha_stall: Stall angle of attack (degrees).
        """
        st_1 = st1(self.m)
        st_2 = st2(self.m, self.alpha)
        k_1 = k1(self.re_c)
        k_2 = k2(self.re_c, self.m, self.alpha)
        delta_k1_val = delta_k1((self.re_c / self.chord * self.delta_p), self.alpha)
        st_p = st(self.frequencies, self.delta_p, self.u)
        st_s = st(self.frequencies, self.delta_s, self.u)

        as_val = tbl_te_a(np.abs(np.log10(st_s / st_1)), self.re_c)
        ap = tbl_te_a(np.abs(np.log10(st_p / st_1)), self.re_c)
        b = tbl_te_b(np.abs(np.log10(st_s / st_2)), self.re_c)
        a_prime = tbl_te_a(np.abs(np.log10(st_s / st_2)), 3 * self.re_c)

        as_exp = as_val[:, np.newaxis, np.newaxis, np.newaxis, :]
        ap_exp = ap[:, np.newaxis, np.newaxis, np.newaxis, :]
        b_exp = b[:, np.newaxis, np.newaxis, np.newaxis, :]
        a_prime_exp = a_prime[:, np.newaxis, np.newaxis, np.newaxis, :]
        spl_s = np.where(
            self.alpha < alpha_stall,
            10 * np.log10((self.delta_s * self.base_val_te[np.newaxis, :]) + 1e-12) + as_exp + k_1 - 3,
            -np.inf,
        )
        spl_p = np.where(
            self.alpha < alpha_stall,
            (
                10 * np.log10((self.delta_p * self.base_val_te[np.newaxis, :]) + 1e-12)
                + ap_exp
                + k_1
                - 3
                + delta_k1_val
            ),
            -np.inf,
        )
        spl_a = np.where(
            self.alpha < alpha_stall,
            10 * np.log10((self.delta_s * self.base_val_te[np.newaxis, :]) + 1e-12) + b_exp + k_2,
            10 * np.log10((self.delta_s * self.base_val_low[np.newaxis, :]) + 1e-12) + a_prime_exp + k_2,
        )
        spp_tbl = np.zeros_like(spl_a)
        spp_tbl += 10 ** (spl_s / 10) + 10 ** (spl_p / 10) + 10 ** (spl_a / 10)
        self.spl_tbl = 10 * np.log10(spp_tbl.sum(axis=(-1, -2)).mean(axis=-1))

    def tv_noise(self) -> None:
        """Compute tip vortex (TV) noise."""
        l_tip = calc_l_tip(self.chord[-1], self.alpha[-1])
        m_max = self.m[-1] * (1 + 0.036 * self.alpha[-1])
        st_2prime = st(
            self.frequencies,
            l_tip,
            np.atleast_1d(self.acoustic_params.a_inf * m_max),
        )

        spl_tip = (
            10
            * np.log10(
                self.m[-1] ** 2
                * m_max**3
                * l_tip**2
                * self.dh_te[..., -1]
                / self.r_dist[..., -1] ** 2
            )
            - 30.5 * (np.log10(st_2prime[:, :, np.newaxis, np.newaxis]) + 0.3) ** 2
            + 126
        )
        spp_tip = np.zeros_like(spl_tip)
        spp_tip += 10 ** (spl_tip / 10)
        self.spl_tip = 10 * np.log10(spp_tip.sum(axis=-1).mean(axis=-1))

    def bwi_noise(
        self,
        xi: float = 1,
        b: float = 1,
        sigma_turb: float = 0.1,
        l_scale: float = 0.125,
    ) -> None:
        """Compute blade-wake interaction (BWI) noise.
        
        Args:
            xi: Coherence decay parameter.
            b: Blade planform reference area (m^2).
            sigma_turb: Turbulence intensity as fraction of local velocity.
            l_scale: Integral length scale as fraction of chord.
        """
        v_t = np.sqrt(self.u**2 - (self.v_inf + self.vi) ** 2)
        y_vt = v_t[None, None, :] * np.sin(
            self.azimuth_positions[:, None, None]
            + self.propeller.params.blade_angles[None, :, None]
        )
        z_vt = (
            -v_t[None, None, :]
            * np.cos(
                self.azimuth_positions[:, None, None]
                + self.propeller.params.blade_angles[None, :, None]
            )
        )
        v_t_vec = np.stack([np.zeros_like(y_vt), y_vt, z_vt], axis=-1)

        r_vec = (
            self.propeller.observer_positions[:, np.newaxis, np.newaxis, np.newaxis, :]
            - self.r_element[np.newaxis, ...]
        )
        r_mag = np.linalg.norm(r_vec, axis=-1)
        sigma = r_mag**2 * (1 - self.m * np.sum(v_t_vec * r_vec, axis=-1))

        f = self.frequencies[:, np.newaxis, np.newaxis, np.newaxis]
        u = self.u[np.newaxis, np.newaxis, np.newaxis, :]

        omega = 2 * np.pi * f
        phi = phi_ww(f, u, sigma_turb, l_scale)
        l_sq = amiet_l(
            omega, u, self.dr, r_vec[..., 1], sigma, self.aero_params.a_inf, xi
        )
        spp_bwi = (
            (omega * r_vec[..., 0] / 4 / np.pi / self.aero_params.a_inf / sigma**2) ** 2
            * b
            * self.chord
            * phi
            * l_sq
        )
        self.spl_bwi = 10 * np.log10(spp_bwi.sum(axis=(2, 3)).mean(axis=-1))