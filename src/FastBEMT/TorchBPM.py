import torch
import numpy as np
from typing import Union, Tuple, Optional
from .Propeller import Propeller


def _torch_select(
    conditions: list, choices: list, default_value: float = 0.0
) -> torch.Tensor:
    """Convert np.select-like logic to torch.where() without type conversion overhead.
    
    Assumes all condition and choice tensors are already on the correct device.
    
    Args:
        conditions: List of boolean tensors
        choices: List of tensors/scalars corresponding to each condition
        default_value: Default value when no condition is met
        
    Returns:
        Tensor with selected values based on conditions
    """
    result = torch.full_like(conditions[0], default_value, dtype=conditions[0].dtype)
    for cond, choice in zip(reversed(conditions), reversed(choices)):
        if not isinstance(choice, torch.Tensor):
            choice = torch.tensor(choice, dtype=result.dtype, device=result.device)
        result = torch.where(cond, choice, result)
    return result


def st(f: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute Strouhal number: f * l / u (expects torch tensors).
    
    Args:
        f: Frequency tensor
        l: Length scale tensor
        u: Velocity tensor
        
    Returns:
        Strouhal number tensor
    """
    return f[:, None] * l / u[None, :]


def st1(m: torch.Tensor) -> torch.Tensor:
    """Compute st1 parameter: 0.02 * m^(-0.6) (expects torch tensor).
    
    Args:
        m: Mach number tensor
        
    Returns:
        st1 parameter tensor
    """
    return 0.02 * torch.pow(m, -0.6)


def st2(m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute st2 parameter based on Mach and angle of attack (expects torch tensors).
    
    Args:
        m: Mach number tensor
        alpha: Angle of attack tensor (degrees)
        
    Returns:
        st2 parameter tensor
    """
    conditions = [
        alpha < 1.33,
        (alpha >= 1.33) & (alpha <= 12.5),
        alpha > 12.5,
    ]
    choices = [
        torch.ones_like(alpha),
        torch.pow(10.0, 0.0054 * torch.pow(alpha - 1.33, 2)),
        torch.full_like(alpha, 4.72),
    ]
    tmp = _torch_select(conditions, choices)
    return st1(m) * tmp


def st1_bar(m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute average st1 parameter (expects torch tensors)."""
    return (st1(m) + st2(m, alpha)) / 2

def tbl_te_a_min(a: torch.Tensor) -> torch.Tensor:
    """Compute minimum TBL trailing edge correction (a component)."""
    conditions = [
        a < 0.204,
        (a >= 0.204) & (a <= 0.244),
        a > 0.244,
    ]
    choices = [
        torch.sqrt(torch.clamp(67.552 - 886.788 * a**2, min=0)) - 8.219,
        -32.665 * a + 3.981,
        -142.795 * a**3 + 103.656 * a**2 - 57.757 * a + 6.006,
    ]
    return _torch_select(conditions, choices)


def tbl_te_a_max(a: torch.Tensor) -> torch.Tensor:
    """Compute maximum TBL trailing edge correction (a component)."""
    conditions = [
        a < 0.13,
        (a >= 0.13) & (a <= 0.321),
        a > 0.321,
    ]
    choices = [
        torch.sqrt(67.552 - 886.788 * a**2) - 8.219,
        -15.901 * a + 1.098,
        -4.669 * a**3 + 3.491 * a**2 - 16.699 * a + 1.149,
    ]
    return _torch_select(conditions, choices)


def tbl_te_a0(re_c: torch.Tensor) -> torch.Tensor:
    """Compute reference TBL parameter a0."""
    conditions = [
        re_c < 9.52e4,
        (re_c >= 9.52e4) & (re_c <= 8.57e5),
        re_c > 8.57e5,
    ]
    choices = [
        torch.full_like(re_c, 0.57),
        (-9.57e-13) * (re_c - 8.57e5) ** 2 + 1.13,
        torch.full_like(re_c, 1.13),
    ]
    return _torch_select(conditions, choices)


def tbl_te_ar(a0: torch.Tensor) -> torch.Tensor:
    """Compute TBL adjustment ratio (a)."""
    return (-20 - tbl_te_a_min(a0)) / (tbl_te_a_max(a0) - tbl_te_a_min(a0))


def tbl_te_a(a: torch.Tensor, re_c: torch.Tensor) -> torch.Tensor:
    """Compute TBL trailing edge correction (a component)."""
    a0 = tbl_te_a0(re_c)
    return tbl_te_a_min(a) + tbl_te_ar(a0) * (tbl_te_a_max(a) - tbl_te_a_min(a))

def tbl_te_b_min(b: torch.Tensor) -> torch.Tensor:
    """Compute minimum TBL trailing edge correction (b component)."""
    conditions = [
        b < 0.13,
        (b >= 0.13) & (b <= 0.145),
        b > 0.145,
    ]
    choices = [
        torch.sqrt(16.888 - 886.788 * b**2) - 4.109,
        -83.607 * b + 8.138,
        -817.810 * b**3 + 355.210 * b**2 - 135.024 * b + 10.619,
    ]
    return _torch_select(conditions, choices)


def tbl_te_b_max(b: torch.Tensor) -> torch.Tensor:
    """Compute maximum TBL trailing edge correction (b component)."""
    conditions = [
        b < 0.10,
        (b >= 0.10) & (b <= 0.187),
        b > 0.187,
    ]
    choices = [
        torch.sqrt(16.888 - 886.788 * b**2) - 4.109,
        -31.330 * b + 1.854,
        -80.541 * b**3 + 44.174 * b**2 - 39.381 * b + 2.344,
    ]
    return _torch_select(conditions, choices)


def tbl_te_b0(re_c: torch.Tensor) -> torch.Tensor:
    """Compute reference TBL parameter b0."""
    conditions = [
        re_c < 9.52e4,
        (re_c >= 9.52e4) & (re_c <= 8.57e5),
        re_c > 8.57e5,
    ]
    choices = [
        torch.full_like(re_c, 0.30),
        (-4.48e-13) * (re_c - 8.57e5) ** 2 + 0.56,
        torch.full_like(re_c, 0.56),
    ]
    return _torch_select(conditions, choices)


def tbl_te_br(b0: torch.Tensor) -> torch.Tensor:
    """Compute TBL adjustment ratio (b)."""
    return (-20 - tbl_te_b_min(b0)) / (tbl_te_b_max(b0) - tbl_te_b_min(b0))


def tbl_te_b(b: torch.Tensor, re_c: torch.Tensor) -> torch.Tensor:
    """Compute TBL trailing edge correction (b component)."""
    b0 = tbl_te_b0(re_c)
    return tbl_te_b_min(b) + tbl_te_br(b0) * (tbl_te_b_max(b) - tbl_te_b_min(b))

def k1(re_c: torch.Tensor) -> torch.Tensor:
    """Compute k1 parameter based on Reynolds number."""
    conditions = [
        re_c < 2.47e5,
        (re_c >= 2.47e5) & (re_c <= 8e5),
        re_c > 8e5,
    ]
    choices = [
        -4.31 * torch.log10(re_c) + 156.3,
        -9.0 * torch.log10(re_c) + 181.6,
        torch.full_like(re_c, 128.5),
    ]
    return _torch_select(conditions, choices)


def delta_k1(re_dp: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute delta_k1 correction."""
    return torch.where(
        re_dp <= 5000,
        alpha * (1.43 * torch.log10(re_dp) - 5.29),
        torch.zeros_like(alpha)
    )


def k2(re_c: torch.Tensor, m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute k2 parameter."""
    gamma = 27.094 * m + 3.31
    gamma0 = 23.43 * m + 4.651
    beta = 72.65 * m + 10.74
    beta0 = -34.19 * m - 13.82

    conditions = [
        alpha < gamma0 - gamma,
        (alpha >= gamma0 - gamma) & (alpha <= gamma0 + gamma),
        alpha > gamma0 + gamma,
    ]

    choices = [
        torch.full_like(alpha, -1000.0),
        torch.sqrt(beta**2 - (beta / gamma) ** 2 * (alpha - gamma0) ** 2) + beta0,
        torch.full_like(alpha, -12.0),
    ]

    tmp = _torch_select(conditions, choices, default_value=-1000.0)
    tmp = torch.where((alpha > gamma0) & (tmp < -12), torch.full_like(tmp, -12.0), tmp)
    return k1(re_c) + tmp

def calculate_directivity(
    r_element: torch.Tensor,
    r_obs: torch.Tensor,
    m: torch.Tensor,
    azimuths: torch.Tensor,
    blade_angles: torch.Tensor,
    directivity_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate acoustic directivity patterns.
    
    Args:
        r_element: Element positions (azim, blade, radius, 3) as torch tensor
        r_obs: Observer positions (obs, 3) as torch tensor
        m: Mach number (radius,) as torch tensor
        azimuths: Azimuthal positions (azim,) as torch tensor
        blade_angles: Blade angles (blade,) as torch tensor
        directivity_type: Type of directivity ("TE", "LE", or "low")
        
    Returns:
        Tuple of (distance, directivity) tensors
    """
    # All inputs are already torch tensors on correct device
    # 1. Setup vectors from elements to observers
    # r_vec shape: (obs, azim, blade, radius, 3)
    r_vec = r_obs[:, None, None, None, :] - r_element[None, ...]
    r_mag = torch.linalg.norm(r_vec, dim=-1, keepdims=True)
    unit_r = r_vec / r_mag

    # 2. Define the Local Basis (Rotation Matrix)
    psi_total = azimuths[:, None, None] + blade_angles[None, :, None]
    device = r_element.device
    
    # xl: Thrust Vector (Axial)
    e_xl = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    
    # yl: Spanwise Vector (Radial)
    e_yl = torch.stack([
        torch.zeros_like(psi_total),
        torch.cos(psi_total),
        torch.sin(psi_total)
    ], dim=-1)
    
    # zl: Direction of Motion (Tangential)
    e_zl = torch.cross(e_xl.expand_as(e_yl), e_yl, dim=-1)

    # 3. Project Observer Unit Vector onto Local Axes
    obs_xl = torch.sum(unit_r * e_xl, dim=-1)
    obs_yl = torch.sum(unit_r * e_yl, dim=-1)
    obs_zl = torch.sum(unit_r * e_zl, dim=-1)

    # 4. Convert to BPM Angles
    phi = torch.atan2(obs_zl, obs_yl)
    theta = torch.acos(torch.clamp(obs_xl, -1.0, 1.0))
    
    cos_theta = obs_xl
    sin_phi = torch.sin(phi)
    sin_theta_2 = torch.sin(theta / 2)
    cos_theta_2 = torch.cos(theta / 2)

    # 5. Expand Mach number tensor
    m = m[None, None, None, :]
    
    if directivity_type == "TE":
        term1 = 2 * (sin_theta_2**2) * (sin_phi**2)
        term2 = (1 + m * cos_theta) * (1 + 0.2 * m * cos_theta)**2
        dh = term1 / term2
    elif directivity_type == "LE":
        term1 = 2 * (cos_theta_2**2) * (sin_phi**2)
        term2 = (1 + m * cos_theta)**3
        dh = term1 / term2
    elif directivity_type == "low":
        term1 = (torch.sin(theta)**2) * (sin_phi**2)
        term2 = (1 + m * cos_theta)**4
        dh = term1 / term2

    return torch.squeeze(r_mag), dh

def st1_prime(re_c: torch.Tensor) -> torch.Tensor:
    """Compute st1_prime parameter."""
    conditions = [
        re_c <= 1.3e5,
        (re_c > 1.3e5) & (re_c <= 4e5),
        re_c > 4e5,
    ]
    choices = [
        torch.full_like(re_c, 0.18),
        0.001756 * torch.pow(re_c, 0.3931),
        torch.full_like(re_c, 0.28),
    ]
    return _torch_select(conditions, choices)


def st_peak_prime(re_c: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute peak Strouhal number with angle correction."""
    return st1_prime(re_c) * torch.pow(10.0, -0.04 * alpha)


def g1(e: torch.Tensor) -> torch.Tensor:
    """Compute g1 function for LBL noise."""
    log_e = torch.log10(e)
    conditions = [
        e <= 0.5974,
        (e > 0.5974) & (e <= 0.8545),
        (e > 0.8545) & (e <= 1.17),
        (e > 1.17) & (e <= 1.674),
        e > 1.674,
    ]
    choices = [
        39.8 * log_e - 11.12,
        98.409 * log_e + 2.0,
        -5.076 + torch.sqrt(torch.clamp(2.484 - 506.25 * log_e**2, min=0)),
        -98.409 * log_e + 2.0,
        -39.8 * log_e - 11.12,
    ]
    return _torch_select(conditions, choices)


def g2(d: torch.Tensor) -> torch.Tensor:
    """Compute g2 function for LBL noise."""
    log_d = torch.log10(d)
    conditions = [
        d <= 0.3237,
        (d > 0.3237) & (d <= 0.5689),
        (d > 0.5689) & (d <= 1.7579),
        (d > 1.7579) & (d <= 3.0889),
        d > 3.0889,
    ]
    choices = [
        77.852 * log_d + 15.328,
        65.188 * log_d + 9.125,
        -114.052 * log_d**2,
        -65.188 * log_d + 9.125,
        -77.852 * log_d + 15.328,
    ]
    return _torch_select(conditions, choices)


def re_c0(alpha: torch.Tensor) -> torch.Tensor:
    """Compute reference Reynolds number."""
    conditions = [
        alpha <= 3.0,
        alpha > 3.0,
    ]
    choices = [
        torch.pow(10.0, 0.215 * alpha + 4.978),
        torch.pow(10.0, 0.120 * alpha + 5.263),
    ]
    return _torch_select(conditions, choices)


def g3(alpha: torch.Tensor) -> torch.Tensor:
    """Compute g3 function for angle effect."""
    return 171.04 - 3.03 * alpha

def calc_delta_avg(delta_p: torch.Tensor, delta_s: torch.Tensor) -> torch.Tensor:
    """Compute average boundary layer thickness."""
    return (delta_p + delta_s) / 2


def st_peak_3prime(q: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute peak Strouhal number for TEB noise."""
    conditions = [
        q < 0.2,
        q >= 0.2,
    ]
    choices = [
        0.1 * q + 0.095 - 0.00243 * psi,
        (0.212 - 0.0045 * psi) / (1 + 0.235 / q - 0.0132 / q**2),
    ]
    return _torch_select(conditions, choices)


def g4(q: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute g4 function for TEB noise."""
    conditions = [
        q <= 5,
        q > 5,
    ]
    choices = [
        17.5 * torch.log10(q) + 157.5 - 1.114 * psi,
        torch.full_like(q, 169.7) - 1.114 * psi,
    ]
    return _torch_select(conditions, choices)


def calc_mu(q: torch.Tensor) -> torch.Tensor:
    """Compute mu parameter for TEB noise."""
    conditions = [
        q < 0.25,
        (q >= 0.25) & (q < 0.62),
        (q >= 0.62) & (q < 1.15),
        q >= 1.15,
    ]
    choices = [
        torch.full_like(q, 0.1221),
        -0.2175 * q + 0.1755,
        -0.0308 * q + 0.0596,
        torch.full_like(q, 0.0242),
    ]
    return _torch_select(conditions, choices)


def calc_m(q: torch.Tensor) -> torch.Tensor:
    """Compute m parameter for g5 calculation."""
    conditions = [
        q <= 0.02,
        (q > 0.02) & (q <= 0.5),
        (q > 0.5) & (q <= 0.62),
        (q > 0.62) & (q <= 1.15),
        (q > 1.15) & (q <= 1.2),
        q > 1.2,
    ]
    choices = [
        torch.zeros_like(q),
        68.724 * q - 1.35,
        308.475 * q - 121.23,
        224.811 * q - 69.35,
        1583.28 * q - 1631.59,
        torch.full_like(q, 268.344),
    ]
    return _torch_select(conditions, choices)


def calc_eta0(m: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Compute eta0 parameter."""
    return -torch.sqrt(m**2 * mu**4 / (6.25 + m**2 * mu**2))


def calc_k(m: torch.Tensor, mu: torch.Tensor, eta0: torch.Tensor) -> torch.Tensor:
    """Compute k parameter for g5."""
    return 2.5 * torch.sqrt(1 - (eta0 / mu) ** 2) - 2.5 - m * eta0


def g5(
    m: torch.Tensor,
    mu: torch.Tensor,
    eta: torch.Tensor,
    eta0: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Compute g5 function for TEB noise."""
    conditions = [
        eta < eta0,
        (eta >= eta0) & (eta < 0),
        (eta >= 0) & (eta < 0.03616),
        eta >= 0.03616,
    ]
    choices = [
        m * eta + k,
        2.5 * torch.sqrt(1 - (eta / mu) ** 2) - 2.5,
        torch.sqrt(1.5625 - 1194.99 * eta**2) - 1.25,
        -155.543 * eta + 4.375,
    ]
    return _torch_select(conditions, choices)


def g5_0(q: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Compute g5_0 function."""
    q_0 = 6.724 * q**2 - 4.019 * q + 1.107
    m = calc_m(q_0)
    mu = calc_mu(q_0)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    return g5(m, mu, eta, eta0, k)


def g5_tot(q: torch.Tensor, eta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute total g5 function."""
    g5_0_val = g5_0(q, eta)
    m = calc_m(q)
    mu = calc_mu(q)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    g5_val = g5(m, mu, eta, eta0, k)
    
    delta_g5 = g5_val - g5_0_val
    correction = 0.0714 * psi * torch.where(delta_g5 < 0, delta_g5, torch.zeros_like(delta_g5))
    return g5_0_val + correction

def calc_l_tip(chord: torch.Tensor, alpha_tip: torch.Tensor) -> torch.Tensor:
    """Compute tip length scale."""
    return chord * 0.008 * alpha_tip


def phi_ww(
    f: torch.Tensor,
    u: torch.Tensor,
    sigma: torch.Tensor,
    l: torch.Tensor,
) -> torch.Tensor:
    """Compute Wittenberg-White spectral density function."""
    k1 = 2 * np.pi * f / u
    num = 2 * sigma**2 * l / np.pi
    denom = (1 + (1.339 * l * k1) ** 2) ** (5 / 6)
    return num / denom


def amiet_l(
    omega: torch.Tensor,
    u: torch.Tensor,
    l: torch.Tensor,
    y: torch.Tensor,
    sigma: torch.Tensor,
    a_inf: float,
    xi: float,
) -> torch.Tensor:
    """Compute Amiet coherence length."""
    return (
        2
        * l
        * (xi * omega / u)
        / ((xi * omega / u) ** 2 + (omega * y / a_inf / sigma) ** 2)
    )

class BPM:
    """BPM broadband noise prediction model using PyTorch.
    
    Implements the BPM model for computing broadband noise sources from propeller
    blades including turbulent boundary layer (TBL), laminar boundary layer (LBL),
    trailing edge (TE), tip vortex (TV), and blade-wake interaction (BWI) noise.
    All computations are device-agnostic (CPU/GPU).
    """

    def __init__(
        self,
        propeller: Propeller,
        frequencies: Union[np.ndarray, torch.Tensor],
        num_azim: int = 360,
        device: str = "cpu",
    ) -> None:
        """Initialize BPM model with propeller geometry and acoustic parameters.
        
        Args:
            propeller: Propeller object with solved aerodynamics and geometry.
            frequencies: Frequency array for acoustic analysis (Hz).
            num_azim: Number of azimuthal positions for blade discretization.
            device: Torch device ('cpu' or 'cuda').
        """
        self.device = device
        self.frequencies = torch.as_tensor(frequencies, dtype=torch.float32, device=device)
        self.propeller = propeller
        self.aero_params = propeller.params
        self.acoustic_params = propeller.params
        self.v_inf = propeller.v_inf
        
        # Convert propeller data to torch tensors on device
        self.r = torch.as_tensor(propeller.solution_data['r'].values, dtype=torch.float32, device=device)
        self.dr = torch.as_tensor(propeller.solution_data['dr'].values, dtype=torch.float32, device=device)
        self.chord = torch.as_tensor(propeller.solution_data['chord'].values, dtype=torch.float32, device=device)
        self.alpha = torch.as_tensor(propeller.solution_data['alpha'].values, dtype=torch.float32, device=device)
        self.vi = torch.as_tensor(propeller.solution_data['u'].values, dtype=torch.float32, device=device)
        self.u = torch.as_tensor(propeller.solution_data['W'].values, dtype=torch.float32, device=device)
        self.re_c = torch.as_tensor(propeller.solution_data['Re'].values, dtype=torch.float32, device=device)
        self.m = torch.as_tensor(propeller.solution_data['Ma'].values, dtype=torch.float32, device=device)
        self.delta_p = torch.as_tensor(propeller.solution_data['dp'].values, dtype=torch.float32, device=device)
        self.delta_s = torch.as_tensor(propeller.solution_data['ds'].values, dtype=torch.float32, device=device)
        self.psi = torch.as_tensor(propeller.geometry['boat_tail_angle'], dtype=torch.float32, device=device)
        self.azimuth_positions = torch.linspace(0, 2 * np.pi, num_azim, dtype=torch.float32, device=device)

        # Compute element positions in 3D space
        blade_angles_t = torch.as_tensor(propeller.params.blade_angles, dtype=torch.float32, device=device)
        y_el = (
            self.r[None, None, :]
            * torch.cos(
                self.azimuth_positions[:, None, None]
                + blade_angles_t[None, :, None]
            )
        )
        z_el = (
            self.r[None, None, :]
            * torch.sin(
                self.azimuth_positions[:, None, None]
                + blade_angles_t[None, :, None]
            )
        )

        self.r_element = torch.zeros((*y_el.shape, 3), dtype=torch.float32, device=device)
        self.r_element[..., 1] = y_el
        self.r_element[..., 2] = z_el

        # Compute directivity patterns
        obs_pos = torch.as_tensor(propeller.observer_positions, dtype=torch.float32, device=device)
        self.r_dist, self.dh_te = calculate_directivity(
            self.r_element, obs_pos, self.m, self.azimuth_positions, blade_angles_t, directivity_type="TE"
        )
        _, self.dh_le = calculate_directivity(
            self.r_element, obs_pos, self.m, self.azimuth_positions, blade_angles_t, directivity_type="LE"
        )
        _, self.dl = calculate_directivity(
            self.r_element, obs_pos, self.m, self.azimuth_positions, blade_angles_t, directivity_type="low"
        )

        # Compute base directivity values
        self.base_val_te = (self.m**5 * self.dr * self.dh_te) / (self.r_dist**2)
        self.base_val_le = (self.m**5 * self.dr * self.dh_le) / (self.r_dist**2)
        self.base_val_low = (self.m**5 * self.dr * self.dl) / (self.r_dist**2)

    def run_bpm(
        self,
        device: Optional[str] = None,
        lt: float = 1e6,
        i: float = 0.005,
        alpha_stall: float = 15,
        h: Optional[float] = None,
    ) -> None:
        """Run all BPM noise mechanisms on specified device.
        
        Args:
            device: Torch device ('cpu' or 'cuda'). If None, uses initialized device.
            lt: Turbulence length scale (m).
            i: Turbulence intensity (fraction of flow velocity).
            alpha_stall: Stall angle of attack (degrees).
            h: Trailing edge thickness (m). If None, computed as 1% of chord.
        """
        if device is not None:
            self.device = device
        
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
        freq_5d = self.frequencies[:, None, None, None, None]
        f_co_4d = f_co[None, None, None, :]
        dh_avg = (self.dh_te + self.dh_le) / 2
        d = torch.where(freq_5d < f_co_4d, self.dl, dh_avg)
        k1_val = 2 * np.pi * self.frequencies[:, None] / self.u[None, :]
        k1_bar = k1_val * self.chord / 2
        ke = 3 / 4 / lt
        k1_hat = k1_val.to(torch.float64) / ke
        k1_hat = k1_hat[:, None, None, None, :]
        base_val_ti = (self.m**5 * self.dr * d) / (self.r_dist[None, :, :, :, :]**2)
        beta = 1 - self.m**2
        s_sq = 1 / (
            2 * np.pi * k1_bar / beta**2 + 1 / (1 + 2.4 * k1_bar / beta**2)
        )
        lfc = 10 * s_sq * self.m * k1_bar**(-2) * beta**(-2)
        lfc = lfc[:, None, None, None, :]
        spl_ti = (
            10
            * torch.log10(
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
            + 10 * torch.log10(1 + 9 * self.alpha[None, None, None, None, :] ** 2)
            + 10 * torch.log10(lfc / (1 + lfc))
        )
        spp_ti = torch.zeros_like(spl_ti)
        spp_ti += 10 ** (spl_ti / 10)
        self.spl_ti = 10 * torch.log10(spp_ti.sum(axis=(-1, -2)).mean(axis=-1))

    def teb_noise(self, h: Optional[float] = None) -> None:
        """Compute trailing edge bluntness (TEB) noise.
        
        Args:
            h: Trailing edge thickness (m). If None, uses 1% of chord.
        """
        if h is None:
            h = 1e-2 * self.chord
        else:
            h = torch.full_like(self.chord, h, device=self.device)
            
        delta_avg = calc_delta_avg(self.delta_p, self.delta_s)
        q = h / delta_avg
        st_3prime = st(self.frequencies, h, self.u)
        st_3prime_peak = st_peak_3prime(q, self.psi)
        eta = torch.log10(st_3prime / st_3prime_peak)

        spl_teb = (
            10 * torch.log10((h * self.base_val_te[None, :] * torch.pow(self.m, 0.5)) + 1e-12)
            + g4(q, self.psi)[None, None, None, None, :]
            + g5_tot(q, eta, self.psi)[:, None, None, None, :]
        )
        spp_teb = torch.zeros_like(spl_teb)
        spp_teb += 10 ** (spl_teb / 10)
        self.spl_teb = 10 * torch.log10(spp_teb.sum(dim=(2, 3)).mean(dim=-1))
        
    def lbl_noise(self) -> None:
        """Compute laminar boundary layer (LBL) noise."""
        e = st(self.frequencies, self.delta_p, self.u) / st_peak_prime(self.re_c, self.alpha)
        d = self.re_c / re_c0(self.alpha)

        spl_lbl = (
            10 * torch.log10((self.delta_p * self.base_val_le[None, :]) + 1e-12)
            + g1(e)[:, None, None, None, :]
            + g2(d)[None, None, None, None, :]
            + g3(self.alpha)[None, None, None, None, :]
        )
        spp_lbl = torch.zeros_like(spl_lbl)
        spp_lbl += 10 ** (spl_lbl / 10)
        self.spl_lbl = 10 * torch.log10(spp_lbl.sum(dim=(-1, -2)).mean(dim=-1))
    
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

        as_val = tbl_te_a(torch.abs(torch.log10(st_s / st_1)), self.re_c)
        ap = tbl_te_a(torch.abs(torch.log10(st_p / st_1)), self.re_c)
        b = tbl_te_b(torch.abs(torch.log10(st_s / st_2)), self.re_c)
        a_prime = tbl_te_a(torch.abs(torch.log10(st_s / st_2)), 3 * self.re_c)

        as_exp = as_val[:, None, None, None, :]
        ap_exp = ap[:, None, None, None, :]
        b_exp = b[:, None, None, None, :]
        a_prime_exp = a_prime[:, None, None, None, :]
        spl_s = torch.where(
            self.alpha < alpha_stall,
            10 * torch.log10((self.delta_s * self.base_val_te[None, :]) + 1e-12) + as_exp + k_1 - 3,
            torch.full_like(self.alpha, -torch.inf),
        )
        spl_p = torch.where(
            self.alpha < alpha_stall,
            (
                10 * torch.log10((self.delta_p * self.base_val_te[None, :]) + 1e-12)
                + ap_exp
                + k_1
                - 3
                + delta_k1_val
            ),
            torch.full_like(self.alpha, -torch.inf),
        )
        spl_a = torch.where(
            self.alpha < alpha_stall,
            10 * torch.log10((self.delta_s * self.base_val_te[None, :]) + 1e-12) + b_exp + k_2,
            10 * torch.log10((self.delta_s * self.base_val_low[None, :]) + 1e-12) + a_prime_exp + k_2,
        )
        spp_tbl = torch.zeros_like(spl_a)
        spp_tbl += 10 ** (spl_s / 10) + 10 ** (spl_p / 10) + 10 ** (spl_a / 10)
        self.spl_tbl = 10 * torch.log10(spp_tbl.sum(dim=(-1, -2)).mean(dim=-1))

    def tv_noise(self) -> None:
        """Compute tip vortex (TV) noise."""
        l_tip = calc_l_tip(self.chord[-1], self.alpha[-1])
        m_max = self.m[-1] * (1 + 0.036 * self.alpha[-1])
        st_2prime = st(
            self.frequencies,
            l_tip,
            torch.atleast_1d(torch.tensor(self.acoustic_params.a_inf, dtype=torch.float32, device=self.device) * m_max),
        )

        spl_tip = (
            10
            * torch.log10(
                self.m[-1] ** 2
                * m_max**3
                * l_tip**2
                * self.dh_te[..., -1]
                / self.r_dist[..., -1] ** 2
            )
            - 30.5 * (torch.log10(st_2prime[:, :, None, None]) + 0.3) ** 2
            + 126
        )
        spp_tip = torch.zeros_like(spl_tip)
        spp_tip += 10 ** (spl_tip / 10)
        self.spl_tip = 10 * torch.log10(spp_tip.sum(dim=-1).mean(dim=-1))

    def bwi_noise(
        self,
        xi: float = 1,
        b: float = 1,
        sigma_turb: float = 0.1,
        l_scale: float = 0.125,
    ) -> None:
        """Compute blade-wake interaction (BWI) noise.
        
        Args:
            xi
            b
            sigma_turb: Turbulence intensity as fraction of local velocity.
            l_scale: Integral length scale.
        """
        v_t = torch.sqrt(self.u**2 - (self.v_inf + self.vi) ** 2)
        blade_angles_t = torch.as_tensor(self.propeller.params.blade_angles, dtype=torch.float32, device=self.device)
        y_vt = v_t[None, None, :] * torch.sin(
            self.azimuth_positions[:, None, None]
            + blade_angles_t[None, :, None]
        )
        z_vt = (
            -v_t[None, None, :]
            * torch.cos(
                self.azimuth_positions[:, None, None]
                + blade_angles_t[None, :, None]
            )
        )
        v_t_vec = torch.stack([torch.zeros_like(y_vt), y_vt, z_vt], dim=-1)

        obs_pos = torch.as_tensor(self.propeller.observer_positions, dtype=torch.float32, device=self.device)
        r_vec = (
            obs_pos[:, None, None, None, :]
            - self.r_element[None, ...]
        )
        r_mag = torch.linalg.norm(r_vec, dim=-1)
        sigma = r_mag**2 * (1 - self.m * torch.sum(v_t_vec * r_vec, dim=-1))

        f = self.frequencies[:, None, None, None]
        u = self.u[None, None, None, :]

        omega = 2 * np.pi * f
        phi = phi_ww(f, u, sigma_turb, l_scale)
        l_sq = amiet_l(
            omega, u, self.dr, r_vec[..., 1], sigma, self.acoustic_params.a_inf, xi
        )
        spp_bwi = (
            (omega * r_vec[..., 0] / 4 / np.pi / self.acoustic_params.a_inf / sigma**2) ** 2
            * b
            * self.chord
            * phi
            * l_sq
        )
        self.spl_bwi = 10 * torch.log10(spp_bwi.sum(dim=(2, 3)).mean(dim=-1))