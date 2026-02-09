import torch
import numpy as np
from typing import Dict, Sequence, Tuple, Optional, Union


def _torch_select(
    conditions: Sequence[torch.Tensor],
    choices: Sequence[Union[torch.Tensor, float]],
    default_value: float = 0.0,
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
    """Compute average st1 parameter: (st1 + st2) / 2.

    Args:
        m: Mach number tensor
        alpha: Angle of attack tensor (degrees)

    Returns:
        Average st1 parameter tensor
    """
    return (st1(m) + st2(m, alpha)) / 2


def tbl_te_a_min(a: torch.Tensor) -> torch.Tensor:
    """Compute minimum TBL trailing edge correction (a component).

    Args:
        a: Log-frequency parameter tensor

    Returns:
        Minimum a correction tensor
    """
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
    """Compute maximum TBL trailing edge correction (a component).

    Args:
        a: Log-frequency parameter tensor

    Returns:
        Maximum a correction tensor
    """
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
    """Compute reference TBL parameter a0 based on Reynolds number.

    Args:
        re_c: Reynolds number based on chord tensor

    Returns:
        Reference parameter a0 tensor
    """
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
    """Compute TBL adjustment ratio (a).

    Args:
        a0: Reference parameter a0 tensor

    Returns:
        Adjustment ratio tensor
    """
    return (-20 - tbl_te_a_min(a0)) / (tbl_te_a_max(a0) - tbl_te_a_min(a0))


def tbl_te_a(a: torch.Tensor, re_c: torch.Tensor) -> torch.Tensor:
    """Compute TBL trailing edge correction (a component).

    Args:
        a: Log-frequency parameter tensor
        re_c: Reynolds number based on chord tensor

    Returns:
        a correction tensor
    """
    a0 = tbl_te_a0(re_c)
    return tbl_te_a_min(a) + tbl_te_ar(a0) * (tbl_te_a_max(a) - tbl_te_a_min(a))


def tbl_te_b_min(b: torch.Tensor) -> torch.Tensor:
    """Compute minimum TBL trailing edge correction (b component).

    Args:
        b: Log-frequency parameter tensor

    Returns:
        Minimum b correction tensor
    """
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
    """Compute maximum TBL trailing edge correction (b component).

    Args:
        b: Log-frequency parameter tensor

    Returns:
        Maximum b correction tensor
    """
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
    """Compute reference TBL parameter b0 based on Reynolds number.

    Args:
        re_c: Reynolds number based on chord tensor

    Returns:
        Reference parameter b0 tensor
    """
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
    """Compute TBL adjustment ratio (b).

    Args:
        b0: Reference parameter b0 tensor

    Returns:
        Adjustment ratio tensor
    """
    return (-20 - tbl_te_b_min(b0)) / (tbl_te_b_max(b0) - tbl_te_b_min(b0))


def tbl_te_b(b: torch.Tensor, re_c: torch.Tensor) -> torch.Tensor:
    """Compute TBL trailing edge correction (b component).

    Args:
        b: Log-frequency parameter tensor
        re_c: Reynolds number based on chord tensor

    Returns:
        b correction tensor
    """
    b0 = tbl_te_b0(re_c)
    return tbl_te_b_min(b) + tbl_te_br(b0) * (tbl_te_b_max(b) - tbl_te_b_min(b))


def k1(re_c: torch.Tensor) -> torch.Tensor:
    """Compute k1 parameter based on Reynolds number.

    Args:
        re_c: Reynolds number based on chord tensor

    Returns:
        k1 parameter tensor
    """
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
    """Compute delta_k1 correction for blunt trailing edge.

    Args:
        re_dp: Reynolds number parameter tensor
        alpha: Angle of attack tensor (degrees)

    Returns:
        delta_k1 correction tensor
    """
    return torch.where(
        re_dp <= 5000,
        alpha * (1.43 * torch.log10(re_dp) - 5.29),
        torch.zeros_like(alpha),
    )


def k2(re_c: torch.Tensor, m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute k2 parameter for separated flow noise.

    Args:
        re_c: Reynolds number based on chord tensor
        m: Mach number tensor
        alpha: Angle of attack tensor (degrees)

    Returns:
        k2 parameter tensor
    """
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate acoustic directivity patterns (deprecated, for reference only).

    Args:
        r_element: Element positions (azim, blade, radius, 3) as torch tensor
        r_obs: Observer positions (obs, 3) as torch tensor
        m: Mach number (radius,) as torch tensor
        azimuths: Azimuthal positions (azim,) as torch tensor
        blade_angles: Blade angles (blade,) as torch tensor

    Returns:
        Tuple of (r_mag, dh_te, dh_le, dl) directivity factor tensors
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
    e_yl = torch.stack(
        [torch.zeros_like(psi_total), torch.cos(psi_total), torch.sin(psi_total)],
        dim=-1,
    )

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

    term1 = 2 * (sin_theta_2**2) * (sin_phi**2)
    term2 = (1 + m * cos_theta) * (1 + 0.2 * m * cos_theta) ** 2
    dh_te = term1 / term2
    term1 = 2 * (cos_theta_2**2) * (sin_phi**2)
    term2 = (1 + m * cos_theta) ** 3
    dh_le = term1 / term2
    term1 = (torch.sin(theta) ** 2) * (sin_phi**2)
    term2 = (1 + m * cos_theta) ** 4
    dl = term1 / term2

    return torch.squeeze(r_mag), dh_te, dh_le, dl


def st1_prime(re_c: torch.Tensor) -> torch.Tensor:
    """Compute peak Strouhal number reference parameter.

    Evaluates the Strouhal number at maximum sound pressure level as a function
    of Reynolds number based on chord length.

    Args:
        re_c: Reynolds number based on chord tensor

    Returns:
        st1_prime parameter tensor representing reference Strouhal frequency
    """
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
    """Compute peak Strouhal number with angle of attack correction.

    Args:
        re_c: Reynolds number based on chord tensor
        alpha: Angle of attack tensor (degrees)

    Returns:
        Peak Strouhal number tensor
    """
    return st1_prime(re_c) * torch.pow(10.0, -0.04 * alpha)


def g1(e: torch.Tensor) -> torch.Tensor:
    """Compute g1 function for laminar boundary layer noise.

    Args:
        e: Dimensionless frequency parameter tensor

    Returns:
        g1 function value tensor
    """
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
    """Compute g2 function for laminar boundary layer noise.

    Args:
        d: Dimensionless Reynolds number parameter tensor

    Returns:
        g2 function value tensor
    """
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
    """Compute reference Reynolds number based on angle of attack.

    Args:
        alpha: Angle of attack tensor (degrees)

    Returns:
        Reference Reynolds number tensor
    """
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
    """Compute g3 function for angle of attack effect on LBL noise.

    Args:
        alpha: Angle of attack tensor (degrees)

    Returns:
        g3 function value tensor
    """
    return 171.04 - 3.03 * alpha


def calc_delta_avg(delta_p: torch.Tensor, delta_s: torch.Tensor) -> torch.Tensor:
    """Compute average boundary layer thickness from pressure and suction side.

    Args:
        delta_p: Pressure side boundary layer thickness tensor
        delta_s: Suction side boundary layer thickness tensor

    Returns:
        Average boundary layer thickness tensor
    """
    return (delta_p + delta_s) / 2


def st_peak_3prime(q: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute peak Strouhal number for trailing edge bluntness noise.

    Args:
        q: Bluntness ratio tensor
        psi: Boat tail angle tensor (degrees)

    Returns:
        Peak Strouhal number tensor
    """
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
    """Compute g4 function for trailing edge bluntness noise.

    Args:
        q: Bluntness ratio tensor
        psi: Boat tail angle tensor (degrees)

    Returns:
        g4 function value tensor
    """
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
    """Compute mu parameter for trailing edge bluntness noise calculations.

    Args:
        q: Bluntness ratio tensor

    Returns:
        mu parameter tensor
    """
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
    """Compute m parameter for g5 calculation in TEB noise.

    Args:
        q: Bluntness ratio tensor

    Returns:
        m parameter tensor
    """
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
    """Compute eta0 parameter for g5 calculation.

    Args:
        m: m parameter tensor
        mu: mu parameter tensor

    Returns:
        eta0 parameter tensor
    """
    return -torch.sqrt(m**2 * mu**4 / (6.25 + m**2 * mu**2))


def calc_k(m: torch.Tensor, mu: torch.Tensor, eta0: torch.Tensor) -> torch.Tensor:
    """Compute k parameter for g5 function in TEB noise.

    Args:
        m: m parameter tensor
        mu: mu parameter tensor
        eta0: eta0 parameter tensor

    Returns:
        k parameter tensor
    """
    return 2.5 * torch.sqrt(1 - (eta0 / mu) ** 2) - 2.5 - m * eta0


def g5(
    m: torch.Tensor,
    mu: torch.Tensor,
    eta: torch.Tensor,
    eta0: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Compute g5 function for trailing edge bluntness noise.

    Args:
        m: m parameter tensor
        mu: mu parameter tensor
        eta: dimensionless frequency parameter tensor
        eta0: eta0 parameter tensor
        k: k parameter tensor

    Returns:
        g5 function value tensor
    """
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
    """Compute g5_0 reference function for TEB noise.

    Args:
        q: Bluntness ratio tensor
        eta: Dimensionless frequency parameter tensor

    Returns:
        g5_0 function value tensor
    """
    q_0 = 6.724 * q**2 - 4.019 * q + 1.107
    m = calc_m(q_0)
    mu = calc_mu(q_0)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    return g5(m, mu, eta, eta0, k)


def g5_tot(q: torch.Tensor, eta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute total g5 function with boat tail angle correction.

    Args:
        q: Bluntness ratio tensor
        eta: Dimensionless frequency parameter tensor
        psi: Boat tail angle tensor (degrees)

    Returns:
        Total g5 function value tensor
    """
    g5_0_val = g5_0(q, eta)
    m = calc_m(q)
    mu = calc_mu(q)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    g5_val = g5(m, mu, eta, eta0, k)

    delta_g5 = g5_val - g5_0_val
    correction = (
        0.0714 * psi * torch.where(delta_g5 < 0, delta_g5, torch.zeros_like(delta_g5))
    )
    # correction = (
    #     0.0714 * psi * delta_g5
    # )
    return g5_0_val + correction


def calc_l_tip(chord: torch.Tensor, alpha_tip: torch.Tensor) -> torch.Tensor:
    """Compute tip vortex length scale.

    Args:
        chord: Chord length tensor (m)
        alpha_tip: Angle of attack at tip tensor (degrees)

    Returns:
        Tip length scale tensor
    """
    return chord * 0.008 * alpha_tip


class BPM:
    """BPM broadband noise prediction model using PyTorch.

    Implements the BPM model for computing broadband noise sources from propeller
    blades including turbulent boundary layer (TBL), laminar boundary layer (LBL),
    trailing edge (TE), tip vortex (TV), and blade-wake interaction (BWI) noise.
    All computations are device-agnostic (CPU/GPU).
    """

    def __init__(
        self,
        frequencies: Union[np.ndarray, torch.Tensor],
        r: Union[np.ndarray, torch.Tensor],
        dr: Union[np.ndarray, torch.Tensor],
        chord: Union[np.ndarray, torch.Tensor],
        alpha: Union[np.ndarray, torch.Tensor],
        vi: Union[np.ndarray, torch.Tensor],
        u: Union[np.ndarray, torch.Tensor],
        re_c: Union[np.ndarray, torch.Tensor],
        m: Union[np.ndarray, torch.Tensor],
        delta_p: Union[np.ndarray, torch.Tensor],
        delta_s: Union[np.ndarray, torch.Tensor],
        boat_tail_angle: Union[np.ndarray, float],
        src_times: Union[np.ndarray, torch.Tensor],
        a_inf: float,
        rho: float,
        omega: float,
        blade_angles: Union[np.ndarray, torch.Tensor],
        twist: Union[np.ndarray, torch.Tensor],
        com_shift_forward: Union[float, np.ndarray, torch.Tensor],
        com_shift_up: Union[float, np.ndarray, torch.Tensor],
        observer_time_range: float,
        num_obs_times: int,
        device: str,
    ) -> None:
        """Initialize BPM model with explicit geometry and acoustic parameters.

        Args:
            frequencies: Frequency array for acoustic analysis (Hz).
            r, dr, chord, alpha, vi, u, re_c, m, delta_p, delta_s: per-section arrays.
            boat_tail_angle: per-section or scalar boat tail angle.
            src_times: Source emission times array.
            a_inf: Ambient sound speed (m/s).
            omega: Propeller rotation speed (rad/s).
            blade_angles: Blade phase offsets (rad).
            twist: Per-section twist (degrees).
            com_shift_forward, com_shift_up: chord offset scalars.
            observer_time_range: duration for uniform observer time grid.
            num_obs_times: number of uniform observer time steps.
            device: Torch device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.dtype = torch.float32

        # Core arrays (move to device)
        self.frequencies = torch.as_tensor(
            frequencies, dtype=self.dtype, device=self.device
        )
        self.r = torch.as_tensor(r, dtype=self.dtype, device=self.device)
        self.dr = torch.as_tensor(dr, dtype=self.dtype, device=self.device)
        self.chord = torch.as_tensor(chord, dtype=self.dtype, device=self.device)
        self.alpha = torch.as_tensor(alpha, dtype=self.dtype, device=self.device)
        self.vi = torch.as_tensor(vi, dtype=self.dtype, device=self.device)
        self.u = torch.as_tensor(u, dtype=self.dtype, device=self.device)
        self.re_c = torch.as_tensor(re_c, dtype=self.dtype, device=self.device)
        self.m = torch.as_tensor(m, dtype=self.dtype, device=self.device)
        self.delta_p = torch.as_tensor(delta_p, dtype=self.dtype, device=self.device)
        self.delta_s = torch.as_tensor(delta_s, dtype=self.dtype, device=self.device)
        self.psi = torch.as_tensor(
            boat_tail_angle, dtype=self.dtype, device=self.device
        )

        # Acoustic & kinematic parameters
        self.tau = torch.as_tensor(src_times, dtype=self.dtype, device=self.device)
        self.a_inf = torch.tensor(a_inf, dtype=self.dtype, device=self.device)
        self.rho = torch.tensor(rho, dtype=self.dtype, device=self.device)
        self.omega = torch.tensor(omega, dtype=self.dtype, device=self.device)
        self.blade_angles = torch.as_tensor(
            blade_angles, dtype=self.dtype, device=self.device
        )
        self.twist_rad = torch.deg2rad(
            torch.as_tensor(twist, dtype=self.dtype, device=self.device)
        )

        # Geometry offsets
        self.com_shift_forward = torch.as_tensor(
            com_shift_forward, dtype=self.dtype, device=self.device
        )
        self.com_shift_up = torch.as_tensor(
            com_shift_up, dtype=self.dtype, device=self.device
        )

        # Observer interpolation controls
        self.observer_time_range = float(observer_time_range)
        self.num_obs_times = int(num_obs_times)

        # Dimensions
        self.nt, self.ns, self.nb = (
            self.tau.shape[0],
            self.r.shape[0],
            self.blade_angles.shape[0],
        )

    def generate_trajectory_and_basis(self) -> None:
        """Generate 4D blade element trajectory and time-varying local basis vectors.

        Calculates the spatial position of all blade elements at each emission time
        and computes the time-varying local basis vectors (thrust, radial, tangential)
        in the fixed global coordinate frame. Results are stored as instance attributes:
        - pos_fixed: Global positions of shape (time, section, blade, 3)
        - e_xl, e_yl, e_zl: Local basis vectors of shape (time, blade, 3)
        """
        # Blade section positions in blade-fixed (rotating) frame.
        pos_moving: torch.Tensor = torch.stack(
            [
                self.chord * self.com_shift_up * torch.sin(self.twist_rad),
                self.r,
                self.chord * self.com_shift_forward * torch.sin(self.twist_rad),
            ],
            dim=-1,
        )

        # Compute rotation angles at each time step and blade
        angles: torch.Tensor = (
            self.omega * self.tau[:, None] + self.blade_angles[None, :]
        )
        c: torch.Tensor = torch.cos(angles)
        s: torch.Tensor = torch.sin(angles)

        # e_xl: thrust axis (fixed along X for a non-tilted propeller).
        self.e_xl = torch.zeros((self.nt, self.nb, 3), device=self.device)
        self.e_xl[..., 0] = 1.0

        # e_yl: spanwise axis (radial).
        self.e_yl = torch.stack([torch.zeros_like(angles), c, s], dim=-1)

        # e_zl: tangential axis (direction of motion).
        self.e_zl = torch.cross(self.e_xl, self.e_yl, dim=-1)

        # Build rotation matrices (rotation around x-axis)
        # Format: (time, blade, 3x3)
        rot: torch.Tensor = torch.zeros(
            (self.nt, self.nb, 3, 3), dtype=self.dtype, device=self.device
        )
        rot[..., 0, 0] = 1.0
        rot[..., 1, 1], rot[..., 1, 2] = c, -s
        rot[..., 2, 1], rot[..., 2, 2] = s, c

        # Transform positions from blade frame to fixed frame using rotation matrices.
        # Shape: (time, section, blade, xyz)
        self.pos_fixed: torch.Tensor = torch.einsum(
            "tbik,sk->tsbi", rot, pos_moving
        ).contiguous()

    def get_emission_geometry(
        self, r_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute acoustic directivity factors and distances to observer.

        Calculates the spatial relationship between blade elements and observer,
        including distance magnitude and directivity patterns for leading edge (LE),
        trailing edge (TE), and dipole (low-frequency) acoustic sources.

        Args:
            r_obs: Observer positions of shape (n_obs, 3) in global frame (x, y, z).

        Returns:
            Tuple of four tensors, each shape (time, section, blade, n_obs):
                - r_mag: Distance from source to observer (m)
                - dh_te: Trailing edge directivity factor (dimensionless)
                - dh_le: Leading edge directivity factor (dimensionless)
                - dl: Low-frequency directivity factor (dimensionless)
        """
        # r_vec: (nt, ns, nb, n_obs, 3)
        r_vec = r_obs[None, None, None, :, :] - self.pos_fixed
        r_mag = torch.linalg.norm(r_vec, dim=-1)
        inv_r = 1.0 / r_mag
        unit_r = r_vec * inv_r[..., None]

        # Projections onto local basis
        obs_xl = torch.einsum('tsbni,tbi->tsbn', unit_r, self.e_xl)
        obs_yl = torch.einsum('tsbni,tbi->tsbn', unit_r, self.e_yl)
        obs_zl = torch.einsum('tsbni,tbi->tsbn', unit_r, self.e_zl)

        # Elevation between propeller plane and observer.
        phi = torch.asin(obs_xl)

        # Azimuth between propeller plane and observer.
        theta = torch.atan2(obs_zl, obs_yl) + torch.pi / 2

        sin_phi_sq = torch.sin(phi) ** 2
        # k = 0.5
        # sin_phi_sq = k*torch.sin(phi) ** 2 / (k**2 * torch.cos(phi)**2 + torch.sin(phi)**2)**(3/2)

        two_sin_theta_2_sq = 2 * torch.sin(theta / 2) ** 2
        two_cos_theta_2_sq = 2 * torch.cos(theta / 2) ** 2
        sin_theta_sq = torch.sin(theta) ** 2

        m = self.m[None, :, None, None]
        doppler = 1.0 + m * torch.cos(theta)

        # Doppler-corrected directivity factors.
        dh_te = (two_sin_theta_2_sq * sin_phi_sq) / (
            doppler * (1.0 + 0.2 * m * torch.cos(theta)) ** 2
        )
        dh_le = (two_cos_theta_2_sq * sin_phi_sq) / (doppler**3)
        dl = (sin_theta_sq * sin_phi_sq) / (doppler**4)

        return r_mag, dh_te, dh_le, dl

    def run_forward_bpm(
        self,
        observer_positions: np.ndarray,
        bpm_obs_times: torch.Tensor,
        bpm_output_times: torch.Tensor,
        lt: float,
        i: float,
        alpha_stall: float,
    ) -> Dict[str, torch.Tensor]:
        """Run full BPM suite and return individual SPL component tensors.

        Computes all broadband noise sources (TBL, LBL, TEB, TI, TV), returning
        raw spectral power density (SPP) tensors for each component. Results represent
        physical acoustic emission at source time without propagation corrections.

        Args:
            observer_positions: Observer positions array of shape (n_obs, 3) in meters.
                Coordinates in (x, y, z) format in global fixed frame.

        Returns:
            Dictionary mapping component names to SPP tensors of shape
            (n_freq, time, section, blade, n_obs):
                - 'tbl': Turbulent boundary layer noise
                - 'lbl': Laminar boundary layer noise
                - 'teb': Trailing edge bluntness noise
                - 'ti': Turbulence ingestion noise
                - 'tv': Tip vortex noise
        """
        r_obs = torch.as_tensor(
            observer_positions, dtype=self.dtype, device=self.device
        )

        self.generate_trajectory_and_basis()
        self.pos_fixed = self.interpolate_positions(
            bpm_output_times, bpm_obs_times, self.pos_fixed
        )

        # 1. Geometry and Base Values at Emission Time
        r_mag, dh_te, dh_le, dl = self.get_emission_geometry(r_obs)

        m5_dr = (self.m**5 * self.dr)[None, :, None, None]
        r_mag_sq = r_mag**2
        self.base_val_te = (m5_dr * dh_te) / r_mag_sq
        self.base_val_le = (m5_dr * dh_le) / r_mag_sq
        self.base_val_low = (m5_dr * dl) / r_mag_sq

        # 2. Compute raw SPP tensors (5D: n_freq, nt, ns, nb, n_obs).
        components_raw = {
            "tbl": self.tbl_noise(alpha_stall=alpha_stall).sum(dim=(2, 3)),
            "lbl": self.lbl_noise().sum(dim=(2, 3)),
            "teb": self.teb_noise().sum(dim=(2, 3)),
            "ti": self.ti_noise(lt=lt, i=i).sum(dim=(2, 3)),
            "tv": self.tv_noise().sum(dim=(2, 3)),
        }
        return components_raw

    def ti_noise(self, lt: float, i: float) -> torch.Tensor:
        """Compute turbulence ingestion (TI) broadband noise.

        Models the interaction of ingested turbulence with the blade surfaces,
        accounting for frequency-dependent selectivity and Mach-number compressibility.

        Args:
            lt: Integral turbulence length scale (m). Default: 1e6 (uniform flow).
            i: Turbulence intensity as fraction (0-1). Default: 0.01 (1%).

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """
        # Use consistent shape: (nf, 1, ns, 1, 1) for all per-section parameters.
        # Base values: (nt, ns, nb, n_obs) -> (1, nt, ns, nb, n_obs).
        bv_le = self.base_val_le[None, :, :, :, :]
        bv_low = self.base_val_low[None, :, :, :, :]

        # Frequency-dependent parameters.
        f_co = 10.0 * self.u / (np.pi * self.chord)  # (ns,)
        k1_val = 2.0 * np.pi * self.frequencies[:, None] / self.u[None, :]  # (nf, ns)

        k1_bar = k1_val * self.chord[None, :] * 0.5  # (nf, ns)

        # Reshape for 5D: (nf, 1, ns, 1, 1).
        f_co_5d = f_co[None, None, :, None, None]
        freq_2d = self.frequencies[:, None, None, None, None]

        # Selectivity based on frequency.
        bv_ti = torch.where(freq_2d < f_co_5d, bv_low, bv_le)

        # Mach and compressibility: (ns,) -> (nf, 1, ns, 1, 1).
        beta_sq = 1.0 - self.m**2  # (ns,)

        # LFC correction factor.
        k1_beta = k1_bar / beta_sq[None, :]  # (nf, ns)
        denom = 2.0 * np.pi * k1_beta + 1.0 / (1.0 + 2.4 * k1_beta)
        s_sq = 1.0 / denom
        lfc = 10.0 * s_sq * self.m[None, :] * (k1_bar ** 2) / beta_sq[None, :]
        lfc_term = torch.clamp(lfc / (1.0 + lfc), min=1e-15)  # (nf, ns)
        lfc_5d = lfc_term[:, None, :, None, None]

        # Spectral power density.
        k1_hat = k1_val / (3.0 / (4.0 * lt))  # (nf, ns)
        phi_term = (k1_hat**3) / ((1.0 + k1_hat**2) ** (7.0 / 3.0))  # (nf, ns)

        # Alpha term.
        alpha_sq = (self.alpha**2)[None, None, :, None, None]  # (1, 1, ns, 1, 1)

        inner_val = (
            (self.rho**2)
            * (self.a_inf**4)
            * lt
            * 0.5
            * (i**2)
            * phi_term.view(self.frequencies.shape[0], 1, self.ns, 1, 1)
            * bv_ti
        )

        spl_ti = (
            10.0 * torch.log10(torch.clamp(inner_val, min=1e-20))
            + 78.4
            + 10.0 * torch.log10(1.0 + 9.0 * alpha_sq)
            + 10.0 * torch.log10(lfc_5d)
        )

        return 10.0 ** (spl_ti / 10.0)  # (nf, nt, ns, nb, n_obs)

    def teb_noise(self, h: Optional[float] = None) -> torch.Tensor:
        """Compute trailing edge bluntness (TEB) broadband noise.

        Models the scattering of incoming vorticity by blunt trailing edges,
        including effects of boat tail angle and boundary layer thickness.

        Args:
            h: Trailing edge thickness (m). If None, defaults to 1% of chord length.

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """
        h_val = (
            (self.chord * 1e-5) if h is None else torch.full_like(self.chord, h)
        )  # (ns,)

        delta_avg = (self.delta_p + self.delta_s) * 0.5  # (ns,)
        psi = self.psi  # (ns,)
        m = self.m  # (ns,)

        # Expand to (1, 1, ns, 1, 1) for broadcasting.
        h_5d = h_val[None, None, :, None, None]
        delta_avg_5d = delta_avg[None, None, :, None, None]
        psi_5d = psi[None, None, :, None, None]
        m_5d = m[None, None, :, None, None]

        q = h_5d / delta_avg_5d

        # st() returns (nf, ns), expand to (nf, 1, ns, 1, 1).
        st_3p = st(self.frequencies, h_val, self.u)[:, None, :, None, None]
        st_3p_pk = st_peak_3prime(q, psi_5d)
        eta = torch.log10(st_3p / st_3p_pk)

        # base_val_te: (nt, ns, nb, n_obs) -> (1, nt, ns, nb, n_obs).
        bv_te = self.base_val_te[None, :, :, :, :]
        log_h_bv = 10 * torch.log10((h_5d * bv_te * torch.sqrt(m_5d)) + 1e-12)

        spl_teb = log_h_bv + g4(q, psi_5d) + g5_tot(q, eta, psi_5d)

        return 10 ** (spl_teb / 10)  # (nf, nt, ns, nb, n_obs)

    def lbl_noise(self) -> torch.Tensor:
        """Compute laminar boundary layer (LBL) broadband noise.

        Models noise from instabilities in laminar boundary layers, which occurs
        at low frequencies and low to moderate angles of attack.

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """
        # Expand to (1, 1, ns, 1, 1) for broadcasting.
        delta_p_5d = self.delta_p[None, None, :, None, None]
        re_c_5d = self.re_c[None, None, :, None, None]
        alpha_5d = self.alpha[None, None, :, None, None]

        # st() returns (nf, ns), expand to (nf, 1, ns, 1, 1).
        e = st(self.frequencies, self.delta_p, self.u)[
            :, None, :, None, None
        ] / st_peak_prime(re_c_5d, alpha_5d)
        d = re_c_5d / re_c0(alpha_5d)

        # base_val_le: (nt, ns, nb, n_obs) -> (1, nt, ns, nb, n_obs).
        bv_le = self.base_val_le[None, :, :, :, :]

        log_dp_bv = 10 * torch.log10((delta_p_5d * bv_le) + 1e-12)
        spl_lbl = log_dp_bv + g1(e) + g2(d) + g3(alpha_5d)

        return 10 ** (spl_lbl / 10)  # (nf, nt, ns, nb, n_obs)

    def tbl_noise(self, alpha_stall: float = 15.0) -> torch.Tensor:
        """Compute turbulent boundary layer (TBL) broadband noise.

        Models the dominant broadband noise source from turbulent boundary layer
        pressure fluctuations on blade surfaces. Includes suction-side, pressure-side,
        and separated-flow components.

        Args:
            alpha_stall: Stall angle of attack (degrees). Default: 15.0.

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """

        # Expand to (1, 1, ns, 1, 1) for broadcasting.
        m_5d = self.m[None, None, :, None, None]
        alpha_5d = self.alpha[None, None, :, None, None]
        re_c_5d = self.re_c[None, None, :, None, None]
        delta_p_5d = self.delta_p[None, None, :, None, None]
        delta_s_5d = self.delta_s[None, None, :, None, None]
        chord_5d = self.chord[None, None, :, None, None]

        # Compute parameters (broadcasting handles nt, nb, and n_obs dims).
        st_1 = st1(m_5d)
        st_2 = st2(m_5d, alpha_5d)
        k_1 = k1(re_c_5d)
        k_2 = k2(re_c_5d, m_5d, alpha_5d)

        # st() expects (n_freq, ns). Result: (nf, 1, ns, 1, 1).
        st_p = st(self.frequencies, self.delta_p, self.u)[:, None, :, None, None]
        st_s = st(self.frequencies, self.delta_s, self.u)[:, None, :, None, None]

        # Trailing edge corrections.
        log_st_s_st1 = torch.abs(torch.log10(st_s / st_1))
        log_st_p_st1 = torch.abs(torch.log10(st_p / st_1))
        log_st_s_st2 = torch.abs(torch.log10(st_s / st_2))

        as_val = tbl_te_a(log_st_s_st1, re_c_5d)
        ap = tbl_te_a(log_st_p_st1, re_c_5d)
        b = tbl_te_b(log_st_s_st2, re_c_5d)
        a_prime = tbl_te_a(log_st_s_st2, 3 * re_c_5d)

        # Base values: (nt, ns, nb, n_obs) -> (1, nt, ns, nb, n_obs).
        bv_te = self.base_val_te[None, :, :, :, :]
        bv_low = self.base_val_low[None, :, :, :, :]

        log_ds_bv_te = 10 * torch.log10((delta_s_5d * bv_te) + 1e-12)
        log_dp_bv_te = 10 * torch.log10((delta_p_5d * bv_te) + 1e-12)
        log_ds_bv_low = 10 * torch.log10((delta_s_5d * bv_low) + 1e-12)

        # SPL components.
        delta_k1_val = delta_k1((re_c_5d / chord_5d * delta_p_5d), alpha_5d)
        alpha_mask = alpha_5d < alpha_stall

        spl_s = torch.where(
            alpha_mask,
            log_ds_bv_te + as_val + k_1 - 3,
            torch.full_like(as_val, -torch.inf),
        )
        spl_p = torch.where(
            alpha_mask,
            log_dp_bv_te + ap + k_1 - 3 + delta_k1_val,
            torch.full_like(ap, -torch.inf),
        )
        spl_a = torch.where(
            alpha_mask, log_ds_bv_te + b + k_2, log_ds_bv_low + a_prime + k_2
        )

        # Final raw SPP: (nf, nt, ns, nb, n_obs).
        return 10 ** (spl_s / 10) + 10 ** (spl_p / 10) + 10 ** (spl_a / 10)

    def tv_noise(self) -> torch.Tensor:
        """Compute tip vortex (TV) broadband noise.

        Models the noise from unsteady loading fluctuations induced by the tip vortex.
        Computed only at the tip section, with zeros elsewhere.

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """
        # Tip-specific parameters.
        chord_tip = self.chord[-1]
        alpha_tip = self.alpha[-1]
        m_tip = self.m[-1]

        l_tip = calc_l_tip(chord_tip, alpha_tip)
        m_max = m_tip * (1 + 0.036 * alpha_tip)

        # Strouhal: ensure shape (nf, 1, 1, 1, 1) regardless of st() return shape.
        st_freq = st(self.frequencies, l_tip, (self.a_inf * m_max).unsqueeze(0))
        st_2p = st_freq[:, :1, None, None, None]  # 2D -> (nf, 1, 1, 1, 1)

        # Directivity: extract tip section and expand to (1, nt, 1, nb, n_obs).
        bv_te_tip = self.base_val_te[:, -1:, :, :][None, :, :, :, :]

        # Mach and geometry factor.
        m_factor = m_tip**2 * m_max**3 * l_tip**2
        log_st = torch.log10(torch.abs(st_2p) + 1e-12)

        # spl_tip: (nf, 1, nt, 1, nb, n_obs) broadcasts to (nf, nt, 1, nb, n_obs).
        spl_tip = (
            10 * torch.log10(m_factor * bv_te_tip + 1e-12)
            - 30.5 * (log_st + 0.3) ** 2
            + 126
        )

        # Place into full 5D output tensor (nf, nt, ns, nb, n_obs).
        spp_full = torch.zeros(
            (
                self.frequencies.shape[0],
                self.nt,
                self.ns,
                self.nb,
                self.base_val_te.shape[-1],
            ),
            device=self.device,
            dtype=self.dtype,
        )

        # Assign to the last radial station (-1).
        spp_full[:, :, -1:, :, :] = 10 ** (spl_tip / 10)

        return spp_full

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

        # 1. Permute to put time (interpolation dim) at the end.
        # From (nt, n_sec, n_b, no) -> (no, n_sec, n_b, nt)
        x_old_p = x_old.permute(3, 1, 2, 0).contiguous()
        y_old_p = y_old.permute(3, 1, 2, 0).contiguous()

        # 2. Prepare x_new: (n_steps, no) -> (no, n_sec, n_b, n_steps).
        # Transpose x_new to (no, n_steps), then expand to match dimensions
        x_new_p = (
            x_new.T.view(n_obs, 1, 1, n_steps).expand(-1, n_sec, n_b, -1).contiguous()
        )

        # 3. Find bracketing indices.
        # searchsorted works on the last dimension (nt)
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, nt - 1)

        # 4. Gather bracketing points.
        # Dim 3 is the time dimension we are interpolating within
        x0 = torch.gather(x_old_p, 3, idx - 1)
        x1 = torch.gather(x_old_p, 3, idx)
        y0 = torch.gather(y_old_p, 3, idx - 1)
        y1 = torch.gather(y_old_p, 3, idx)

        # 5. Linear interpolation formula.
        # (y - y0) / (x - x0) = (y1 - y0) / (x1 - x0)
        weights = (x_new_p - x0) / (x1 - x0 + 1e-12)
        interp_vals = y0 + weights * (y1 - y0)

        # 6. Reduction.
        # Sum across sections (dim 1) and blades (dim 2)
        # Result shape: (no, n_steps)
        summed = torch.sum(interp_vals, dim=(1, 2))

        # Transpose back to (n_steps, n_observers)
        return summed.T

    def interpolate_positions(
        self,
        x_new: torch.Tensor,
        x_old: torch.Tensor,
        y_old: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate blade positions onto a new observer time grid.

        Args:
            x_new: Observer time grid, shape (n_steps, n_obs).
            x_old: Retarded times, shape (nt, n_sec, n_b, n_obs).
            y_old: Source positions, shape (nt, n_sec, n_b, 3).

        Returns:
            Interpolated positions with shape (n_steps, n_sec, n_b, n_obs, 3).
        """
        nt, n_sec, n_b, _ = x_old.shape
        n_steps, n_obs = x_new.shape

        # Prepare x_old: (n_obs, n_sec, n_b, nt).
        x_old_p = x_old.permute(3, 1, 2, 0).contiguous()

        # Prepare x_new: (n_obs, n_sec, n_b, n_steps).
        x_new_p = (
            x_new.T.view(n_obs, 1, 1, n_steps).expand(-1, n_sec, n_b, -1).contiguous()
        )

        # Bracketing indices: (n_obs, n_sec, n_b, n_steps).
        idx = torch.searchsorted(x_old_p, x_new_p)
        idx = torch.clamp(idx, 1, nt - 1)

        # Prepare y_old for broadcasting.
        y_old_p = y_old.permute(3, 1, 2, 0).contiguous()  # (3, n_sec, n_b, nt)

        # Expand y to add n_obs, then gather bracketing points.
        y_old_exp = y_old_p.unsqueeze(1).expand(-1, n_obs, -1, -1, -1)

        # Expand idx to match coordinate dimension.
        idx_exp = idx.unsqueeze(0).expand(3, -1, -1, -1, -1)

        y0 = torch.gather(y_old_exp, 4, idx_exp - 1)
        y1 = torch.gather(y_old_exp, 4, idx_exp)

        # Gather x values (already have n_obs dimension).
        x0 = torch.gather(x_old_p, 3, idx - 1)
        x1 = torch.gather(x_old_p, 3, idx)

        # Linear interpolation.
        t = (x_new_p - x0) / (x1 - x0 + 1e-12)
        interp_vals = y0 + t.unsqueeze(0) * (y1 - y0)

        # Permute to final shape: (n_steps, n_sec, n_b, n_obs, 3).
        return interp_vals.permute(4, 2, 3, 1, 0)
