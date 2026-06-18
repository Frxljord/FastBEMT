import torch
import numpy as np
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict, Sequence, Tuple, Optional, Union


def _load_bpm_component(name: str) -> ModuleType:
    """Load a BPM component module from the sibling BPM directory."""
    component_path = Path(__file__).with_suffix("") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{__name__}_{name}", component_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load BPM component module: {component_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tbl_component = _load_bpm_component("tbl")
_lbl_component = _load_bpm_component("lbl")
_teb_component = _load_bpm_component("teb")
_ti_component = _load_bpm_component("ti")
_tv_component = _load_bpm_component("tv")


def _torch_select(
    conditions: Sequence[torch.Tensor],
    choices: Sequence[Union[torch.Tensor, float]],
    default_value: float = 0.0,
) -> torch.Tensor:
    '''Torch-based conditional selection similar to np.select.

    Args:
        conditions: List of boolean tensors.
        choices: List of tensors/scalars for each condition.
        default_value: Default value when no condition is met.

    Returns:
        Selected values based on conditions.
    '''
    result = torch.full_like(conditions[0], default_value, dtype=conditions[0].dtype)
    for cond, choice in zip(reversed(conditions), reversed(choices)):
        if not isinstance(choice, torch.Tensor):
            choice = torch.tensor(choice, dtype=result.dtype, device=result.device)
        result = torch.where(cond, choice, result)
    return result


def st(f: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    '''Compute Strouhal number: St = f * l / u.

    Args:
        f: Frequency, shape (n_freqs,).
        l: Length scale, shape (n_sections,).
        u: Velocity, shape (n_sections,).

    Returns:
        Strouhal number, shape (n_freqs, n_sections).
    '''
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
        alpha <= 1.333,
        (alpha > 1.333) & (alpha <= 12.5),
        alpha > 12.5,
    ]
    choices = [
        torch.ones_like(alpha),
        torch.pow(10.0, 0.0054 * torch.pow(alpha - 1.333, 2)),
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
        -31.313 * b + 1.854,
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
        alpha <= gamma0 - gamma,
        (alpha > gamma0 - gamma) & (alpha <= gamma0 + gamma),
        alpha > gamma0 + gamma,
    ]

    choices = [
        torch.full_like(alpha, -1000.0),
        torch.sqrt(
            torch.clamp(beta**2 - (beta / gamma) ** 2 * (alpha - gamma0) ** 2, min=0)
        )
        + beta0,
        torch.full_like(alpha, -12.0),
    ]

    tmp = _torch_select(conditions, choices, default_value=-1000.0)
    return k1(re_c) + tmp


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
    """Compute total g5 function with bounded boat tail angle correction.

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

    # Retain the FastBEMT guard against the positive boat-tail correction branch,
    # which can otherwise drive unrealistically large TEB levels for this geometry.
    delta_g5 = g5_val - g5_0_val
    correction = 0.0714 * psi * torch.where(
        delta_g5 < 0, delta_g5, torch.zeros_like(delta_g5)
    )
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
    '''Brooks-Pope-Marcolini broadband noise model.

    Implements BPM noise prediction for turbulent boundary layer (TBL),
    laminar boundary layer (LBL), trailing edge (TE), tip vortex (TV),
    and blade-wake interaction (BWI) noise sources. GPU-accelerated via PyTorch.
    '''

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
        kinematics: Optional[object] = None,
        section_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        c1: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        '''Initialize BPM model.

        Args:
            frequencies: Frequency array (Hz), shape (n_freqs,).
            r: Radial positions (m), shape (n_sections,).
            dr: Radial widths (m), shape (n_sections,).
            chord: Chord lengths (m), shape (n_sections,).
            c1: Distance from pitch axis to leading edge (m), shape
                (n_sections,). If omitted, defaults to 0.25 * chord like the
                BroadbandBPM.jl examples.
            alpha: Angle of attack (deg), shape (n_sections,).
            vi: Induced velocity (m/s), shape (n_sections,).
            u: Relative velocity (m/s), shape (n_sections,).
            re_c: Reynolds number, shape (n_sections,).
            m: Mach number, shape (n_sections,).
            delta_p: Pressure side displacement thickness (m), shape (n_sections,).
            delta_s: Suction side displacement thickness (m), shape (n_sections,).
            boat_tail_angle: Trailing edge angle (deg), scalar or shape (n_sections,).
            src_times: Source emission times (s), shape (n_times,).
            a_inf: Speed of sound (m/s).
            rho: Density (kg/m³).
            omega: Angular velocity (rad/s).
            blade_angles: Blade phase offsets (rad), shape (n_blades,).
            twist: Twist angles (deg), shape (n_sections,).
            com_shift_forward: Chordwise center offset, scalar or shape (n_sections,).
            com_shift_up: Normal center offset, scalar or shape (n_sections,).
            observer_time_range: Duration for observer time grid (s).
            num_obs_times: Number of observer time steps.
            device: PyTorch device ('cpu' or 'cuda').
            kinematics: Optional shared Kinematics object. When provided, BPM
                uses its source-time rotations and section-center positions.
            section_indices: Full-geometry indices for the BPM section subset,
                required when kinematics contains more sections than BPM.
        '''
        self.device = torch.device(device)
        self.dtype = torch.float64

        # Core arrays (move to device)
        self.frequencies = torch.as_tensor(
            frequencies, dtype=self.dtype, device=self.device
        )
        self.r = torch.as_tensor(r, dtype=self.dtype, device=self.device)
        self.dr = torch.as_tensor(dr, dtype=self.dtype, device=self.device)
        self.chord = torch.as_tensor(chord, dtype=self.dtype, device=self.device)
        self.c1 = (
            0.25 * self.chord
            if c1 is None
            else torch.as_tensor(c1, dtype=self.dtype, device=self.device)
        )
        self.alpha = torch.as_tensor(alpha.copy(), dtype=self.dtype, device=self.device)
        self.vi = torch.as_tensor(vi.copy(), dtype=self.dtype, device=self.device)
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
        self.kinematics = kinematics
        self.section_indices = (
            None
            if section_indices is None
            else torch.as_tensor(
                section_indices,
                dtype=torch.long,
                device=self.device,
            )
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

    def _source_blade_angles(self) -> torch.Tensor:
        """Return source-time blade azimuths from Kinematics or BPM inputs."""
        if self.kinematics is None:
            return self.omega * self.tau[:, None] + self.blade_angles[None, :]

        source_times = torch.as_tensor(
            self.kinematics.source_times,
            dtype=self.dtype,
            device=self.device,
        )
        if source_times.shape[0] < self.nt or not torch.allclose(
            source_times[: self.nt],
            self.tau,
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(
                "The supplied Kinematics source times do not match BPM src_times."
            )
        return torch.as_tensor(
            self.kinematics.blade_angles[: self.nt],
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def generate_trajectory_and_basis(self) -> None:
        """Generate Julia-style BPM trailing-edge source positions.

        BroadbandBPM.jl assumes the rotor axis is its global y-axis and rotates
        source points in the x-z plane.  FastBEMT's Kinematics rotates about
        global x.  The mapping used here is: Julia x -> FastBEMT z, Julia y
        -> FastBEMT x, and Julia z -> FastBEMT y.
        """
        beta = self._source_blade_angles()
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        te_offset = self.chord - self.c1

        x_fast = torch.zeros(
            (self.nt, self.ns, self.nb),
            dtype=self.dtype,
            device=self.device,
        )
        y_fast = (
            cos_beta[:, None, :] * self.r[None, :, None]
            + sin_beta[:, None, :] * te_offset[None, :, None]
        )
        z_fast = (
            sin_beta[:, None, :] * self.r[None, :, None]
            - cos_beta[:, None, :] * te_offset[None, :, None]
        )

        self.pos_fixed_TE = torch.stack([x_fast, y_fast, z_fast], dim=-1)
        self.beta_fixed = beta[:, None, :, None].expand(-1, self.ns, -1, 1).contiguous()


    @torch.inference_mode()
    def get_emission_geometry(
        self, r_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute acoustic directivity factors and distances to observer.

        Calculates the spatial relationship between blade elements and observer
        with the same axis convention as BroadbandBPM.jl's observer_location.

        Args:
            r_obs: Observer positions of shape (n_obs, 3) in global frame (x, y, z).

        Returns:
            Tuple of four tensors, each shape (time, section, blade, n_obs):
                - r_mag: Distance from source to observer (m)
                - dh_te: High-frequency directivity factor (dimensionless)
                - dh_le: Alias of dh_te for compatibility with component APIs
                - dl: Low-frequency directivity factor (dimensionless)
        """
        r_vec_TE = r_obs[None, None, None, :, :] - self.pos_fixed_TE
        xe_d = r_vec_TE[..., 2]
        ye_d = r_vec_TE[..., 0]
        ze_d = r_vec_TE[..., 1]

        beta = self.beta_fixed.squeeze(-1)
        theta_rot = torch.pi - beta
        cos_theta = torch.cos(theta_rot)
        sin_theta = torch.sin(theta_rot)
        xe = cos_theta * xe_d + sin_theta * ze_d
        ze = -sin_theta * xe_d + cos_theta * ze_d

        r_mag_TE = torch.sqrt(ye_d**2 + xe**2 + ze**2)
        theta_TE = torch.atan2(torch.sqrt(ye_d**2 + ze**2), xe)
        phi_TE = torch.atan2(ye_d, ze)

        phi_abs = torch.abs(phi_TE)
        phi_sign = torch.where(
            phi_TE >= 0.0,
            torch.ones_like(phi_TE),
            -torch.ones_like(phi_TE),
        )
        phi_abs_deg = torch.rad2deg(phi_abs)
        phi_small = phi_sign * torch.deg2rad(0.1 * phi_abs_deg**2 + 2.5)
        phi_large = phi_sign * torch.deg2rad(
            -0.1 * (phi_abs_deg - 180.0) ** 2 + 177.5
        )
        phi_TE = torch.where(
            phi_abs < torch.deg2rad(torch.tensor(5.0, dtype=self.dtype, device=self.device)),
            phi_small,
            torch.where(
                phi_abs > torch.deg2rad(torch.tensor(175.0, dtype=self.dtype, device=self.device)),
                phi_large,
                phi_TE,
            ),
        )

        m = self.m[None, :, None, None] # (1, ns, 1, 1)

        dh_te = (2 * torch.sin(theta_TE / 2)**2 * torch.sin(phi_TE)**2) / (
            (1.0 + m * torch.cos(theta_TE)) * (1.0 + 0.2 * m * torch.cos(theta_TE)) ** 2
        )

        dl = (torch.sin(theta_TE)**2 * torch.sin(phi_TE)**2) / (1.0 + m * torch.cos(theta_TE))**4

        return r_mag_TE, dh_te, dh_te, dl

    @torch.inference_mode()
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
        self.pos_fixed_TE = self.interpolate_positions(
            bpm_output_times, bpm_obs_times, self.pos_fixed_TE
        )
        self.beta_fixed = self.interpolate_positions(
            bpm_output_times, bpm_obs_times, self.beta_fixed
        )

        # 1. Geometry and Base Values at Emission Time
        r_mag, dh_te, dh_le, dl = self.get_emission_geometry(r_obs)

        m5_dr = (self.m**5 * self.dr)[None, :, None, None]
        r_mag_sq = r_mag**2
        self.base_val_te = (m5_dr * dh_te) / r_mag_sq
        self.base_val_le = (m5_dr * dh_le) / r_mag_sq
        self.base_val_low = (m5_dr * dl) / r_mag_sq

        # 2. Compute raw SPP tensors and collapse section/blade dimensions.
        components_raw = self.compute_noise_components(
            lt=lt,
            i=i,
            alpha_stall=alpha_stall,
        )
        return {
            name: component.sum(dim=(2, 3))
            for name, component in components_raw.items()
        }

    @torch.inference_mode()
    def compute_noise_components(
        self,
        lt: float,
        i: float,
        alpha_stall: float = 15.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute all BPM component tensors before section/blade summation."""
        return {
            "tbl": self.tbl_noise(alpha_stall=alpha_stall),
            "lbl": self.lbl_noise(),
            "teb": self.teb_noise(),
            "ti": self.ti_noise(lt=lt, i=i),
            "tv": self.tv_noise(),
        }

    @torch.inference_mode()
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
        return _ti_component.compute_ti_noise(
            frequencies=self.frequencies,
            chord=self.chord,
            alpha=self.alpha,
            u=self.u,
            m=self.m,
            rho=self.rho,
            a_inf=self.a_inf,
            base_val_le=self.base_val_le,
            base_val_low=self.base_val_low,
            lt=lt,
            i=i,
        )

    @torch.inference_mode()
    def teb_noise(self, h: Optional[float] = None) -> torch.Tensor:
        """Compute trailing edge bluntness (TEB) broadband noise.

        Models the scattering of incoming vorticity by blunt trailing edges,
        including effects of boat tail angle and boundary layer thickness.

        Args:
            h: Trailing edge thickness (m). If None, defaults to 1% of chord length.

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """
        return _teb_component.compute_teb_noise(
            frequencies=self.frequencies,
            chord=self.chord,
            u=self.u,
            m=self.m,
            delta_p=self.delta_p,
            delta_s=self.delta_s,
            psi=self.psi,
            base_val_te=self.base_val_te,
            h=h,
        )

    @torch.inference_mode()
    def lbl_noise(self) -> torch.Tensor:
        """Compute laminar boundary layer (LBL) broadband noise.

        Models noise from instabilities in laminar boundary layers, which occurs
        at low frequencies and low to moderate angles of attack.

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """
        return _lbl_component.compute_lbl_noise(
            frequencies=self.frequencies,
            alpha=self.alpha,
            u=self.u,
            re_c=self.re_c,
            delta_p=self.delta_p,
            base_val_le=self.base_val_le,
        )

    @torch.inference_mode()
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

        return _tbl_component.compute_tbl_noise(
            frequencies=self.frequencies,
            chord=self.chord,
            alpha=self.alpha,
            u=self.u,
            re_c=self.re_c,
            m=self.m,
            delta_p=self.delta_p,
            delta_s=self.delta_s,
            base_val_te=self.base_val_te,
            base_val_low=self.base_val_low,
            alpha_stall=alpha_stall,
        )

    @torch.inference_mode()
    def tv_noise(self) -> torch.Tensor:
        """Compute tip vortex (TV) broadband noise.

        Models the noise from unsteady loading fluctuations induced by the tip vortex.
        Computed only at the tip section, with zeros elsewhere.

        Returns:
            Raw spectral power density tensor of shape (n_freq, nt, ns, nb, n_obs).
        """
        return _tv_component.compute_tv_noise(
            frequencies=self.frequencies,
            r=self.r,
            dr=self.dr,
            chord=self.chord,
            alpha=self.alpha,
            m=self.m,
            a_inf=self.a_inf,
            base_val_te=self.base_val_te,
        )

    @torch.inference_mode()
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

        coord_count = y_old.shape[-1]

        # Prepare y_old for broadcasting.
        y_old_p = y_old.permute(3, 1, 2, 0).contiguous()  # (coord, n_sec, n_b, nt)

        # Expand y to add n_obs, then gather bracketing points.
        y_old_exp = y_old_p.unsqueeze(1).expand(-1, n_obs, -1, -1, -1)

        # Expand idx to match coordinate dimension.
        idx_exp = idx.unsqueeze(0).expand(coord_count, -1, -1, -1, -1)

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
