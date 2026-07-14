"""Auxiliary correlations for BPM turbulent boundary-layer noise."""

import torch

from .._bpm_common import (
    safe_log10,
    st,
    torch_select as _torch_select,
)


def st1(m: torch.Tensor) -> torch.Tensor:
    """Compute st1 parameter: 0.02 * m^(-0.6)."""
    return 0.02 * torch.pow(m, -0.6)


def st2(m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute st2 parameter based on Mach number and angle of attack."""
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


def tbl_te_a_min(a: torch.Tensor) -> torch.Tensor:
    """Compute minimum TBL trailing edge correction for the A component."""
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
    """Compute maximum TBL trailing edge correction for the A component."""
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
    """Compute reference TBL A parameter based on chord Reynolds number."""
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
    """Compute TBL A adjustment ratio."""
    return (-20 - tbl_te_a_min(a0)) / (tbl_te_a_max(a0) - tbl_te_a_min(a0))


def tbl_te_a(a: torch.Tensor, re_c: torch.Tensor) -> torch.Tensor:
    """Compute TBL trailing edge correction for the A component."""
    a0 = tbl_te_a0(re_c)
    return tbl_te_a_min(a) + tbl_te_ar(a0) * (tbl_te_a_max(a) - tbl_te_a_min(a))


def tbl_te_b_min(b: torch.Tensor) -> torch.Tensor:
    """Compute minimum TBL trailing edge correction for the B component."""
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
    """Compute maximum TBL trailing edge correction for the B component."""
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
    """Compute reference TBL B parameter based on chord Reynolds number."""
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
    """Compute TBL B adjustment ratio."""
    return (-20 - tbl_te_b_min(b0)) / (tbl_te_b_max(b0) - tbl_te_b_min(b0))


def tbl_te_b(b: torch.Tensor, re_c: torch.Tensor) -> torch.Tensor:
    """Compute TBL trailing edge correction for the B component."""
    b0 = tbl_te_b0(re_c)
    return tbl_te_b_min(b) + tbl_te_br(b0) * (tbl_te_b_max(b) - tbl_te_b_min(b))


def k1(re_c: torch.Tensor) -> torch.Tensor:
    """Compute k1 parameter based on chord Reynolds number."""
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
    """Compute delta_k1 pressure-side correction."""
    return torch.where(
        re_dp <= 5000,
        alpha * (1.43 * torch.log10(re_dp) - 5.29),
        torch.zeros_like(alpha),
    )


def k2(re_c: torch.Tensor, m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute k2 parameter for separated-flow TBL noise."""
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


@torch.inference_mode()
def compute_tbl_noise(
    frequencies: torch.Tensor,
    chord: torch.Tensor,
    alpha: torch.Tensor,
    u: torch.Tensor,
    re_c: torch.Tensor,
    m: torch.Tensor,
    delta_p: torch.Tensor,
    delta_s: torch.Tensor,
    base_val_te: torch.Tensor,
    base_val_low: torch.Tensor,
) -> torch.Tensor:
    """Compute turbulent boundary-layer broadband noise."""
    m_5d = m[None, None, :, None, None]
    alpha_5d = alpha[None, None, :, None, None]
    re_c_5d = re_c[None, None, :, None, None]
    delta_p_5d = delta_p[None, None, :, None, None]
    delta_s_5d = delta_s[None, None, :, None, None]
    chord_5d = chord[None, None, :, None, None]

    st_1 = st1(m_5d)
    st_2 = st2(m_5d, alpha_5d)
    k_1 = k1(re_c_5d)
    k_2 = k2(re_c_5d, m_5d, alpha_5d)

    st_p = st(frequencies, delta_p, u)[:, None, :, None, None]
    st_s = st(frequencies, delta_s, u)[:, None, :, None, None]

    st_1_bar = (st_1 + st_2) * 0.5

    log_st_s_st1 = torch.abs(torch.log10(st_s / st_1_bar))
    log_st_p_st1 = torch.abs(torch.log10(st_p / st_1))
    log_st_s_st2 = torch.abs(torch.log10(st_s / st_2))

    as_val = tbl_te_a(log_st_s_st1, re_c_5d)
    ap = tbl_te_a(log_st_p_st1, re_c_5d)
    b = tbl_te_b(log_st_s_st2, re_c_5d)
    a_prime = tbl_te_a(log_st_s_st2, 3 * re_c_5d)

    bv_te = base_val_te[None, :, :, :, :]
    bv_low = base_val_low[None, :, :, :, :]

    log_ds_bv_te = 10 * safe_log10(delta_s_5d * bv_te)
    log_dp_bv_te = 10 * safe_log10(delta_p_5d * bv_te)
    log_ds_bv_low = 10 * safe_log10(delta_s_5d * bv_low)
    log_dp_bv_low = 10 * safe_log10(delta_p_5d * bv_low)

    delta_k1_val = delta_k1((re_c_5d / chord_5d * delta_p_5d), alpha_5d)
    gamma0 = 23.430 * m_5d + 4.651
    alpha_stall_julia = torch.minimum(torch.full_like(gamma0, 12.5), gamma0)
    alpha_mask = alpha_5d < alpha_stall_julia

    spl_s = torch.where(alpha_mask, log_ds_bv_te + as_val + k_1 - 3, log_ds_bv_low)
    spl_p = torch.where(
        alpha_mask,
        log_dp_bv_te + ap + k_1 - 3 + delta_k1_val,
        log_dp_bv_low,
    )
    spl_a = torch.where(
        alpha_mask, log_ds_bv_te + b + k_2, log_ds_bv_low + a_prime + k_2
    )

    return 10 ** (spl_s / 10) + 10 ** (spl_p / 10) + 10 ** (spl_a / 10)
