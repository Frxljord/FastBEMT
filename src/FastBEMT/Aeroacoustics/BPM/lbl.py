"""Auxiliary correlations for BPM laminar boundary-layer noise."""

from typing import Sequence, Union

import torch


def _torch_select(
    conditions: Sequence[torch.Tensor],
    choices: Sequence[Union[torch.Tensor, float]],
    default_value: float = 0.0,
) -> torch.Tensor:
    """Torch-based conditional selection similar to np.select."""
    result = torch.full_like(conditions[0], default_value, dtype=conditions[0].dtype)
    for cond, choice in zip(reversed(conditions), reversed(choices)):
        if not isinstance(choice, torch.Tensor):
            choice = torch.tensor(choice, dtype=result.dtype, device=result.device)
        result = torch.where(cond, choice, result)
    return result


def st(f: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute Strouhal number: St = f * l / u."""
    return f[:, None] * l / u[None, :]


def safe_log10(value: torch.Tensor) -> torch.Tensor:
    """Compute log10 with a tiny clamp instead of additive bias."""
    return torch.log10(torch.clamp(value, min=torch.finfo(value.dtype).tiny))


def st1_prime(re_c: torch.Tensor) -> torch.Tensor:
    """Compute reference peak Strouhal number from chord Reynolds number."""
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
    """Compute peak Strouhal number with angle-of-attack correction."""
    return st1_prime(re_c) * torch.pow(10.0, -0.04 * alpha)


def g1(e: torch.Tensor) -> torch.Tensor:
    """Compute the LBL dimensionless frequency correction."""
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
    """Compute the LBL Reynolds-number correction."""
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
    """Compute reference Reynolds number from angle of attack."""
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
    """Compute the LBL angle-of-attack correction."""
    return 171.04 - 3.03 * alpha


@torch.inference_mode()
def compute_lbl_noise(
    frequencies: torch.Tensor,
    alpha: torch.Tensor,
    u: torch.Tensor,
    re_c: torch.Tensor,
    delta_p: torch.Tensor,
    base_val_le: torch.Tensor,
) -> torch.Tensor:
    """Compute laminar boundary-layer broadband noise."""
    delta_p_5d = delta_p[None, None, :, None, None]
    re_c_5d = re_c[None, None, :, None, None]
    alpha_5d = alpha[None, None, :, None, None]

    e = st(frequencies, delta_p, u)[:, None, :, None, None] / st_peak_prime(
        re_c_5d, alpha_5d
    )
    d = re_c_5d / re_c0(alpha_5d)

    bv_le = base_val_le[None, :, :, :, :]
    log_dp_bv = 10 * safe_log10(delta_p_5d * bv_le)
    spl_lbl = log_dp_bv + g1(e) + g2(d) + g3(alpha_5d)

    return 10 ** (spl_lbl / 10)


__all__ = [
    "compute_lbl_noise",
    "st",
    "safe_log10",
    "st1_prime",
    "st_peak_prime",
    "g1",
    "g2",
    "re_c0",
    "g3",
]
