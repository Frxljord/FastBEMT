"""Auxiliary correlations for BPM trailing-edge bluntness noise."""

import torch

from FastBEMT.Aeroacoustics._bpm_common import (
    safe_log10,
    st,
    torch_select as _torch_select,
)


def st_peak_3prime(q: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute peak Strouhal number for trailing-edge bluntness noise."""
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
    """Compute the TEB bluntness-amplitude correction."""
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
    """Compute the mu parameter for trailing-edge bluntness noise."""
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
    """Compute the m parameter for g5 in trailing-edge bluntness noise."""
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
    """Compute the eta0 parameter for g5."""
    return -torch.sqrt(m**2 * mu**4 / (6.25 + m**2 * mu**2))


def calc_k(m: torch.Tensor, mu: torch.Tensor, eta0: torch.Tensor) -> torch.Tensor:
    """Compute the k parameter for g5."""
    return 2.5 * torch.sqrt(1 - (eta0 / mu) ** 2) - 2.5 - m * eta0


def g5(
    m: torch.Tensor,
    mu: torch.Tensor,
    eta: torch.Tensor,
    eta0: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Compute the TEB spectral-shape correction."""
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
    """Compute the g5_0 reference function for TEB noise."""
    q_0 = 6.724 * q**2 - 4.019 * q + 1.107
    m = calc_m(q_0)
    mu = calc_mu(q_0)
    eta0 = calc_eta0(m, mu)
    k = calc_k(m, mu, eta0)
    return g5(m, mu, eta, eta0, k)


def g5_tot(q: torch.Tensor, eta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute total g5 with the bounded boat-tail angle correction."""
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


@torch.inference_mode()
def compute_teb_noise(
    frequencies: torch.Tensor,
    chord: torch.Tensor,
    u: torch.Tensor,
    m: torch.Tensor,
    delta_p: torch.Tensor,
    delta_s: torch.Tensor,
    psi: torch.Tensor,
    base_val_te: torch.Tensor,
    h: float | None = None,
) -> torch.Tensor:
    """Compute trailing-edge bluntness broadband noise."""
    h_val = (chord * 0.01) if h is None else torch.full_like(chord, h)

    delta_avg = (delta_p + delta_s) * 0.5
    h_5d = h_val[None, None, :, None, None]
    delta_avg_5d = delta_avg[None, None, :, None, None]
    psi_5d = psi[None, None, :, None, None]
    m_5d = m[None, None, :, None, None]

    q = h_5d / delta_avg_5d

    st_3p = st(frequencies, h_val, u)[:, None, :, None, None]
    st_3p_pk = st_peak_3prime(q, psi_5d)
    eta = torch.log10(st_3p / st_3p_pk)

    bv_te = base_val_te[None, :, :, :, :]
    log_h_bv = 10 * safe_log10(h_5d * bv_te * torch.sqrt(m_5d))

    spl_teb = log_h_bv + g4(q, psi_5d) + g5_tot(q, eta, psi_5d)

    return 10 ** (spl_teb / 10)


__all__ = [
    "compute_teb_noise",
    "st",
    "safe_log10",
    "st_peak_3prime",
    "g4",
    "calc_mu",
    "calc_m",
    "calc_eta0",
    "calc_k",
    "g5",
    "g5_0",
    "g5_tot",
]
